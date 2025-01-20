//
// Created by Jacob Jennings on 1/12/25.
//

#ifndef CLARIFICATIONDATASET_H
#define CLARIFICATIONDATASET_H

#include <omp.h>

#include <algorithm>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <ranges>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <torch/torch.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
}

class ClarificationDataset final {
public:
    explicit ClarificationDataset(
        torch::Device device,
        const std::string &base_dir,
        const std::string &csv_filename,
        int num_preload_batches = 16,
        int batch_size = 16,
        int num_threads = 0
    ) : device_(device),
        num_preload_batches_(num_preload_batches),
        batch_size_(batch_size),
        num_threads_(num_threads),
        base_dir_(std::move(base_dir)),
        producer_thread_([this] { PreloadBatchesLoop(); }) {
        // Launch producer thread
        if (batch_size <= 0) {
            throw std::invalid_argument("batch_size must be > 0 in Batch mode");
        }

        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        }

        auto info_csv_path = std::filesystem::path(base_dir_) / "info.csv";
        std::ifstream csvfile(info_csv_path);

        av_log_set_level(AV_LOG_FATAL);

        const AVCodec *codec = nullptr;
        void *i = nullptr;
        while ((codec = av_codec_iterate(&i)) != nullptr) {
            if (std::string(codec->name).find("opus") != std::string::npos) {
                std::cout << "ffmpeg reports opus presence: " << codec->name << std::endl;
                break;
            }
        }
        if (codec == nullptr || std::string(codec->name).find("opus") == std::string::npos) {
            throw std::runtime_error("opus codec not found");
        }

        if (!csvfile.is_open()) {
            throw std::runtime_error("Failed to open info.csv");
        }

        std::string line;

        std::getline(csvfile, line); // Skip header

        std::getline(csvfile, line);
        std::erase(line, '\n');
        std::erase(line, '\r');

        std::stringstream ss(line);
        std::string value;
        std::getline(ss, value, ',');
        sample_rate_ = std::stoi(value);
        std::getline(ss, value, ',');
        sample_size = std::stoi(value);
        std::getline(ss, value, ',');
        overlap_size_ = std::stoi(value);

        std::cout << "ClarificationDataset initialized with sample_rate: " << sample_rate_ << ", sample_size: " <<
                sample_size << ", overlap_size_: " << overlap_size_ << std::endl;

        auto samples_csv_path = std::filesystem::path(base_dir_) / csv_filename;
        std::cout << "Path to samples.csv: " << samples_csv_path << std::endl;
        std::ifstream samples_csv(samples_csv_path);

        if (!samples_csv.is_open()) {
            throw std::runtime_error("Failed to open samples CSV file");
        }

        std::getline(samples_csv, line); // Skip header
        while (std::getline(samples_csv, line)) {
            std::stringstream strs(line);
            std::string path;
            std::getline(strs, path, ',');
            std::erase(path, '\n');
            std::erase(path, '\r');
            sample_infos_.push_back({path});
        }
    }

    ~ClarificationDataset() { {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            quit_ = true;
        }
        preload_cv_.notify_all();
        producer_thread_.join();
    }

    torch::Tensor next() {
        torch::NoGradGuard no_grad;

        std::unique_lock<std::mutex> lock(queue_mutex_);
        preload_cv_.wait(lock, [this] { return !preloaded_batches_.empty() || quit_; });

        if (preloaded_batches_.empty()) {
            throw std::out_of_range("End of dataset reached");
        }

        const auto batch_tensor = preloaded_batches_.front();
        preloaded_batches_.pop();
        lock.unlock();
        preload_cv_.notify_one(); // Notify that a batch has been consumed

        return batch_tensor.view({batch_size_, 2, sample_size});
    }

    int sample_size = 0;
    int file_idx = 0;

private:
    std::string base_dir_;
    torch::Device device_;
    int sample_rate_ = 0;
    int overlap_size_ = 0;
    int num_preload_batches_ = 0;
    int batch_size_ = 0;
    int num_threads_ = 0;

    struct SampleInfo {
        std::string path;
    };

    std::vector<SampleInfo> sample_infos_;
    std::queue<torch::Tensor> preloaded_batches_;
    std::mutex queue_mutex_;
    std::condition_variable preload_cv_;
    std::atomic<bool> quit_{false};
    std::thread producer_thread_;

    void PreloadBatchesLoop() {
        while (!quit_) {
            int num_batches_to_preload; {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                num_batches_to_preload = num_preload_batches_ - preloaded_batches_.size();
            }

            if (num_batches_to_preload > 0) {
                PreloadBatches(num_batches_to_preload);
            }

            preload_cv_.notify_all();

            // Sleep if no preloading is needed
            if (num_batches_to_preload <= 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }

    void PreloadBatches(int num_batches_to_preload) {
        std::vector<torch::Tensor> new_frames_vector;
        int current_batch_frames = 0;

        // Use a local file index to avoid race conditions
        int local_file_idx = file_idx;

        // Use OpenMP to parallelize audio file processing
#pragma omp parallel num_threads(num_threads_) shared(new_frames_vector, current_batch_frames, local_file_idx)
        {
            std::vector<torch::Tensor> local_frames_vector;
            int local_batch_frames = 0;

#pragma omp for nowait
            for (int i = 0; i < num_batches_to_preload * batch_size_; ++i) {
                if (local_file_idx >= sample_infos_.size()) {
                    continue;
                }

                const auto &[path] = sample_infos_[local_file_idx];
                const std::filesystem::path absolute_path = std::filesystem::path(base_dir_) / path;
                const auto absolute_path_string = absolute_path.string();

                torch::Tensor audio_samples = ProcessAudioFile(absolute_path_string);

                local_frames_vector.push_back(audio_samples);
                local_batch_frames += audio_samples.size(0);

                // Increment the file index safely
#pragma omp critical
                {
                    file_idx++;
                    local_file_idx = file_idx;
                }
            }

            // Gather the results from each thread
#pragma omp critical
            {
                new_frames_vector.insert(new_frames_vector.end(), local_frames_vector.begin(),
                                         local_frames_vector.end());
                current_batch_frames += local_batch_frames;
            }
        }

        // If no new frames were loaded, return early
        if (new_frames_vector.empty()) {
            return;
        }

        {
            // Concatenate tensors along dimension 0
            const torch::Tensor new_frames_tensor = torch::cat(new_frames_vector, 0);

            // Split the new_frames_tensor into batches
            const int num_frames = new_frames_tensor.size(0);
            const int num_full_batches = num_frames / batch_size_;
            std::lock_guard<std::mutex> lock(queue_mutex_);
            for (int i = 0; i < num_full_batches; ++i) {
                const int start_frame = i * batch_size_;
                const int end_frame = start_frame + batch_size_; // end_frame is now exclusive

                // No need to check start_frame < end_frame, as we are only iterating through full batches
                torch::Tensor batch_tensor = new_frames_tensor.narrow(0, start_frame, batch_size_);
                preloaded_batches_.push(batch_tensor.view({batch_size_, 2, sample_size}));
            }
        }
        std::cout << "Preloaded batches sizes: " << preloaded_batches_.size() << std::endl;
    }

    // Helper function to get the number of audio frames in a file
    [[nodiscard]] int GetNumFramesInAudioFile(const std::string &absolute_path_string) const {
        AVFormatContext *format_context = avformat_alloc_context();
        if (format_context == nullptr) {
            return -1; // Indicate an error
        }

        if (avformat_open_input(&format_context, absolute_path_string.c_str(), nullptr, nullptr) != 0) {
            avformat_free_context(format_context);
            return -1; // Error opening file
        }

        if (avformat_find_stream_info(format_context, nullptr) < 0) {
            avformat_close_input(&format_context);
            return -1; // Error finding stream info
        }

        int audio_stream_index = -1;
        for (unsigned int i = 0; i < format_context->nb_streams; i++) {
            if (format_context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
                audio_stream_index = i;
                break;
            }
        }

        if (audio_stream_index == -1) {
            avformat_close_input(&format_context);
            return -1; // No audio stream found
        }

        const AVStream *audio_stream = format_context->streams[audio_stream_index];
        int num_frames = audio_stream->nb_frames;
        if (num_frames == 0) {
            // Estimate the number of frames based on duration and frame rate
            if (audio_stream->duration != AV_NOPTS_VALUE && audio_stream->r_frame_rate.num != 0 && audio_stream->
                r_frame_rate.den != 0) {
                const double duration_in_seconds = static_cast<double>(audio_stream->duration) * av_q2d(
                                                       audio_stream->time_base);
                const double frame_rate = av_q2d(audio_stream->r_frame_rate);
                num_frames = static_cast<int>(duration_in_seconds * frame_rate);
            }
        }

        avformat_close_input(&format_context);
        return num_frames / sample_size;
    }

    [[nodiscard]] torch::Tensor ProcessAudioFile(const std::string &absolute_path_string) const {
        // Use ffmpeg to load the opus file at absolute_path at sample_rate into a tensor named audio.
        AVFormatContext *format_context = avformat_alloc_context();

        if (const AVInputFormat *input_format = av_find_input_format("ogg"); input_format == nullptr) {
            throw std::runtime_error("Input format ogg not found");
        }

        if (const auto open_input_error = avformat_open_input(&format_context, absolute_path_string.c_str(),
                                                              nullptr, nullptr); open_input_error != 0) {
            char error_buf[256] = {0};
            const auto error_string = std::string(
                av_make_error_string(error_buf, sizeof(error_buf), open_input_error));
            std::cout << "Error opening input: " << error_string << std::endl;
            std::string error_message = "Failed to open audio file ";
            error_message += absolute_path_string;
            error_message += ", msg: ";
            error_message += error_string;
            throw std::runtime_error(error_message);
        }

        if (avformat_find_stream_info(format_context, nullptr) < 0) {
            throw std::runtime_error("Failed to retrieve stream info");
        }

        // Find the audio stream
        int audio_stream_index = -1;
        const AVCodecParameters *codec_params = nullptr;
        for (unsigned int stream_idx = 0; stream_idx < format_context->nb_streams; ++stream_idx) {
            codec_params = format_context->streams[stream_idx]->codecpar;
            if (codec_params->codec_type == AVMEDIA_TYPE_AUDIO) {
                audio_stream_index = stream_idx;
                break;
            }
        }
        if (audio_stream_index == -1) {
            throw std::runtime_error("Failed to find audio stream");
        }

        // Find and open the libopus decoder
        const AVCodec *codec = avcodec_find_decoder_by_name("libopus");
        if (codec == nullptr) {
            throw std::runtime_error("Failed to find libopus codec");
        }
        AVCodecContext *codec_context = avcodec_alloc_context3(codec);
        if (codec_context == nullptr) {
            throw std::runtime_error("Failed to allocate codec context");
        }
        if (avcodec_parameters_to_context(codec_context, codec_params) < 0) {
            throw std::runtime_error("Failed to copy codec parameters to context");
        }

        // Set the desired output sample rate
        codec_context->sample_rate = sample_rate_;
        codec_context->request_sample_fmt = AV_SAMPLE_FMT_FLT;
        if (avcodec_open2(codec_context, codec, nullptr) < 0) {
            throw std::runtime_error("Failed to open codec");
        }

        // Confirm the sample rate and throw exception if it does not match
        if (codec_context->sample_rate != sample_rate_) {
            throw std::runtime_error("Sample rate mismatch");
        }

        // Decode frames
        AVPacket *packet = av_packet_alloc();
        AVFrame *frame = av_frame_alloc();

        // Use a map to store audio data for each channel separately
        std::map<int, std::vector<float> > audio_data_map;

        while (av_read_frame(format_context, packet) >= 0) {
            if (packet->stream_index == audio_stream_index) {
                if (avcodec_send_packet(codec_context, packet) == 0) {
                    while (avcodec_receive_frame(codec_context, frame) == 0) {
                        // Determine if the frame is planar
                        const bool is_planar = av_sample_fmt_is_planar(static_cast<AVSampleFormat>(frame->format));
                        const int num_channels = codec_context->ch_layout.nb_channels;
                        const int num_samples = frame->nb_samples;

                        if (is_planar) {
                            // Planar format: Data for each channel is in a separate plane.
                            for (int ch = 0; ch < num_channels; ++ch) {
                                const float *channel_data = reinterpret_cast<const float *>(frame->extended_data[ch]);
                                for (int i = 0; i < num_samples; ++i) {
                                    audio_data_map[ch].push_back(channel_data[i]);
                                }
                            }
                        } else {
                            // Non-planar format: Data for all channels is interleaved in a single plane.
                            const float *interleaved_data = reinterpret_cast<const float *>(frame->extended_data[0]);
                            for (int i = 0; i < num_samples; ++i) {
                                for (int ch = 0; ch < num_channels; ++ch) {
                                    audio_data_map[ch].push_back(interleaved_data[i * num_channels + ch]);
                                }
                            }
                        }
                    }
                }
            }
            av_packet_unref(packet);
        }

        // Flush the decoder
        if (avcodec_send_packet(codec_context, nullptr) != 0) {
            throw std::runtime_error("Error flushing the decoder");
        }
        while (avcodec_receive_frame(codec_context, frame) == 0) {
            const bool is_planar = av_sample_fmt_is_planar(static_cast<AVSampleFormat>(frame->format));
            const int num_channels = codec_context->ch_layout.nb_channels;
            const int num_samples = frame->nb_samples;

            if (is_planar) {
                for (int ch = 0; ch < num_channels; ++ch) {
                    const float *channel_data = reinterpret_cast<const float *>(frame->extended_data[ch]);
                    for (int i = 0; i < num_samples; ++i) {
                        audio_data_map[ch].push_back(channel_data[i]);
                    }
                }
            } else {
                const float *interleaved_data = reinterpret_cast<const float *>(frame->extended_data[0]);
                for (int i = 0; i < num_samples; ++i) {
                    for (int ch = 0; ch < num_channels; ++ch) {
                        audio_data_map[ch].push_back(interleaved_data[i * num_channels + ch]);
                    }
                }
            }
        }

        // Clean up
        av_frame_free(&frame);
        av_packet_free(&packet);
        avcodec_free_context(&codec_context);
        avformat_close_input(&format_context);

        // Create tensors for each channel from the map using std::views::values
        std::vector<torch::Tensor> audio_tensors;
        for (const auto &audio_data: audio_data_map | std::views::values) {
            auto audio_tensor = torch::from_blob(
                const_cast<float *>(audio_data.data()),
                {static_cast<long>(audio_data.size())},
                torch::TensorOptions().dtype(torch::kFloat)
            ).clone().to(device_);
            audio_tensors.push_back(audio_tensor);
        }

        // Verify that we have data for all channels
        if (audio_tensors.size() != 2) {
            throw std::runtime_error("Could not retrieve data for both channels.");
        }

        // Combine channels into a single tensor [1, channels, samples]
        const auto audio_combined = torch::stack(audio_tensors, 0).unsqueeze(0);

        const auto total_size = audio_combined.size(2);
        const auto truncated_size = (total_size / sample_size) * sample_size;

        // Truncate the tensor
        const auto truncated_audio = audio_combined.narrow(2, 0, truncated_size);

        // Split and stack into [frames, channels, samples]
        auto audio_samples = torch::stack(truncated_audio.split(sample_size, /*dim=*/2)).squeeze(1);

        return audio_samples;
    }
};

#endif // CLARIFICATIONDATASET_H
