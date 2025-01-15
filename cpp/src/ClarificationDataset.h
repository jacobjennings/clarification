//
// Created by jacob on 1/12/25.
//

#ifndef CLARIFICATIONDATASET_H
#define CLARIFICATIONDATASET_H


#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <torch/torch.h>
#include <locale>
#include <codecvt>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
}

class ClarificationDataset final : public torch::data::Dataset<ClarificationDataset, torch::Tensor> {
public:
    ClarificationDataset(
        size_t batch_size,
        torch::Device device,
        const std::string &base_dir,
        const std::string &csv_filename
    ) : base_dir(base_dir), device(device) {
        auto info_csv_path = std::filesystem::path(base_dir) / "info.csv";
        std::ifstream csvfile(info_csv_path);

        av_log_set_level(AV_LOG_FATAL);

        const AVCodec *codec = nullptr;
        void *iter = nullptr;
        while ((codec = av_codec_iterate(&iter))) {
            if (std::string(codec->name).find("opus") != std::string::npos) {
                std::cout << "ffmpeg reports opus presence: " << codec->name << std::endl;
                break;
            }
        }
        if (!codec || std::string(codec->name).find("opus") == std::string::npos) {
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
        sample_rate = std::stoi(value);
        std::getline(ss, value, ',');
        sample_size = std::stoi(value);
        std::getline(ss, value, ',');
        overlap_size = std::stoi(value);
        std::getline(ss, value, ',');
        consumption_batch_size = std::stoi(value);

        this->batch_size = batch_size;
        if (this->batch_size % this->consumption_batch_size != 0) {
            throw std::runtime_error("batch_size must be a multiple of consumption_batch_size");
        }

        consumption_batches_multiplier = this->batch_size / this->consumption_batch_size;

        std::cout << "ClarificationDataset initialized with sample_rate: " << sample_rate << ", sample_size: " <<
        sample_size << ", overlap_size: " << overlap_size << ", consumption_batch_size: " <<
        consumption_batch_size << ", consumption_batches_multiplier: " << consumption_batches_multiplier << std::endl;

        auto samples_csv_path = std::filesystem::path(base_dir) / csv_filename;
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
            sample_infos.push_back({path});
        }
    }

    torch::Tensor get(const size_t batch_idx) override {
        torch::NoGradGuard no_grad;
        // TODO calculate
        constexpr auto expected_size = 7200;
        torch::Tensor audio_aggregated;
        for (long i = 0; i < consumption_batches_multiplier; ++i) {
            const long consumption_batch_idx = batch_idx * consumption_batches_multiplier + i;
            const auto &sample_info = sample_infos[consumption_batch_idx];
            const std::filesystem::path absolute_path = std::filesystem::path(base_dir) / sample_info.path;
            auto absolute_path_string = absolute_path.string();

            // Use ffmpeg to load the opus file at absolute_path at sample_rate into a tensor named audio.
            AVFormatContext *format_context = avformat_alloc_context();

            if (const AVInputFormat *input_format = av_find_input_format("ogg"); !input_format) {
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
            if (!codec) {
                throw std::runtime_error("Failed to find libopus codec");
            }
            AVCodecContext *codec_context = avcodec_alloc_context3(codec);
            if (!codec_context) {
                throw std::runtime_error("Failed to allocate codec context");
            }
            if (avcodec_parameters_to_context(codec_context, codec_params) < 0) {
                throw std::runtime_error("Failed to copy codec parameters to context");
            }

            // Set the desired output sample rate
            codec_context->sample_rate = sample_rate;
            codec_context->request_sample_fmt = AV_SAMPLE_FMT_FLT;
            if (avcodec_open2(codec_context, codec, nullptr) < 0) {
                throw std::runtime_error("Failed to open codec");
            }

            // Confirm the sample rate and throw exception if it does not match (not my ffmpeg fork)
            if (codec_context->sample_rate != sample_rate) {
                throw std::runtime_error("Sample rate mismatch");
            }

            // Decode frames
            AVPacket *packet = av_packet_alloc();
            AVFrame *frame = av_frame_alloc();
            std::vector<float> audio_data;
            while (av_read_frame(format_context, packet) >= 0) {
                if (packet->stream_index == audio_stream_index) {
                    if (avcodec_send_packet(codec_context, packet) == 0) {
                        decode_audio_frames(codec_context, packet, frame, audio_data);
                    }
                }
                av_packet_unref(packet);
            }

            // Flush the decoder
            avcodec_send_packet(codec_context, nullptr);
            while (avcodec_receive_frame(codec_context, frame) == 0) {
                avcodec_send_packet(codec_context, nullptr);
                decode_audio_frames(codec_context, packet, frame, audio_data);
            }

            // Clean up
            av_frame_free(&frame);
            av_packet_free(&packet);
            avcodec_free_context(&codec_context);
            avformat_close_input(&format_context);

            // Create the tensor
            const auto audio = torch::from_blob(audio_data.data(), {static_cast<long>(audio_data.size())},
                                                torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU));
            if (i == 0) {
                // initialize with cpu device

                audio_aggregated = torch::empty({consumption_batches_multiplier, static_cast<long>(audio_data.size())},
                                                torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU));
            }
            audio_aggregated[i] = audio;
        }

        // std::cout << "Device: " << device << std::endl;
        // audio_aggregated = audio_aggregated.to(device);

        const auto total_size = audio_aggregated.size(1);
        const auto truncated_size = (total_size / sample_size) * sample_size; // Largest multiple of sample_size

        // Truncate the tensor
        const auto truncated_audio = audio_aggregated.narrow(1, 0, truncated_size);

        // Split and stack
        auto audio_samples = torch::stack(truncated_audio.split(sample_size, /*dim=*/1));

        return audio_samples;
    }

    static void check_bad_chars(const std::string &str) {
        for (const char ch: str) {
            if (!std::isalnum(ch) && ch != '/' && ch != '\\' && ch != '-' && ch != '_' && ch != '.') {
                std::cout << "Invalid character code in str: " << static_cast<int>(ch) << " at index " << str.find(ch)
                        << ". Length: " << std::to_string(str.length()) << std::endl;
            }
        }
    }

    static void decode_audio_frames(AVCodecContext *codec_context, AVPacket *packet, AVFrame *frame,
                                    std::vector<float> &audio_data) {
        while (avcodec_receive_frame(codec_context, frame) == 0) {
            if (frame->format == AV_SAMPLE_FMT_FLTP) {
                for (int i = 0; i < frame->nb_samples; ++i) {
                    for (int ch = 0; ch < codec_context->ch_layout.nb_channels; ++ch) {
                        const auto *data = (float *) frame->data[ch];
                        audio_data.push_back(data[i]);
                    }
                }
            } else if (frame->format == AV_SAMPLE_FMT_FLT) {
                const auto *data = (float *) frame->data[0];
                for (int i = 0; i < frame->nb_samples * codec_context->ch_layout.nb_channels; ++i) {
                    audio_data.push_back(data[i]);
                }
            } else {
                throw std::runtime_error("Unsupported audio format");
            }
        }
    }

    [[nodiscard]] torch::optional<size_t> size() const override {
        return sample_infos.size() / consumption_batches_multiplier;
    }

private:
    std::string base_dir;
    torch::Device device;
    int sample_rate;
    int sample_size;
    int overlap_size;
    int consumption_batch_size;
    long batch_size;
    long consumption_batches_multiplier;

    struct SampleInfo {
        std::string path;
    };

    std::vector<SampleInfo> sample_infos;
};


#endif //CLARIFICATIONDATASET_H
