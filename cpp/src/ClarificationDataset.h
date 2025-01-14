//
// Created by jacob on 1/12/25.
//

#ifndef CLARIFICATIONDATASET_H
#define CLARIFICATIONDATASET_H



#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <torch/torch.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
}

class ClarificationDataset final : public torch::data::Dataset<ClarificationDataset, torch::Tensor> {
public:
    ClarificationDataset(
        size_t batch_size,
        torch::Device device,
        const std::string& base_dir,
        const std::string& csv_filename
    ) : base_dir(base_dir), device(device) {
        std::ifstream csvfile(base_dir + "/info.csv");
        if (!csvfile.is_open()) {
            throw std::runtime_error("Failed to open info.csv");
        }

        std::string line;
        std::getline(csvfile, line); // Skip header

        std::getline(csvfile, line);
        std::stringstream ss(line);
        std::string value;
        std::getline(ss, value, ','); sample_rate = std::stoi(value);
        std::getline(ss, value, ','); sample_size = std::stoi(value);
        std::getline(ss, value, ','); overlap_size = std::stoi(value);
        std::getline(ss, value, ','); consumption_batch_size = std::stoi(value);

        this->batch_size = batch_size;
        if (this->batch_size % this->consumption_batch_size != 0) {
            throw std::runtime_error("batch_size must be a multiple of consumption_batch_size");
        }

        consumption_batches_multiplier = this->batch_size / this->consumption_batch_size;

        std::ifstream samples_csv(base_dir + "/" + csv_filename);
        if (!samples_csv.is_open()) {
            throw std::runtime_error("Failed to open samples CSV file");
        }

        std::getline(samples_csv, line); // Skip header
        while (std::getline(samples_csv, line)) {
            std::stringstream strs(line);
            std::string path;
            std::getline(strs, path, ',');
            sample_infos.push_back({path});
        }
    }

    torch::Tensor get(size_t batch_idx) override {
        torch::NoGradGuard no_grad;

        torch::Tensor audio_aggregated;
        for (long i = 0; i < consumption_batches_multiplier; ++i) {
            const long consumption_batch_idx = batch_idx * consumption_batches_multiplier + i;
            const auto& sample_info = sample_infos[consumption_batch_idx];
            std::string path = sample_info.path;
            std::string absolute_path = base_dir + "/" + path;

            // Use ffmpeg to load the opus file at absolute_path at sample_rate into a tensor named audio.
            AVFormatContext* format_context = avformat_alloc_context();
            if (avformat_open_input(&format_context, absolute_path.c_str(), nullptr, nullptr) != 0) {
                throw std::runtime_error("Failed to open audio file");
            }
            if (avformat_find_stream_info(format_context, nullptr) < 0) {
                throw std::runtime_error("Failed to retrieve stream info");
            }

            // Find the audio stream
            int audio_stream_index = -1;
            AVCodecParameters* codec_params = nullptr;
            for (unsigned int i = 0; i < format_context->nb_streams; ++i) {
                codec_params = format_context->streams[i]->codecpar;
                if (codec_params->codec_type == AVMEDIA_TYPE_AUDIO) {
                    audio_stream_index = i;
                    break;
                }
            }
            if (audio_stream_index == -1) {
                throw std::runtime_error("Failed to find audio stream");
            }

            // Find and open the decoder
            const AVCodec* codec = avcodec_find_decoder(codec_params->codec_id);
            if (!codec) {
                throw std::runtime_error("Failed to find codec");
            }
            AVCodecContext* codec_context = avcodec_alloc_context3(codec);
            if (!codec_context) {
                throw std::runtime_error("Failed to allocate codec context");
            }
            if (avcodec_parameters_to_context(codec_context, codec_params) < 0) {
                throw std::runtime_error("Failed to copy codec parameters to context");
            }

            // Set the desired output sample rate
            codec_context->sample_rate = sample_rate;

            if (avcodec_open2(codec_context, codec, nullptr) < 0) {
                throw std::runtime_error("Failed to open codec");
            }

            // Decode frames
            AVPacket* packet = av_packet_alloc();
            AVFrame* frame = av_frame_alloc();
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
            audio = torch::from_blob(audio_data.data(), {1, static_cast<long>(audio_data.size())}, torch::kFloat);
            audio = audio.to(device);
            if (i == 0) {
                audio_aggregated = torch::empty({consumption_batches_multiplier, static_cast<long>(audio_data.size())}, torch::kFloat).to(device);
            }
            audio_aggregated[i] = audio.first;
        }

        auto audio_samples = torch::stack(audio_aggregated.split(sample_size, /*dim=*/1));
        return audio_samples;
    }

    static void decode_audio_frames(AVCodecContext* codec_context, AVPacket* packet, AVFrame* frame, std::vector<float>& audio_data) {
        while (avcodec_receive_frame(codec_context, frame) == 0) {
            if (frame->format == AV_SAMPLE_FMT_FLTP) {
                for (int i = 0; i < frame->nb_samples; ++i) {
                    for (int ch = 0; ch < codec_context->ch_layout.nb_channels; ++ch) {
                        const auto* data = (float*)frame->data[ch];
                        audio_data.push_back(data[i]);
                    }
                }
            } else if (frame->format == AV_SAMPLE_FMT_FLT) {
                const auto* data = (float*)frame->data[0];
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
    size_t batch_size;
    long consumption_batches_multiplier;

    struct SampleInfo {
        std::string path;
    };
    std::vector<SampleInfo> sample_infos;
};



#endif //CLARIFICATIONDATASET_H
