//
// Created by jacob on 2/1/25.
//

#ifndef CLARIFICATIONOPUSDATASET_H
#define CLARIFICATIONOPUSDATASET_H

#include "ClarificationDataset.h"

/**
 * Opus audio dataset loader.
 * Reads .opus files using FFmpeg/libopus decoder.
 */
class ClarificationOpusDataset final : public ClarificationDataset {
public:
    using ClarificationDataset::ClarificationDataset;
    
    ~ClarificationOpusDataset() override {
        // MUST stop producer thread before base class destructor runs
        // to avoid "pure virtual method called" errors
        stopProducerThread();
    }

    [[nodiscard]] torch::Tensor ProcessAudioFile(
        const std::string &absolute_path_string,
        std::vector<char> *decompressed_buffer,
        std::vector<char> *compressed_buffer) const override {
        
        // These buffers are not used for Opus (FFmpeg handles I/O)
        (void)decompressed_buffer;
        (void)compressed_buffer;

        // Silence FFmpeg timestamp warnings (harmless but very noisy)
        av_log_set_level(AV_LOG_ERROR);

        AVFormatContext *format_context = avformat_alloc_context();
        if (!format_context) {
            throw std::runtime_error("Failed to allocate format context");
        }

        if (avformat_open_input(&format_context, absolute_path_string.c_str(), nullptr, nullptr) != 0) {
            avformat_free_context(format_context);
            throw std::runtime_error("Failed to open audio file: " + absolute_path_string);
        }

        if (avformat_find_stream_info(format_context, nullptr) < 0) {
            avformat_close_input(&format_context);
            throw std::runtime_error("Failed to retrieve stream info: " + absolute_path_string);
        }

        // Find audio stream
        int audio_stream_index = -1;
        const AVCodecParameters *codec_params = nullptr;
        for (unsigned int i = 0; i < format_context->nb_streams; ++i) {
            if (format_context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
                audio_stream_index = static_cast<int>(i);
                codec_params = format_context->streams[i]->codecpar;
                break;
            }
        }
        
        if (audio_stream_index == -1) {
            avformat_close_input(&format_context);
            throw std::runtime_error("No audio stream found: " + absolute_path_string);
        }

        // Open decoder
        const AVCodec *codec = avcodec_find_decoder(codec_params->codec_id);
        if (!codec) {
            avformat_close_input(&format_context);
            throw std::runtime_error("Failed to find decoder: " + absolute_path_string);
        }

        AVCodecContext *codec_context = avcodec_alloc_context3(codec);
        if (!codec_context) {
            avformat_close_input(&format_context);
            throw std::runtime_error("Failed to allocate codec context");
        }

        if (avcodec_parameters_to_context(codec_context, codec_params) < 0) {
            avcodec_free_context(&codec_context);
            avformat_close_input(&format_context);
            throw std::runtime_error("Failed to copy codec parameters");
        }

        codec_context->request_sample_fmt = AV_SAMPLE_FMT_FLT;
        if (avcodec_open2(codec_context, codec, nullptr) < 0) {
            avcodec_free_context(&codec_context);
            avformat_close_input(&format_context);
            throw std::runtime_error("Failed to open codec: " + absolute_path_string);
        }

        // Get source sample rate from codec
        const int src_sample_rate = codec_context->sample_rate;
        const int dst_sample_rate = sample_rate;  // Target rate from info.csv (24000)
        const bool needs_resample = (src_sample_rate != dst_sample_rate);

        // Setup resampler if needed
        SwrContext *swr_ctx = nullptr;
        if (needs_resample) {
            swr_ctx = swr_alloc();
            if (!swr_ctx) {
                avcodec_free_context(&codec_context);
                avformat_close_input(&format_context);
                throw std::runtime_error("Failed to allocate resampler context");
            }

            AVChannelLayout src_ch_layout = codec_context->ch_layout;
            AVChannelLayout dst_ch_layout;
            av_channel_layout_default(&dst_ch_layout, 2);  // Stereo output

            av_opt_set_chlayout(swr_ctx, "in_chlayout", &src_ch_layout, 0);
            av_opt_set_chlayout(swr_ctx, "out_chlayout", &dst_ch_layout, 0);
            av_opt_set_int(swr_ctx, "in_sample_rate", src_sample_rate, 0);
            av_opt_set_int(swr_ctx, "out_sample_rate", dst_sample_rate, 0);
            av_opt_set_sample_fmt(swr_ctx, "in_sample_fmt", AV_SAMPLE_FMT_FLTP, 0);
            av_opt_set_sample_fmt(swr_ctx, "out_sample_fmt", AV_SAMPLE_FMT_FLT, 0);  // Interleaved output

            if (swr_init(swr_ctx) < 0) {
                swr_free(&swr_ctx);
                avcodec_free_context(&codec_context);
                avformat_close_input(&format_context);
                throw std::runtime_error("Failed to init resampler: " + absolute_path_string);
            }
        }

        // Decode audio
        AVPacket *packet = av_packet_alloc();
        AVFrame *frame = av_frame_alloc();

        std::vector<float> audio_channels[2];
        audio_channels[0].reserve(dst_sample_rate * 30);
        audio_channels[1].reserve(dst_sample_rate * 30);

        // Lambda to process decoded frame (with optional resampling)
        auto process_frame = [&]() {
            if (needs_resample && swr_ctx) {
                // Calculate output samples needed
                int64_t out_samples = av_rescale_rnd(
                    swr_get_delay(swr_ctx, src_sample_rate) + frame->nb_samples,
                    dst_sample_rate, src_sample_rate, AV_ROUND_UP);

                // Allocate output buffer (interleaved stereo float)
                std::vector<float> resampled(out_samples * 2);
                uint8_t *out_ptr = reinterpret_cast<uint8_t*>(resampled.data());

                int converted = swr_convert(swr_ctx, &out_ptr, out_samples,
                    const_cast<const uint8_t**>(frame->extended_data), frame->nb_samples);

                if (converted > 0) {
                    // Deinterleave to separate channels
                    for (int i = 0; i < converted; ++i) {
                        audio_channels[0].push_back(resampled[i * 2]);
                        audio_channels[1].push_back(resampled[i * 2 + 1]);
                    }
                }
            } else {
                // No resampling needed - direct copy
                const bool is_planar = av_sample_fmt_is_planar(static_cast<AVSampleFormat>(frame->format));
                const int num_channels = std::min(codec_context->ch_layout.nb_channels, 2);
                const int num_samples = frame->nb_samples;

                if (is_planar) {
                    for (int ch = 0; ch < num_channels; ++ch) {
                        const auto *data = reinterpret_cast<const float*>(frame->extended_data[ch]);
                        audio_channels[ch].insert(audio_channels[ch].end(), data, data + num_samples);
                    }
                } else {
                    const auto *interleaved = reinterpret_cast<const float*>(frame->extended_data[0]);
                    for (int i = 0; i < num_samples; ++i) {
                        for (int ch = 0; ch < num_channels; ++ch) {
                            audio_channels[ch].push_back(interleaved[i * num_channels + ch]);
                        }
                    }
                }
            }
        };

        while (av_read_frame(format_context, packet) >= 0) {
            if (packet->stream_index == audio_stream_index) {
                if (avcodec_send_packet(codec_context, packet) == 0) {
                    while (avcodec_receive_frame(codec_context, frame) == 0) {
                        process_frame();
                    }
                }
            }
            av_packet_unref(packet);
        }

        // Flush decoder
        avcodec_send_packet(codec_context, nullptr);
        while (avcodec_receive_frame(codec_context, frame) == 0) {
            process_frame();
        }

        // Flush resampler (get any remaining samples)
        if (needs_resample && swr_ctx) {
            int64_t out_samples = swr_get_delay(swr_ctx, dst_sample_rate);
            if (out_samples > 0) {
                std::vector<float> resampled(out_samples * 2);
                uint8_t *out_ptr = reinterpret_cast<uint8_t*>(resampled.data());
                int converted = swr_convert(swr_ctx, &out_ptr, out_samples, nullptr, 0);
                if (converted > 0) {
                    for (int i = 0; i < converted; ++i) {
                        audio_channels[0].push_back(resampled[i * 2]);
                        audio_channels[1].push_back(resampled[i * 2 + 1]);
                    }
                }
            }
        }

        // Cleanup FFmpeg
        if (swr_ctx) {
            swr_free(&swr_ctx);
        }
        av_frame_free(&frame);
        av_packet_free(&packet);
        avcodec_free_context(&codec_context);
        avformat_close_input(&format_context);

        // Check we have stereo data
        if (audio_channels[0].empty() || audio_channels[1].empty()) {
            return torch::empty({0, 2, sample_size}, torch::kFloat16);
        }

        // Create tensors for each channel (on CPU, fp32 from FFmpeg)
        std::vector<torch::Tensor> channel_tensors;
        for (int ch = 0; ch < 2; ++ch) {
            auto tensor = torch::from_blob(
                audio_channels[ch].data(),
                {static_cast<int64_t>(audio_channels[ch].size())},
                torch::kFloat32
            ).clone();
            channel_tensors.push_back(tensor);
        }

        // Stack channels: [2, total_samples]
        auto audio_combined = torch::stack(channel_tensors, 0);

        // Truncate to multiple of sample_size
        int64_t total_samples = audio_combined.size(1);
        int64_t num_chunks = total_samples / sample_size;
        
        if (num_chunks == 0) {
            return torch::empty({0, 2, sample_size}, torch::kFloat16);
        }

        int64_t truncated_size = num_chunks * sample_size;
        audio_combined = audio_combined.narrow(1, 0, truncated_size);

        // Reshape to [2, num_chunks, sample_size] then permute to [num_chunks, 2, sample_size]
        // Convert to fp16 for consistent training dtype (Opus decodes at fp32 internally)
        return audio_combined.view({2, num_chunks, sample_size}).permute({1, 0, 2}).contiguous().to(torch::kFloat16);
    }
};

#endif //CLARIFICATIONOPUSDATASET_H
