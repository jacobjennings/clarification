//
// Created by jacob on 2/1/25.
//

#ifndef CLARIFICATIONLZ4DATASET_H
#define CLARIFICATIONLZ4DATASET_H

#include <torch/torch.h>
#include <string>
#include <lz4.h>
#include <fstream>
#include <vector>
#include <stdexcept>

#include "ClarificationDataset.h"

/**
 * LZ4-compressed raw audio dataset loader.
 * Reads .wav.lz4 files containing fp16 interleaved stereo audio.
 */
class ClarificationLz4Dataset final : public ClarificationDataset {
public:
    using ClarificationDataset::ClarificationDataset;
    
    ~ClarificationLz4Dataset() override {
        // MUST stop producer thread before base class destructor runs
        // to avoid "pure virtual method called" errors
        stopProducerThread();
    }

    [[nodiscard]] torch::Tensor ProcessAudioFile(
        const std::string &absolute_path_string,
        std::vector<char> *decompressed_buffer,
        std::vector<char> *compressed_buffer) const override {
        
        // Step 1: Read the LZ4-compressed file into memory
        std::ifstream file(absolute_path_string, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + absolute_path_string);
        }

        std::streamsize file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        // Validate file size fits in pre-allocated buffer
        if (static_cast<size_t>(file_size) > compressed_buffer->size()) {
            throw std::runtime_error("File too large for buffer (" + std::to_string(file_size) + 
                " bytes, buffer is " + std::to_string(compressed_buffer->size()) + " bytes): " + absolute_path_string);
        }

        if (!file.read(compressed_buffer->data(), file_size)) {
            throw std::runtime_error("Failed to read file: " + absolute_path_string);
        }

        // Step 2: Decompress LZ4 data
        int actual_decompressed_size = LZ4_decompress_safe(
            compressed_buffer->data(),
            decompressed_buffer->data(),
            static_cast<int>(file_size),
            static_cast<int>(decompressed_buffer->size())
        );
        
        if (actual_decompressed_size < 0) {
            throw std::runtime_error("LZ4 decompression failed (buffer=" + 
                std::to_string(decompressed_buffer->size()) + " bytes, compressed=" + 
                std::to_string(file_size) + " bytes): " + absolute_path_string);
        }

        // Step 3: Convert decompressed raw audio to torch::Tensor
        // Data format: interleaved fp16 stereo [sample0_L, sample0_R, sample1_L, sample1_R, ...]
        const auto* audio_data = reinterpret_cast<const void*>(decompressed_buffer->data());
        int num_samples = actual_decompressed_size / 2;  // 2 bytes per fp16 sample

        // Create tensor from raw fp16 data (stays on CPU)
        torch::Tensor audio_tensor = torch::from_blob(
            const_cast<void*>(audio_data),
            {num_samples},
            torch::kFloat16
        ).clone();  // Clone to ensure ownership

        // Reshape: [N] -> [N/2, 2] -> transpose -> [2, N/2] -> [2, chunks, sample_size] -> [chunks, 2, sample_size]
        // The data is interleaved: [L0, R0, L1, R1, ...] so we reshape to [samples, 2] then transpose
        int64_t num_stereo_samples = num_samples / 2;
        int64_t num_chunks = num_stereo_samples / sample_size;
        
        if (num_chunks == 0) {
            return torch::empty({0, 2, sample_size}, torch::kFloat16);
        }

        // Truncate to exact multiple of sample_size
        int64_t truncated_samples = num_chunks * sample_size * 2;  // *2 for stereo
        audio_tensor = audio_tensor.narrow(0, 0, truncated_samples);

        // Reshape to [chunks, sample_size, 2] then permute to [chunks, 2, sample_size]
        return audio_tensor.view({num_chunks, sample_size, 2}).permute({0, 2, 1}).contiguous();
    }
};

#endif //CLARIFICATIONLZ4DATASET_H
