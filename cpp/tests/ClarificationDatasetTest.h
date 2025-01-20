//
// Created by Jacob Jennings on 1/12/25.
//

#ifndef CLARIFICATIONDATASETTEST_H
#define CLARIFICATIONDATASETTEST_H

#include <gtest/gtest.h>
#include "../src/ClarificationDataset.h"
#include <chrono>
#include <iomanip>  // Required for std::setprecision
#include <thread>   // Required for std::this_thread::sleep_for

// Helper function to format large numbers with suffixes (k, M, G, etc.)
inline std::string FormatWithSuffix(const double value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);

    if (value >= 1e9) {
        oss << value / 1e9 << "G";
    } else if (value >= 1e6) {
        oss << value / 1e6 << "M";
    } else if (value >= 1e3) {
        oss << value / 1e3 << "k";
    } else {
        oss << std::fixed << std::setprecision(0) << value;
    }
    return oss.str();
}

TEST(ClarificationDatasetTest, PerformanceBenchmark) {
    const torch::Device device(torch::kCPU);
    // Replace with your dataset path
    const std::string base_dir = "/workspace/noisy-commonvoice-24k-300ms-2ms-opus-4-en/test";
    const std::string csv_filename = "test.csv";

    constexpr int kBatchSize = 16;
    constexpr int kNumPreloadBatches = 16;
    ClarificationDataset dataset(
        device, base_dir, csv_filename, kNumPreloadBatches, kBatchSize, 16);

    // Wait for 4 seconds to allow preloading to saturate.
    std::this_thread::sleep_for(std::chrono::seconds(4));

    // Get the first batch
    const auto first_batch = dataset.next();

    // 1. Validate dimensions of the first output
    std::cout << "First batch sizes: " << first_batch.sizes() << std::endl;
    ASSERT_EQ(first_batch.size(0), kBatchSize); // Check batch size (number of frames)
    ASSERT_EQ(first_batch.size(1), 2); // Check number of channels (2 for noisy and clean)
    ASSERT_EQ(first_batch.size(2), dataset.sample_size); // Check number of samples per frame

    // 2. Validate that the dataset is returning different data during each call to next()
    const auto second_batch = dataset.next();
    ASSERT_FALSE(torch::allclose(first_batch, second_batch)); // Should be different

    // 3. Performance benchmark
    constexpr long kNumBatchesToTest = 3000;
    const auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < kNumBatchesToTest; ++i) {
        const auto data = dataset.next();
    }
    const auto end = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double> duration = end - start;
    const double frames_per_second = (kNumBatchesToTest * kBatchSize) / duration.count();
    std::cout << "Frames per second: " << FormatWithSuffix(frames_per_second) << std::endl;
    const double samples_per_second = frames_per_second * dataset.sample_size;
    // print samples per microsecond
    std::cout << "Samples per microsecond: " << FormatWithSuffix(samples_per_second / 1e6) << std::endl;
    std::cout << "Files processed: " << dataset.file_idx << std::endl;

    ASSERT_GT(duration.count(), 0);
}

#endif  // CLARIFICATIONDATASETTEST_H
