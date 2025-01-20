//
// Created by jacob on 1/12/25.
//

#ifndef CLARIFICATIONDATASETTEST_H
#define CLARIFICATIONDATASETTEST_H

#include <gtest/gtest.h>
#include "../src/ClarificationDataset.h"
#include <chrono>
#include <thread> // Required for std::this_thread::sleep_for

TEST(ClarificationDatasetTest, PerformanceBenchmark) {
    const torch::Device device(torch::kCPU);
    const std::string base_dir = "/workspace/noisy-commonvoice-24k-300ms-2ms-opus-4-en/test"; // Replace with your dataset path
    const std::string csv_filename = "test.csv";

    constexpr int batch_size = 16;
    constexpr int num_preload_batches = 16;
    ClarificationDataset dataset(
        device, base_dir, csv_filename, ClarificationDataset::Mode::Batch, num_preload_batches, batch_size, 16);

    // Wait for 4 seconds to allow preloading to saturate
    std::this_thread::sleep_for(std::chrono::seconds(4));

    // Get the first batch
    const auto first_batch = dataset.next();

    // 1. Validate dimensions of the first output
    std::cout << "First batch sizes: " << first_batch.sizes() << std::endl;
    ASSERT_EQ(first_batch.size(0), batch_size);       // Check batch size (number of frames)
    ASSERT_EQ(first_batch.size(1), 2);  // Check number of channels (should be 2 for noisy and clean)
    ASSERT_EQ(first_batch.size(2), dataset.sample_size);  // Check number of samples per frame (should be 7200 in your case)

    // 2. Performance benchmark (rest of the test)
    constexpr long num_batches_to_test = 10;
    const auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_batches_to_test; ++i) {
        const auto data = dataset.next();
    }
    const auto end = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double> duration = end - start;
    std::cout << "Frames per second: " << (num_batches_to_test * batch_size) / duration.count() << std::endl;

    ASSERT_GT(duration.count(), 0);
}

#endif //CLARIFICATIONDATASETTEST_H
