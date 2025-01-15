//
// Created by jacob on 1/12/25.
//

#ifndef CLARIFICATIONDATASETTEST_H
#define CLARIFICATIONDATASETTEST_H

#include <gtest/gtest.h>
#include "../src/ClarificationDataset.h"
#include <chrono>

TEST(ClarificationDatasetTest, PerformanceBenchmark) {
    const size_t batch_size = 32;
    const torch::Device device(torch::kCPU);
    const std::string base_dir = "/workspace/mounted_image/noisy-commonvoice-24k-300ms-5ms-opus/train";
    const std::string csv_filename = "train.csv";

    ClarificationDataset dataset(batch_size, device, base_dir, csv_filename);

    // testing::internal::CaptureStdout();

    const long num_batches_to_test = 100;
    const auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_batches_to_test; ++i) {
        const auto data = dataset.get(0);
    }
    const auto end = std::chrono::high_resolution_clock::now();

    // std::string output = testing::internal::GetCapturedStdout();
    // std::cout << output << std::endl;

    const std::chrono::duration<double> duration = end - start;
    // Print batches per second
    std::cout << "Batches per second: " << num_batches_to_test / duration.count() << std::endl;

    ASSERT_GT(duration.count(), 0);
}



#endif //CLARIFICATIONDATASETTEST_H
