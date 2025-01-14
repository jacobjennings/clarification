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
    const std::string base_dir = "/workspace/mounted_image/";
    const std::string csv_filename = "samples.csv";

    ClarificationDataset dataset(batch_size, device, base_dir, csv_filename);

    auto start = std::chrono::high_resolution_clock::now();
    auto data = dataset.get(0);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken to load batch: " << duration.count() << " seconds" << std::endl;

    ASSERT_GT(duration.count(), 0);
}



#endif //CLARIFICATIONDATASETTEST_H
