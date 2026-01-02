//
// Created by Jacob Jennings on 1/12/25.
//

#ifndef CLARIFICATIONDATASET_H
#define CLARIFICATIONDATASET_H

#include <omp.h>

#include <algorithm>
#include <condition_variable>
#include <cstring>
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
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
}

/**
 * Abstract base class for audio dataset loaders.
 * Provides common preloading, batching, and queue management.
 * Subclasses implement ProcessAudioFile() for specific formats (Opus, LZ4, etc.)
 */
class ClarificationDataset {
public:
    ClarificationDataset(
        torch::Device device,
        const std::string &base_dir,
        const std::string &csv_filename,
        int num_preload_batches = 16,
        int batch_size = 16,
        int num_threads = 0
    ) : device(device),
        base_dir_(base_dir),
        num_preload_batches_(num_preload_batches),
        batch_size_(batch_size),
        num_threads_(num_threads) {
        
        if (batch_size <= 0) {
            throw std::invalid_argument("batch_size must be > 0");
        }

        // Default to (available CPUs - 2) to leave headroom for other processes
        // Use at least 1 thread, cap at 32
        const int available_cpus = omp_get_max_threads();
        const int default_threads = std::max(1, std::min(available_cpus - 2, 32));
        num_threads_ = num_threads > 0 ? num_threads : default_threads;

        auto info_csv_path = std::filesystem::path(base_dir_) / "info.csv";
        std::ifstream csvfile(info_csv_path);

        if (!csvfile.is_open()) {
            throw std::runtime_error("Failed to open info.csv at " + info_csv_path.string());
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

        std::cout << "ClarificationDataset initialized with sample_rate: " << sample_rate 
                  << ", sample_size: " << sample_size 
                  << ", overlap_size: " << overlap_size << std::endl;

        auto samples_csv_path = std::filesystem::path(base_dir_) / csv_filename;
        std::cout << "Path to samples CSV: " << samples_csv_path << std::endl;
        std::ifstream samples_csv(samples_csv_path);

        if (!samples_csv.is_open()) {
            throw std::runtime_error("Failed to open samples CSV: " + samples_csv_path.string());
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

        // Allocate per-thread buffers for file I/O (used by LZ4 loader)
        thread_buffers_.resize(num_threads_);
        thread_compressed_buffers_.resize(num_threads_);
        
        // Buffer sizes: LZ4 files can be 30-40 MB decompressed
        const size_t max_buffer_size = 160 * 1024 * 1024;      // 160 MiB decompressed
        const size_t max_compressed_size = 80 * 1024 * 1024;  // 80 MiB compressed
        
        for (int idx = 0; idx < num_threads_; ++idx) {
            thread_buffers_[idx] = std::vector<char>(max_buffer_size);
            thread_compressed_buffers_[idx] = std::vector<char>(max_compressed_size);
            // Pre-touch pages to avoid page fault storms
            std::memset(thread_buffers_[idx].data(), 0, max_buffer_size);
            std::memset(thread_compressed_buffers_[idx].data(), 0, max_compressed_size);
        }
        
        std::cout << "Using " << num_threads_ << " threads, "
                  << (num_threads_ * (max_buffer_size + max_compressed_size)) / (1024 * 1024) 
                  << " MiB total buffer memory" << std::endl;

        leftover_frames_ = torch::Tensor();
        
        // Start producer thread AFTER all initialization is complete
        producer_thread_ = std::thread([this] { PreloadBatchesLoop(); });
    }

    virtual ~ClarificationDataset() {
        // Note: Subclasses MUST call stopProducerThread() in their destructors
        // before this base destructor runs, to avoid "pure virtual method called" errors.
        // This is a safety net in case they forget.
        stopProducerThread();
    }

    // Non-copyable
    ClarificationDataset(const ClarificationDataset&) = delete;
    ClarificationDataset& operator=(const ClarificationDataset&) = delete;

protected:
    /**
     * Stop the producer thread. MUST be called by subclass destructors
     * before the base class destructor runs.
     */
    void stopProducerThread() {
        // Set quit flag
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            quit_ = true;
        }
        preload_cv_.notify_all();
        
        // Wait for producer thread to finish
        if (producer_thread_.joinable()) {
            producer_thread_.join();
        }
        
        // Wait for any in-flight ProcessAudioFile calls to complete
        while (in_flight_count_.load() > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

public:

    torch::Tensor next() {
        torch::NoGradGuard no_grad;

        std::unique_lock<std::mutex> lock(queue_mutex_);
        preload_cv_.wait(lock, [this] { 
            return !preloaded_batches_.empty() || quit_ || all_files_processed_; 
        });

        if (preloaded_batches_.empty()) {
            throw std::out_of_range("End of dataset reached");
        }

        torch::Tensor batch_tensor = preloaded_batches_.front();
        preloaded_batches_.pop();
        lock.unlock();
        preload_cv_.notify_one();

        return batch_tensor.to(device, /*non_blocking=*/true);
    }

    void reset() {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        file_idx = 0;
        all_files_processed_ = false;
        while (!preloaded_batches_.empty()) {
            preloaded_batches_.pop();
        }
        leftover_frames_ = torch::Tensor();
        preload_cv_.notify_all();
    }

    /**
     * Set file_idx directly without consuming batches (O(1) operation).
     * Used for fair comparison mode to restore loader position efficiently.
     * Clears preload queue and leftover frames, then starts preloading from new position.
     */
    void set_file_idx(int target_idx) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        // Clamp to valid range
        target_idx = std::max(0, std::min(target_idx, static_cast<int>(sample_infos_.size())));
        file_idx = target_idx;
        all_files_processed_ = (target_idx >= static_cast<int>(sample_infos_.size()));
        // Clear preload queue - will be refilled from new position
        while (!preloaded_batches_.empty()) {
            preloaded_batches_.pop();
        }
        leftover_frames_ = torch::Tensor();
        preload_cv_.notify_all();
    }

    int total_files() const {
        return static_cast<int>(sample_infos_.size());
    }

    // Public members for Python bindings
    int sample_size = 0;
    int sample_rate = 0;
    int overlap_size = 0;
    int file_idx = 0;
    torch::Device device;

protected:
    /**
     * Pure virtual method - subclasses implement format-specific audio loading.
     * Should return tensor of shape [num_chunks, 2, sample_size] on CPU.
     */
    [[nodiscard]] virtual torch::Tensor ProcessAudioFile(
        const std::string &absolute_path_string,
        std::vector<char> *decompressed_buffer,
        std::vector<char> *compressed_buffer) const = 0;

    std::string base_dir_;

private:
    int num_preload_batches_ = 0;
    int batch_size_ = 0;
    int num_threads_ = 0;

    struct SampleInfo {
        std::string path;
    };

    std::vector<SampleInfo> sample_infos_;
    std::queue<torch::Tensor> preloaded_batches_;
    mutable std::mutex queue_mutex_;
    std::condition_variable preload_cv_;
    std::atomic<bool> quit_{false};
    std::atomic<bool> all_files_processed_{false};
    std::atomic<int> in_flight_count_{0};  // Count of in-flight ProcessAudioFile calls
    std::thread producer_thread_;
    std::vector<std::vector<char>> thread_buffers_;
    std::vector<std::vector<char>> thread_compressed_buffers_;
    torch::Tensor leftover_frames_;

    void PreloadBatchesLoop() {
        while (!quit_) {
            int queue_space;
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                queue_space = num_preload_batches_ - static_cast<int>(preloaded_batches_.size());
            }

            // Only load if queue has space
            if (queue_space > 0) {
                bool has_more_files;
                {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    has_more_files = file_idx < static_cast<int>(sample_infos_.size());
                }

                if (has_more_files) {
                    // Load just 1-2 batches worth at a time for smoother CPU usage
                    // instead of filling the entire queue at once
                    const int max_batches_per_iteration = 2;
                    int batches_to_load = std::min(queue_space, max_batches_per_iteration);
                    PreloadBatches(batches_to_load);
                    
                    // Small yield to avoid tight spinning
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                } else {
                    all_files_processed_ = true;
                    preload_cv_.notify_all();
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
            } else {
                // Queue is full, wait a bit before checking again
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
    }

    void PreloadBatches(int num_batches_to_preload) {
        // Early exit if we're shutting down
        if (quit_.load()) {
            return;
        }

        int start_idx;
        int actual_to_load;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            start_idx = file_idx;
            actual_to_load = std::min(
                num_batches_to_preload * batch_size_,
                static_cast<int>(sample_infos_.size()) - file_idx
            );
            if (actual_to_load <= 0) {
                return;
            }
            file_idx += actual_to_load;
        }

        std::vector<torch::Tensor> new_frames_vector(actual_to_load);

        #pragma omp parallel for num_threads(num_threads_)
        for (int i = 0; i < actual_to_load; ++i) {
            // Check if we're shutting down (avoid calling virtual methods during destruction)
            if (quit_.load()) {
                continue;
            }
            
            const int current_file_idx = start_idx + i;
            const auto &[path] = sample_infos_[current_file_idx];
            const std::filesystem::path absolute_path = std::filesystem::path(base_dir_) / path;
            const int thread_idx = omp_get_thread_num();
            
            // Track in-flight calls to avoid destruction during processing
            in_flight_count_.fetch_add(1);
            
            try {
                // Double-check quit flag before calling virtual method
                if (!quit_.load()) {
                    new_frames_vector[i] = ProcessAudioFile(
                        absolute_path.string(),
                        &thread_buffers_[thread_idx],
                        &thread_compressed_buffers_[thread_idx]
                    );
                }
            } catch (const std::exception& e) {
                std::cerr << "Error processing file " << absolute_path << ": " << e.what() << std::endl;
                new_frames_vector[i] = torch::Tensor();
            }
            
            in_flight_count_.fetch_sub(1);
        }
        
        // Exit early if shutting down
        if (quit_.load()) {
            return;
        }

        // Collect valid tensors
        std::vector<torch::Tensor> valid_tensors;
        for (auto& t : new_frames_vector) {
            if (t.numel() > 0) {
                valid_tensors.push_back(t);
            }
        }

        if (valid_tensors.empty()) {
            return;
        }

        // Concatenate all frames
        torch::Tensor new_frames_all = torch::cat(valid_tensors, 0);

        // Batch the frames
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            
            torch::Tensor frames_to_batch;
            if (leftover_frames_.numel() > 0) {
                frames_to_batch = torch::cat({leftover_frames_, new_frames_all}, 0);
                leftover_frames_ = torch::Tensor();
            } else {
                frames_to_batch = new_frames_all;
            }

            const int64_t num_frames = frames_to_batch.size(0);
            const int64_t num_full_batches = num_frames / batch_size_;
            const int64_t remaining_frames = num_frames % batch_size_;

            for (int64_t i = 0; i < num_full_batches; ++i) {
                // CRITICAL: Use .clone() to create independent tensors!
                // Without clone(), narrow() creates a VIEW that keeps the entire
                // frames_to_batch tensor alive until ALL batches are consumed.
                torch::Tensor batch_tensor = frames_to_batch.narrow(0, i * batch_size_, batch_size_).clone();
                preloaded_batches_.push(batch_tensor);
            }

            if (remaining_frames > 0) {
                leftover_frames_ = frames_to_batch.narrow(0, num_full_batches * batch_size_, remaining_frames).clone();
            }
        }
        preload_cv_.notify_all();
    }
};

#endif // CLARIFICATIONDATASET_H
