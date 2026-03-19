/**
 * @file migration_engine.cpp
 * @brief 动态多级缓存迁移引擎实现
 */

#include "hetkvcache/migration/migration_engine.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <fstream>
#include <cstring>
#include <unistd.h>

namespace hetkvcache {

MigrationEngine::MigrationEngine()
    : initialized_(false)
    , vram_base_(nullptr)
    , vram_size_(0)
    , ram_base_(nullptr)
    , ram_size_(0)
    , ssd_fd_(-1)
    , running_(false)
    , next_task_id_(1)
    , vram_usage_(0)
    , ram_usage_(0)
    , ssd_usage_(0)
{
}

MigrationEngine::MigrationEngine(const MigrationEngineConfig& config)
    : config_(config)
    , initialized_(false)
    , vram_base_(nullptr)
    , vram_size_(0)
    , ram_base_(nullptr)
    , ram_size_(0)
    , ssd_fd_(-1)
    , running_(false)
    , next_task_id_(1)
    , vram_usage_(0)
    , ram_usage_(0)
    , ssd_usage_(0)
{
}

MigrationEngine::~MigrationEngine() {
    shutdown();
}

bool MigrationEngine::initialize(void* vram_base, size_t vram_size,
                                 void* ram_base, size_t ram_size,
                                 int ssd_fd) {
    if (initialized_.load()) {
        return true;
    }
    
    vram_base_ = vram_base;
    vram_size_ = vram_size;
    ram_base_ = ram_base;
    ram_size_ = ram_size;
    ssd_fd_ = ssd_fd;
    
    // 创建 CUDA Stream 池
    for (size_t i = 0; i < config_.max_concurrent_transfers; i++) {
        cudaStream_t stream;
        cudaError_t err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            // 清理已创建的 streams
            for (auto s : stream_pool_) {
                cudaStreamDestroy(s);
            }
            stream_pool_.clear();
            return false;
        }
        stream_pool_.push_back(stream);
        available_streams_.push(stream);
    }
    
    // 启动工作线程
    running_ = true;
    for (size_t i = 0; i < config_.max_concurrent_transfers; i++) {
        worker_threads_.emplace_back(&MigrationEngine::workerThread, this);
    }
    
    initialized_ = true;
    return true;
}

void MigrationEngine::shutdown() {
    if (!initialized_.load()) {
        return;
    }
    
    running_ = false;
    task_cv_.notify_all();
    
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
    
    // 销毁 CUDA Streams
    for (auto stream : stream_pool_) {
        cudaStreamDestroy(stream);
    }
    stream_pool_.clear();
    while (!available_streams_.empty()) {
        available_streams_.pop();
    }
    
    initialized_ = false;
}

uint64_t MigrationEngine::submitMigration(BlockId block_id,
                                          TierType source_tier,
                                          TierType target_tier,
                                          MigrationPriority priority) {
    std::lock_guard<std::mutex> lock(task_mutex_);
    
    MigrationTaskInfo task;
    task.task_id = getNextTaskId();
    task.block_id = block_id;
    task.source_tier = source_tier;
    task.target_tier = target_tier;
    task.priority = priority;
    task.status = MigrationStatus::PENDING;
    task.submit_time = getCurrentTimestamp();
    
    // 获取块地址
    auto addr_it = block_addresses_.find(block_id);
    if (addr_it != block_addresses_.end()) {
        task.source_ptr = addr_it->second[static_cast<int>(source_tier)];
        task.target_ptr = addr_it->second[static_cast<int>(target_tier)];
    }
    
    auto size_it = block_sizes_.find(block_id);
    if (size_it != block_sizes_.end()) {
        task.size = size_it->second;
    }
    
    uint64_t task_id = task.task_id;
    tasks_[task_id] = std::move(task);
    task_priority_queue_.push({static_cast<int>(priority) * 1000 +
                               static_cast<int>(task_id % 1000),
                               task_id});
    task_cv_.notify_one();
    
    return task_id;
}

uint64_t MigrationEngine::submitMigration(BlockId block_id,
                                          void* source_ptr,
                                          void* target_ptr,
                                          size_t size,
                                          TierType source_tier,
                                          TierType target_tier,
                                          MigrationPriority priority) {
    std::lock_guard<std::mutex> lock(task_mutex_);
    
    MigrationTaskInfo task;
    task.task_id = getNextTaskId();
    task.block_id = block_id;
    task.source_ptr = source_ptr;
    task.target_ptr = target_ptr;
    task.size = size;
    task.source_tier = source_tier;
    task.target_tier = target_tier;
    task.priority = priority;
    task.status = MigrationStatus::PENDING;
    task.submit_time = getCurrentTimestamp();
    
    uint64_t task_id = task.task_id;
    tasks_[task_id] = std::move(task);
    task_priority_queue_.push({static_cast<int>(priority) * 1000 + 
                               static_cast<int>(task.task_id % 1000), 
                               task.task_id});
    task_cv_.notify_one();
    
    return task.task_id;
}

std::vector<uint64_t> MigrationEngine::submitMigrationBatch(
    const std::vector<std::tuple<BlockId, TierType, TierType>>& tasks) {
    
    std::vector<uint64_t> task_ids;
    task_ids.reserve(tasks.size());
    
    for (const auto& [block_id, source, target] : tasks) {
        task_ids.push_back(submitMigration(block_id, source, target));
    }
    
    return task_ids;
}

bool MigrationEngine::awaitMigration(uint64_t task_id, uint64_t timeout_ms) {
    std::unique_lock<std::mutex> lock(task_mutex_);
    
    auto it = tasks_.find(task_id);
    if (it == tasks_.end()) {
        return false;
    }
    
    if (timeout_ms == 0) {
        // 无限等待
        task_cv_.wait(lock, [&]() {
            auto task_it = tasks_.find(task_id);
            return task_it == tasks_.end() || 
                   task_it->second.status == MigrationStatus::COMPLETED ||
                   task_it->second.status == MigrationStatus::FAILED;
        });
    } else {
        task_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&]() {
            auto task_it = tasks_.find(task_id);
            return task_it == tasks_.end() || 
                   task_it->second.status == MigrationStatus::COMPLETED ||
                   task_it->second.status == MigrationStatus::FAILED;
        });
    }
    
    auto result_it = tasks_.find(task_id);
    if (result_it != tasks_.end()) {
        return result_it->second.status == MigrationStatus::COMPLETED;
    }
    return false;
}

void MigrationEngine::awaitAllMigrations() {
    std::unique_lock<std::mutex> lock(task_mutex_);
    task_cv_.wait(lock, [&]() {
        return task_priority_queue_.empty() && 
               std::all_of(tasks_.begin(), tasks_.end(),
                          [](const auto& p) {
                              return p.second.status == MigrationStatus::COMPLETED ||
                                     p.second.status == MigrationStatus::FAILED;
                          });
    });
}

bool MigrationEngine::cancelMigration(uint64_t task_id) {
    std::lock_guard<std::mutex> lock(task_mutex_);
    
    auto it = tasks_.find(task_id);
    if (it != tasks_.end() && it->second.status == MigrationStatus::PENDING) {
        it->second.status = MigrationStatus::CANCELLED;
        return true;
    }
    return false;
}

void MigrationEngine::setMigrationCallback(MigrationCallback callback) {
    callback_ = std::move(callback);
}

MigrationStatus MigrationEngine::getTaskStatus(uint64_t task_id) const {
    std::lock_guard<std::mutex> lock(task_mutex_);
    auto it = tasks_.find(task_id);
    return (it != tasks_.end()) ? it->second.status : MigrationStatus::PENDING;
}

MigrationStats MigrationEngine::getStats() const {
    return stats_;
}

void MigrationEngine::resetStats() {
    stats_.reset();
}

size_t MigrationEngine::getPendingCount() const {
    std::lock_guard<std::mutex> lock(task_mutex_);
    size_t count = 0;
    for (const auto& [_, task] : tasks_) {
        if (task.status == MigrationStatus::PENDING || 
            task.status == MigrationStatus::IN_PROGRESS) {
            count++;
        }
    }
    return count;
}

void MigrationEngine::processHeatRecommendations(
    const std::vector<MigrationRecommendation>& recommendations) {
    
    for (const auto& rec : recommendations) {
        MigrationPriority priority = static_cast<MigrationPriority>(
            std::min(rec.priority / 25, 3));
        submitMigration(rec.block_id, rec.from_tier, rec.to_tier, priority);
    }
}

void MigrationEngine::setBlockAddress(BlockId block_id, TierType tier,
                                      void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(address_mutex_);
    block_addresses_[block_id][static_cast<int>(tier)] = ptr;
    block_sizes_[block_id] = size;
}

void* MigrationEngine::getBlockAddress(BlockId block_id, TierType tier) const {
    std::lock_guard<std::mutex> lock(address_mutex_);
    auto it = block_addresses_.find(block_id);
    if (it != block_addresses_.end()) {
        return it->second[static_cast<int>(tier)];
    }
    return nullptr;
}

void MigrationEngine::triggerPrefetch(const std::vector<BlockId>& blocks) {
    // 将块从冷层预取到热层
    for (BlockId block_id : blocks) {
        auto addr_it = block_addresses_.find(block_id);
        if (addr_it != block_addresses_.end()) {
            // 检查块当前所在层级
            if (addr_it->second[static_cast<int>(TierType::COLD)] != nullptr) {
                submitMigration(block_id, TierType::COLD, TierType::WARM,
                              MigrationPriority::LOW);
            } else if (addr_it->second[static_cast<int>(TierType::WARM)] != nullptr) {
                submitMigration(block_id, TierType::WARM, TierType::HOT,
                              MigrationPriority::LOW);
            }
        }
    }
}

size_t MigrationEngine::getTierUsage(TierType tier) const {
    switch (tier) {
        case TierType::HOT: return vram_usage_.load();
        case TierType::WARM: return ram_usage_.load();
        case TierType::COLD: return ssd_usage_.load();
        default: return 0;
    }
}

size_t MigrationEngine::getTierCapacity(TierType tier) const {
    switch (tier) {
        case TierType::HOT: return vram_size_;
        case TierType::WARM: return ram_size_;
        case TierType::COLD: return 0;  // SSD size not limited
        default: return 0;
    }
}

void MigrationEngine::workerThread() {
    while (running_.load()) {
        uint64_t task_id = 0;
        {
            std::unique_lock<std::mutex> lock(task_mutex_);
            task_cv_.wait(lock, [&]() {
                return !task_priority_queue_.empty() || !running_.load();
            });
            
            if (!running_.load()) {
                break;
            }
            
            if (task_priority_queue_.empty()) {
                continue;
            }
            
            task_id = task_priority_queue_.top().second;
            task_priority_queue_.pop();
        }
        
        // 获取任务
        MigrationTaskInfo* task = nullptr;
        {
            std::lock_guard<std::mutex> lock(task_mutex_);
            auto it = tasks_.find(task_id);
            if (it == tasks_.end() || it->second.status != MigrationStatus::PENDING) {
                continue;
            }
            task = &it->second;
            task->status = MigrationStatus::IN_PROGRESS;
            task->start_time = getCurrentTimestamp();
        }
        
        // 分配 stream
        task->stream = allocateStream();
        
        // 执行迁移
        bool success = executeMigration(*task);
        
        // 释放 stream
        releaseStream(task->stream);
        
        // 更新状态
        {
            std::lock_guard<std::mutex> lock(task_mutex_);
            task->status = success ? MigrationStatus::COMPLETED : MigrationStatus::FAILED;
            if (task->completion_promise) {
                task->completion_promise->set_value(success);
            }
            task->end_time = getCurrentTimestamp();
        }
        
        // 更新统计
        updateStats(*task, success);
        
        // 回调
        if (callback_) {
            callback_(*task);
        }
        
        task_cv_.notify_all();
    }
}

bool MigrationEngine::executeMigration(MigrationTaskInfo& task) {
    if (task.source_ptr == nullptr || task.target_ptr == nullptr) {
        return false;
    }
    
    bool success = false;
    
    switch (task.source_tier) {
        case TierType::HOT:
            switch (task.target_tier) {
                case TierType::WARM:
                    success = transferVRAMToRAM(task.source_ptr, task.target_ptr,
                                               task.size, task.stream);
                    break;
                case TierType::COLD:
                    success = transferVRAMToSSD(task.source_ptr, 0, task.size, task.stream);
                    break;
                default:
                    break;
            }
            break;
            
        case TierType::WARM:
            switch (task.target_tier) {
                case TierType::HOT:
                    success = transferRAMToVRAM(task.source_ptr, task.target_ptr,
                                               task.size, task.stream);
                    break;
                case TierType::COLD:
                    success = transferRAMToSSD(task.source_ptr, 0, task.size);
                    break;
                default:
                    break;
            }
            break;
            
        case TierType::COLD:
            switch (task.target_tier) {
                case TierType::HOT:
                    success = transferSSDToVRAM(0, task.target_ptr, task.size, task.stream);
                    break;
                case TierType::WARM:
                    success = transferSSDToRAM(0, task.target_ptr, task.size);
                    break;
                default:
                    break;
            }
            break;
    }
    
    if (success) {
        // 更新使用量
        size_t size = task.size;
        switch (task.source_tier) {
            case TierType::HOT: vram_usage_ -= size; break;
            case TierType::WARM: ram_usage_ -= size; break;
            case TierType::COLD: ssd_usage_ -= size; break;
        }
        switch (task.target_tier) {
            case TierType::HOT: vram_usage_ += size; break;
            case TierType::WARM: ram_usage_ += size; break;
            case TierType::COLD: ssd_usage_ += size; break;
        }
    }
    
    return success;
}

bool MigrationEngine::transferVRAMToRAM(void* src, void* dst, 
                                        size_t size, cudaStream_t stream) {
    cudaError_t err = cudaMemcpyAsync(dst, src, size, 
                                      cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        return false;
    }
    err = cudaStreamSynchronize(stream);
    return err == cudaSuccess;
}

bool MigrationEngine::transferRAMToVRAM(void* src, void* dst,
                                        size_t size, cudaStream_t stream) {
    cudaError_t err = cudaMemcpyAsync(dst, src, size,
                                      cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        return false;
    }
    err = cudaStreamSynchronize(stream);
    return err == cudaSuccess;
}

bool MigrationEngine::transferRAMToSSD(void* src, uint64_t offset, size_t size) {
    if (ssd_fd_ < 0) {
        return false;
    }
    
    ssize_t written = pwrite(ssd_fd_, src, size, offset);
    return static_cast<size_t>(written) == size;
}

bool MigrationEngine::transferSSDToRAM(uint64_t offset, void* dst, size_t size) {
    if (ssd_fd_ < 0) {
        return false;
    }
    
    ssize_t read_bytes = pread(ssd_fd_, dst, size, offset);
    return static_cast<size_t>(read_bytes) == size;
}

bool MigrationEngine::transferVRAMToSSD(void* src, uint64_t offset,
                                        size_t size, cudaStream_t stream) {
    // 需要经过 RAM 中转
    void* temp_buffer = malloc(size);
    if (!temp_buffer) {
        return false;
    }
    
    bool success = transferVRAMToRAM(src, temp_buffer, size, stream);
    if (success) {
        success = transferRAMToSSD(temp_buffer, offset, size);
    }
    
    free(temp_buffer);
    return success;
}

bool MigrationEngine::transferSSDToVRAM(uint64_t offset, void* dst,
                                        size_t size, cudaStream_t stream) {
    // 需要经过 RAM 中转
    void* temp_buffer = malloc(size);
    if (!temp_buffer) {
        return false;
    }
    
    bool success = transferSSDToRAM(offset, temp_buffer, size);
    if (success) {
        success = transferRAMToVRAM(temp_buffer, dst, size, stream);
    }
    
    free(temp_buffer);
    return success;
}

cudaStream_t MigrationEngine::allocateStream() {
    std::lock_guard<std::mutex> lock(stream_mutex_);
    if (available_streams_.empty()) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        return stream;
    }
    cudaStream_t stream = available_streams_.front();
    available_streams_.pop();
    return stream;
}

void MigrationEngine::releaseStream(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(stream_mutex_);
    available_streams_.push(stream);
}

void MigrationEngine::updateStats(const MigrationTaskInfo& task, bool success) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (success) {
        stats_.total_migrations++;
        stats_.total_bytes_transferred += task.size;
        
        double duration_ms = static_cast<double>(task.end_time - task.start_time) / 1e6;
        stats_.total_migration_time_ms += duration_ms;
        
        if (task.source_tier == TierType::HOT && task.target_tier == TierType::WARM) {
            stats_.vram_to_ram++;
        } else if (task.source_tier == TierType::WARM && task.target_tier == TierType::HOT) {
            stats_.ram_to_vram++;
        } else if (task.source_tier == TierType::WARM && task.target_tier == TierType::COLD) {
            stats_.ram_to_ssd++;
        } else if (task.source_tier == TierType::COLD && task.target_tier == TierType::WARM) {
            stats_.ssd_to_ram++;
        } else if (task.source_tier == TierType::HOT && task.target_tier == TierType::COLD) {
            stats_.vram_to_ssd++;
        } else if (task.source_tier == TierType::COLD && task.target_tier == TierType::HOT) {
            stats_.ssd_to_vram++;
        }
    } else {
        stats_.failed_migrations++;
    }
}

}  // namespace hetkvcache