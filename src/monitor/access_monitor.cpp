/**
 * @file access_monitor.cpp
 * @brief KV Cache 访问监控器实现
 */

#include "hetkvcache/monitor/access_monitor.h"
#include <random>
#include <algorithm>

namespace hetkvcache {

AccessMonitor::AccessMonitor()
    : monitoring_active_(false)
    , initialized_(false)
    , running_(false)
    , mode_(MonitorMode::FULL)
    , sample_rate_(1.0f)
    , history_buffer_(std::make_unique<AccessHistoryBuffer>())
    , total_accesses_(0)
    , read_accesses_(0)
    , write_accesses_(0)
{
}

AccessMonitor::~AccessMonitor() {
    shutdown();
}

bool AccessMonitor::initialize(MonitorMode mode, float sample_rate, size_t history_size) {
    if (initialized_.load()) {
        return true;
    }
    
    mode_ = mode;
    sample_rate_ = (mode == MonitorMode::SAMPLED) ? sample_rate : 1.0f;
    history_buffer_ = std::make_unique<AccessHistoryBuffer>(history_size);
    
    // 启动处理线程
    running_ = true;
    processing_thread_ = std::thread(&AccessMonitor::processQueue, this);
    
    initialized_ = true;
    return true;
}

void AccessMonitor::shutdown() {
    if (!initialized_.load()) {
        return;
    }
    
    stopMonitoring();
    running_ = false;
    queue_cv_.notify_all();
    
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
    
    initialized_ = false;
}

void AccessMonitor::startMonitoring() {
    monitoring_active_ = true;
}

void AccessMonitor::stopMonitoring() {
    monitoring_active_ = false;
}

void AccessMonitor::recordAccess(BlockId block_id, AccessType access_type,
                                 LayerId layer_id, HeadId head_id,
                                 SequencePos sequence_pos) {
    if (!monitoring_active_.load()) {
        return;
    }
    
    // 采样检查
    if (mode_ == MonitorMode::SAMPLED && !shouldSample()) {
        return;
    }
    
    Timestamp timestamp = getCurrentTimestamp();
    AccessRecord record(block_id, timestamp, access_type, layer_id, head_id, sequence_pos);
    
    if (mode_ == MonitorMode::CALLBACK && access_callback_) {
        access_callback_(record);
    } else {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        access_queue_.push(record);
        queue_cv_.notify_one();
    }
    
    total_accesses_++;
    if (access_type == AccessType::READ) {
        read_accesses_++;
    } else {
        write_accesses_++;
    }
}

void AccessMonitor::recordAccessBatch(const std::vector<KVBlockAccess>& accesses) {
    for (const auto& access : accesses) {
        recordAccess(access.block_id, access.access_type,
                    access.layer_id, access.head_id, access.sequence_pos);
    }
}

void AccessMonitor::setAccessCallback(AccessCallback callback) {
    access_callback_ = std::move(callback);
}

std::vector<AccessRecord> AccessMonitor::getBlockHistory(BlockId block_id) const {
    return history_buffer_->getByBlockId(block_id);
}

std::vector<AccessRecord> AccessMonitor::getRecentAccesses(size_t count) const {
    return history_buffer_->getRecent(count);
}

uint64_t AccessMonitor::getAccessCount(BlockId block_id) const {
    std::shared_lock<std::shared_mutex> lock(counts_mutex_);
    auto it = block_access_counts_.find(block_id);
    return (it != block_access_counts_.end()) ? it->second : 0;
}

Timestamp AccessMonitor::getLastAccessTime(BlockId block_id) const {
    std::shared_lock<std::shared_mutex> lock(counts_mutex_);
    auto it = block_last_access_.find(block_id);
    return (it != block_last_access_.end()) ? it->second : 0;
}

uint64_t AccessMonitor::getLayerAccessCount(LayerId layer_id) const {
    std::shared_lock<std::shared_mutex> lock(counts_mutex_);
    auto it = layer_access_counts_.find(layer_id);
    return (it != layer_access_counts_.end()) ? it->second : 0;
}

uint64_t AccessMonitor::getHeadAccessCount(HeadId head_id) const {
    std::shared_lock<std::shared_mutex> lock(counts_mutex_);
    auto it = head_access_counts_.find(head_id);
    return (it != head_access_counts_.end()) ? it->second : 0;
}

std::unordered_map<BlockId, uint64_t> AccessMonitor::getAllAccessCounts() const {
    std::shared_lock<std::shared_mutex> lock(counts_mutex_);
    return block_access_counts_;
}

void AccessMonitor::clearHistory() {
    history_buffer_->clear();
    
    std::unique_lock<std::shared_mutex> lock(counts_mutex_);
    block_access_counts_.clear();
    block_last_access_.clear();
    layer_access_counts_.clear();
    head_access_counts_.clear();
}

AccessMonitor::MonitorStats AccessMonitor::getStats() const {
    MonitorStats stats;
    stats.total_accesses = total_accesses_.load();
    stats.read_accesses = read_accesses_.load();
    stats.write_accesses = write_accesses_.load();
    stats.history_size = history_buffer_->size();
    stats.sampling_rate = sample_rate_;
    
    std::shared_lock<std::shared_mutex> lock(counts_mutex_);
    stats.unique_blocks_accessed = block_access_counts_.size();
    
    return stats;
}

void AccessMonitor::resetStats() {
    total_accesses_ = 0;
    read_accesses_ = 0;
    write_accesses_ = 0;
    clearHistory();
}

void AccessMonitor::processQueue() {
    while (running_.load()) {
        AccessRecord record;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return !access_queue_.empty() || !running_.load();
            });
            
            if (!running_.load() && access_queue_.empty()) {
                break;
            }
            
            if (access_queue_.empty()) {
                continue;
            }
            
            record = access_queue_.front();
            access_queue_.pop();
        }
        
        // 添加到历史缓冲区
        history_buffer_->push(record);
        
        // 更新计数
        updateCounts(record);
    }
}

bool AccessMonitor::shouldSample() const {
    static thread_local std::mt19937 generator(std::random_device{}());
    static thread_local std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    return distribution(generator) < sample_rate_;
}

void AccessMonitor::updateCounts(const AccessRecord& record) {
    std::unique_lock<std::shared_mutex> lock(counts_mutex_);
    
    block_access_counts_[record.block_id]++;
    block_last_access_[record.block_id] = record.timestamp;
    layer_access_counts_[record.layer_id]++;
    head_access_counts_[record.head_id]++;
}

}  // namespace hetkvcache