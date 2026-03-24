/**
 * @file lockfree_monitor.cpp
 * @brief 高性能无锁访问监控器实现
 */

#include "hetkvcache/monitor/lockfree_monitor.h"
#include <chrono>

namespace hetkvcache {

LockFreeAccessMonitor::LockFreeAccessMonitor() {
    // 初始化层和头计数器数组（静态）
    for (size_t i = 0; i < MAX_LAYERS; i++) {
        layer_counts_[i].store(0, std::memory_order_relaxed);
    }
    for (size_t i = 0; i < MAX_HEADS; i++) {
        head_counts_[i].store(0, std::memory_order_relaxed);
    }
    // 块计数器在initialize时动态分配
}

LockFreeAccessMonitor::~LockFreeAccessMonitor() {
    shutdown();
}

bool LockFreeAccessMonitor::initialize(MonitorMode mode, float sample_rate,
                                        size_t history_size, size_t max_blocks) {
    if (initialized_.load(std::memory_order_relaxed)) {
        return true;
    }
    
    mode_ = mode;
    sample_rate_ = (mode == MonitorMode::SAMPLED) ? sample_rate : 1.0f;
    sample_threshold_ = static_cast<uint8_t>(sample_rate_ * 255);
    record_history_ = (history_size > 0);
    max_blocks_ = max_blocks;
    
    // 动态分配块计数器数组
    block_counts_.reset(new alignas(64) std::atomic<uint64_t>[max_blocks_]);
    block_last_access_.reset(new alignas(64) std::atomic<Timestamp>[max_blocks_]);
    
    // 初始化原子计数器
    for (size_t i = 0; i < max_blocks_; i++) {
        block_counts_[i].store(0, std::memory_order_relaxed);
        block_last_access_[i].store(0, std::memory_order_relaxed);
    }
    
    // 启动处理线程
    running_.store(true, std::memory_order_relaxed);
    processing_thread_ = std::thread(&LockFreeAccessMonitor::processLoop, this);
    
    initialized_.store(true, std::memory_order_release);
    return true;
}

void LockFreeAccessMonitor::shutdown() {
    if (!initialized_.load(std::memory_order_relaxed)) {
        return;
    }
    
    stopMonitoring();
    running_.store(false, std::memory_order_relaxed);
    notifyProcessor();
    
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
    
    initialized_.store(false, std::memory_order_relaxed);
}

void LockFreeAccessMonitor::startMonitoring() {
    monitoring_active_.store(true, std::memory_order_relaxed);
}

void LockFreeAccessMonitor::stopMonitoring() {
    monitoring_active_.store(false, std::memory_order_relaxed);
}

void LockFreeAccessMonitor::recordAccessBatch(const std::vector<KVBlockAccess>& accesses) {
    for (const auto& access : accesses) {
        recordAccess(access.block_id, access.access_type,
                    access.layer_id, access.head_id, access.sequence_pos);
    }
}

void LockFreeAccessMonitor::notifyProcessor() {
    notify_pending_.store(true, std::memory_order_release);
}

void LockFreeAccessMonitor::processLoop() {
    while (running_.load(std::memory_order_relaxed)) {
        // 批量处理环形缓冲区中的记录
        CompactAccessRecord record;
        size_t processed = 0;
        
        while (processed < BATCH_SIZE && ring_buffer_.pop(record)) {
            // 可以在这里添加额外的处理逻辑
            // 例如：写入持久化存储、触发回调等
            processed++;
        }
        
        // 等待通知或超时
        if (processed == 0) {
            // 短暂休眠以减少CPU占用
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        
        // 清除通知标志
        notify_pending_.store(false, std::memory_order_relaxed);
    }
}

LockFreeAccessMonitor::Stats LockFreeAccessMonitor::getStats() const {
    Stats stats;
    stats.total_accesses = total_accesses_.load(std::memory_order_relaxed);
    stats.read_accesses = read_accesses_.load(std::memory_order_relaxed);
    stats.write_accesses = write_accesses_.load(std::memory_order_relaxed);
    stats.queue_size = ring_buffer_.size();
    stats.sampling_rate = sample_rate_;
    
    // 统计唯一块数
    uint64_t unique = 0;
    for (size_t i = 0; i < max_blocks_; i++) {
        if (block_counts_[i].load(std::memory_order_relaxed) > 0) {
            unique++;
        }
    }
    stats.unique_blocks = unique;
    
    return stats;
}

void LockFreeAccessMonitor::resetStats() {
    total_accesses_.store(0, std::memory_order_relaxed);
    read_accesses_.store(0, std::memory_order_relaxed);
    write_accesses_.store(0, std::memory_order_relaxed);
    
    for (size_t i = 0; i < max_blocks_; i++) {
        block_counts_[i].store(0, std::memory_order_relaxed);
        block_last_access_[i].store(0, std::memory_order_relaxed);
    }
    for (size_t i = 0; i < MAX_LAYERS; i++) {
        layer_counts_[i].store(0, std::memory_order_relaxed);
    }
    for (size_t i = 0; i < MAX_HEADS; i++) {
        head_counts_[i].store(0, std::memory_order_relaxed);
    }
    
    sample_state_.store(0, std::memory_order_relaxed);
}

void LockFreeAccessMonitor::clearHistory() {
    // 清空环形缓冲区
    CompactAccessRecord record;
    while (ring_buffer_.pop(record)) {
        // 只是消费掉所有记录
    }
    resetStats();
}

}  // namespace hetkvcache