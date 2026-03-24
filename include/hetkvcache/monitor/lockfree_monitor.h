/**
 * @file lockfree_monitor.h
 * @brief 高性能无锁访问监控器
 * 
 * 使用无锁数据结构和原子操作实现高性能访问监控，
 * 目标性能：500K+ ops/sec
 */

#ifndef HETKVCACHE_LOCKFREE_MONITOR_H
#define HETKVCACHE_LOCKFREE_MONITOR_H

#include "hetkvcache/types.h"
#include <atomic>
#include <array>
#include <vector>
#include <thread>
#include <memory>

namespace hetkvcache {

/**
 * @brief 无锁环形缓冲区
 * 
 * 单生产者多消费者（SPMC）无锁队列
 */
template<typename T, size_t Capacity>
class LockFreeRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    
public:
    LockFreeRingBuffer() : head_(0), tail_(0) {
        for (size_t i = 0; i < Capacity; i++) {
            buffer_[i].sequence.store(i);
        }
    }
    
    bool push(const T& item) {
        size_t pos = tail_.load(std::memory_order_relaxed);
        Cell* cell = &buffer_[pos & (Capacity - 1)];
        size_t seq = cell->sequence.load(std::memory_order_acquire);
        
        intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
        if (diff == 0) {
            if (tail_.compare_exchange_weak(pos, pos + 1, 
                std::memory_order_relaxed, std::memory_order_relaxed)) {
                cell->data = item;
                cell->sequence.store(pos + 1, std::memory_order_release);
                return true;
            }
        }
        return false;
    }
    
    bool pop(T& item) {
        size_t pos = head_.load(std::memory_order_relaxed);
        Cell* cell = &buffer_[pos & (Capacity - 1)];
        size_t seq = cell->sequence.load(std::memory_order_acquire);
        
        intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
        if (diff == 0) {
            if (head_.compare_exchange_weak(pos, pos + 1,
                std::memory_order_relaxed, std::memory_order_relaxed)) {
                item = cell->data;
                cell->sequence.store(pos + Capacity, std::memory_order_release);
                return true;
            }
        }
        return false;
    }
    
    size_t size() const {
        return tail_.load(std::memory_order_relaxed) - 
               head_.load(std::memory_order_relaxed);
    }
    
    bool empty() const {
        return size() == 0;
    }
    
    static constexpr size_t capacity() { return Capacity; }
    
private:
    struct Cell {
        std::atomic<size_t> sequence;
        T data;
    };
    
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
    alignas(64) std::array<Cell, Capacity> buffer_;
};

/**
 * @brief 访问记录条目（紧凑版，16字节）
 */
struct alignas(16) CompactAccessRecord {
    uint64_t block_id : 32;      // 块ID
    uint64_t timestamp : 32;     // 时间戳
    uint32_t layer_id : 8;       // 层ID
    uint32_t head_id : 8;        // 头ID
    uint32_t sequence_pos : 16;  // 序列位置
    uint8_t access_type;         // 访问类型
    
    CompactAccessRecord() : block_id(0), timestamp(0), layer_id(0), 
                            head_id(0), sequence_pos(0), access_type(0) {}
    
    CompactAccessRecord(BlockId bid, Timestamp ts, AccessType at,
                        LayerId lid, HeadId hid, SequencePos sp)
        : block_id(bid), timestamp(ts), layer_id(lid),
          head_id(hid), sequence_pos(sp), access_type(static_cast<uint8_t>(at)) {}
};

/**
 * @brief 高性能无锁访问监控器
 * 
 * 特性：
 * - 无锁环形缓冲区，支持高并发写入
 * - 原子计数器数组，O(1)访问计数更新
 * - 批量处理，减少线程唤醒开销
 */
class LockFreeAccessMonitor {
public:
    // 配置常量
    static constexpr size_t RING_BUFFER_SIZE = 65536;  // 2^16
    static constexpr size_t DEFAULT_MAX_BLOCKS = 100000;  // 默认最大块数（可配置）
    static constexpr size_t MAX_LAYERS = 128;          // 最大层数
    static constexpr size_t MAX_HEADS = 128;           // 最大头数
    static constexpr size_t BATCH_SIZE = 64;           // 批处理大小
    
    /**
     * @brief 监控模式
     */
    enum class MonitorMode {
        FULL,       ///< 全量记录
        SAMPLED,    ///< 采样记录
        CALLBACK    ///< 回调模式
    };
    
    /**
     * @brief 统计信息
     */
    struct Stats {
        uint64_t total_accesses;
        uint64_t read_accesses;
        uint64_t write_accesses;
        uint64_t queue_size;
        uint64_t unique_blocks;
        double sampling_rate;
    };
    
    LockFreeAccessMonitor();
    ~LockFreeAccessMonitor();
    
    // 禁止拷贝
    LockFreeAccessMonitor(const LockFreeAccessMonitor&) = delete;
    LockFreeAccessMonitor& operator=(const LockFreeAccessMonitor&) = delete;
    
    /**
     * @brief 初始化监控器
     */
    bool initialize(MonitorMode mode = MonitorMode::FULL,
                   float sample_rate = 1.0f,
                   size_t history_size = 1000000,
                   size_t max_blocks = DEFAULT_MAX_BLOCKS);
    
    /**
     * @brief 关闭监控器
     */
    void shutdown();
    
    /**
     * @brief 开始监控
     */
    void startMonitoring();
    
    /**
     * @brief 停止监控
     */
    void stopMonitoring();
    
    /**
     * @brief 是否正在监控
     */
    bool isMonitoring() const { return monitoring_active_.load(std::memory_order_relaxed); }
    
    /**
     * @brief 记录访问（高性能版本）
     * 
     * 无锁实现，适合高频调用
     */
    inline void recordAccess(BlockId block_id, AccessType access_type,
                            LayerId layer_id, HeadId head_id,
                            SequencePos sequence_pos = 0) {
        if (!monitoring_active_.load(std::memory_order_relaxed)) [[likely]] {
            return;
        }
        
        // 采样检查
        if (mode_ == MonitorMode::SAMPLED && !shouldSample()) {
            return;
        }
        
        // 更新原子计数器（无锁）
        updateAtomicCounts(block_id, access_type, layer_id, head_id);
        
        // 如果需要历史记录，推送到环形缓冲区
        if (record_history_) {
            Timestamp ts = getCurrentTimestamp();
            CompactAccessRecord record(block_id, ts, access_type, 
                                       layer_id, head_id, sequence_pos);
            ring_buffer_.push(record);
        }
        
        // 批量通知
        if (pending_count_.fetch_add(1, std::memory_order_relaxed) + 1 >= BATCH_SIZE) {
            notifyProcessor();
        }
    }
    
    /**
     * @brief 批量记录访问
     */
    void recordAccessBatch(const std::vector<KVBlockAccess>& accesses);
    
    /**
     * @brief 获取块的访问次数
     */
    uint64_t getAccessCount(BlockId block_id) const {
        if (block_id >= max_blocks_) return 0;
        return block_counts_[block_id].load(std::memory_order_relaxed);
    }
    
    /**
     * @brief 获取块的最后访问时间
     */
    Timestamp getLastAccessTime(BlockId block_id) const {
        if (block_id >= max_blocks_) return 0;
        return block_last_access_[block_id].load(std::memory_order_relaxed);
    }
    
    /**
     * @brief 获取层的访问次数
     */
    uint64_t getLayerAccessCount(LayerId layer_id) const {
        if (layer_id >= MAX_LAYERS) return 0;
        return layer_counts_[layer_id].load(std::memory_order_relaxed);
    }
    
    /**
     * @brief 获取头的访问次数
     */
    uint64_t getHeadAccessCount(HeadId head_id) const {
        if (head_id >= MAX_HEADS) return 0;
        return head_counts_[head_id].load(std::memory_order_relaxed);
    }
    
    /**
     * @brief 获取统计信息
     */
    Stats getStats() const;
    
    /**
     * @brief 重置统计信息
     */
    void resetStats();
    
    /**
     * @brief 清除历史记录
     */
    void clearHistory();
    
private:
    // 更新原子计数器
    inline void updateAtomicCounts(BlockId block_id, AccessType access_type,
                                   LayerId layer_id, HeadId head_id) {
        // 使用 relaxed 内存序以获得最佳性能
        if (block_id < max_blocks_) [[likely]] {
            block_counts_[block_id].fetch_add(1, std::memory_order_relaxed);
            block_last_access_[block_id].store(getCurrentTimestamp(), 
                                               std::memory_order_relaxed);
        }
        if (layer_id < MAX_LAYERS) [[likely]] {
            layer_counts_[layer_id].fetch_add(1, std::memory_order_relaxed);
        }
        if (head_id < MAX_HEADS) [[likely]] {
            head_counts_[head_id].fetch_add(1, std::memory_order_relaxed);
        }
        
        total_accesses_.fetch_add(1, std::memory_order_relaxed);
        if (access_type == AccessType::READ) {
            read_accesses_.fetch_add(1, std::memory_order_relaxed);
        } else {
            write_accesses_.fetch_add(1, std::memory_order_relaxed);
        }
    }
    
    // 判断是否应该采样
    inline bool shouldSample() {
        // 使用简单的线性同余生成器，避免锁
        uint64_t val = sample_state_.fetch_add(1, std::memory_order_relaxed);
        return (val & 0xFF) < sample_threshold_;
    }
    
    // 通知处理线程
    void notifyProcessor();
    
    // 处理线程函数
    void processLoop();
    
    // 获取当前时间戳
    static Timestamp getCurrentTimestamp() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();
    }
    
private:
    // 监控状态
    std::atomic<bool> monitoring_active_{false};
    std::atomic<bool> initialized_{false};
    std::atomic<bool> running_{false};
    MonitorMode mode_{MonitorMode::FULL};
    float sample_rate_{1.0f};
    bool record_history_{true};
    size_t max_blocks_{DEFAULT_MAX_BLOCKS};  // 可配置的最大块数
    
    // 无锁环形缓冲区
    using RingBuffer = LockFreeRingBuffer<CompactAccessRecord, RING_BUFFER_SIZE>;
    RingBuffer ring_buffer_;
    
    // 原子计数器数组（动态分配，替代哈希表+锁）
    std::unique_ptr<alignas(64) std::atomic<uint64_t>[]> block_counts_;
    std::unique_ptr<alignas(64) std::atomic<Timestamp>[]> block_last_access_;
    alignas(64) std::atomic<uint64_t> layer_counts_[MAX_LAYERS];
    alignas(64) std::atomic<uint64_t> head_counts_[MAX_HEADS];
    
    // 全局统计
    alignas(64) std::atomic<uint64_t> total_accesses_{0};
    alignas(64) std::atomic<uint64_t> read_accesses_{0};
    alignas(64) std::atomic<uint64_t> write_accesses_{0};
    
    // 批量处理相关
    std::atomic<uint64_t> pending_count_{0};
    
    // 采样相关
    std::atomic<uint64_t> sample_state_{0};
    uint8_t sample_threshold_{255};  // 255 = 100%采样
    
    // 处理线程
    std::thread processing_thread_;
    std::atomic<bool> notify_pending_{false};
};

}  // namespace hetkvcache

#endif  // HETKVCACHE_LOCKFREE_MONITOR_H