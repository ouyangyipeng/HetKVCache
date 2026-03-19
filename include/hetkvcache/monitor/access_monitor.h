/**
 * @file access_monitor.h
 * @brief KV Cache 访问监控器
 * 
 * 负责拦截和记录 LLM 推理进程对 KV Cache 的访问，
 * 为热度评估提供原始数据。
 */

#ifndef HETKVCACHE_ACCESS_MONITOR_H
#define HETKVCACHE_ACCESS_MONITOR_H

#include "hetkvcache/types.h"
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <queue>
#include <unordered_map>
#include <deque>
#include <functional>
#include <vector>

namespace hetkvcache {

/**
 * @brief 访问记录条目
 */
struct AccessRecord {
    BlockId block_id;
    Timestamp timestamp;
    AccessType access_type;
    LayerId layer_id;
    HeadId head_id;
    SequencePos sequence_pos;
    
    AccessRecord() : block_id(INVALID_BLOCK_ID), timestamp(0), 
                     access_type(AccessType::READ), layer_id(0), 
                     head_id(0), sequence_pos(0) {}
    
    AccessRecord(BlockId bid, Timestamp ts, AccessType at, 
                 LayerId lid, HeadId hid, SequencePos sp)
        : block_id(bid), timestamp(ts), access_type(at),
          layer_id(lid), head_id(hid), sequence_pos(sp) {}
};

/**
 * @brief 访问历史缓冲区
 */
class AccessHistoryBuffer {
public:
    explicit AccessHistoryBuffer(size_t max_size = 1000000)
        : max_size_(max_size) {}
    
    void push(const AccessRecord& record) {
        std::lock_guard<std::mutex> lock(mutex_);
        buffer_.push_back(record);
        if (buffer_.size() > max_size_) {
            buffer_.pop_front();
        }
    }
    
    std::vector<AccessRecord> getRecent(size_t count) const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<AccessRecord> result;
        size_t n = std::min(count, buffer_.size());
        result.reserve(n);
        auto it = buffer_.rbegin();
        for (size_t i = 0; i < n && it != buffer_.rend(); ++i, ++it) {
            result.push_back(*it);
        }
        return result;
    }
    
    std::vector<AccessRecord> getByBlockId(BlockId block_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<AccessRecord> result;
        for (const auto& record : buffer_) {
            if (record.block_id == block_id) {
                result.push_back(record);
            }
        }
        return result;
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.size();
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        buffer_.clear();
    }
    
private:
    mutable std::mutex mutex_;
    std::deque<AccessRecord> buffer_;
    size_t max_size_;
};

/**
 * @brief 访问监控回调类型
 */
using AccessCallback = std::function<void(const AccessRecord&)>;

/**
 * @brief KV Cache 访问监控器
 * 
 * 提供 KV Cache 访问的监控、记录和回调功能。
 * 支持多种监控方式：Hook、采样、全量记录。
 */
class AccessMonitor {
public:
    /**
     * @brief 监控模式
     */
    enum class MonitorMode {
        FULL,       ///< 全量记录
        SAMPLED,    ///< 采样记录
        CALLBACK    ///< 回调模式
    };
    
    /**
     * @brief 构造函数
     */
    AccessMonitor();
    
    /**
     * @brief 析构函数
     */
    ~AccessMonitor();
    
    // 禁止拷贝
    AccessMonitor(const AccessMonitor&) = delete;
    AccessMonitor& operator=(const AccessMonitor&) = delete;
    
    /**
     * @brief 初始化监控器
     * @param mode 监控模式
     * @param sample_rate 采样率 (0.0-1.0，仅采样模式有效)
     * @param history_size 历史记录最大数量
     * @return 成功返回 true
     */
    bool initialize(MonitorMode mode = MonitorMode::FULL,
                   float sample_rate = 0.1f,
                   size_t history_size = 1000000);
    
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
    bool isMonitoring() const { return monitoring_active_.load(); }
    
    /**
     * @brief 记录访问
     * @param block_id 块ID
     * @param access_type 访问类型
     * @param layer_id 层ID
     * @param head_id 头ID
     * @param sequence_pos 序列位置
     */
    void recordAccess(BlockId block_id, AccessType access_type,
                     LayerId layer_id, HeadId head_id, 
                     SequencePos sequence_pos = 0);
    
    /**
     * @brief 批量记录访问
     */
    void recordAccessBatch(const std::vector<KVBlockAccess>& accesses);
    
    /**
     * @brief 设置访问回调
     * @param callback 回调函数
     */
    void setAccessCallback(AccessCallback callback);
    
    /**
     * @brief 获取块的访问历史
     * @param block_id 块ID
     * @return 访问记录列表
     */
    std::vector<AccessRecord> getBlockHistory(BlockId block_id) const;
    
    /**
     * @brief 获取最近的访问记录
     * @param count 记录数量
     * @return 访问记录列表
     */
    std::vector<AccessRecord> getRecentAccesses(size_t count) const;
    
    /**
     * @brief 获取块的访问次数
     * @param block_id 块ID
     * @return 访问次数
     */
    uint64_t getAccessCount(BlockId block_id) const;
    
    /**
     * @brief 获取块的最后访问时间
     * @param block_id 块ID
     * @return 最后访问时间戳
     */
    Timestamp getLastAccessTime(BlockId block_id) const;
    
    /**
     * @brief 获取层级的访问统计
     * @param layer_id 层ID
     * @return 访问次数
     */
    uint64_t getLayerAccessCount(LayerId layer_id) const;
    
    /**
     * @brief 获取头的访问统计
     * @param head_id 头ID
     * @return 访问次数
     */
    uint64_t getHeadAccessCount(HeadId head_id) const;
    
    /**
     * @brief 获取所有块的访问计数
     * @return 块ID -> 访问次数映射
     */
    std::unordered_map<BlockId, uint64_t> getAllAccessCounts() const;
    
    /**
     * @brief 清除历史记录
     */
    void clearHistory();
    
    /**
     * @brief 获取统计信息
     */
    struct MonitorStats {
        uint64_t total_accesses;
        uint64_t unique_blocks_accessed;
        uint64_t read_accesses;
        uint64_t write_accesses;
        size_t history_size;
        double sampling_rate;
    };
    
    MonitorStats getStats() const;
    
    /**
     * @brief 重置统计信息
     */
    void resetStats();

private:
    // 内部处理线程
    void processQueue();
    
    // 判断是否应该采样
    bool shouldSample() const;
    
    // 更新访问计数
    void updateCounts(const AccessRecord& record);
    
private:
    // 监控状态
    std::atomic<bool> monitoring_active_;
    std::atomic<bool> initialized_;
    std::atomic<bool> running_;
    MonitorMode mode_;
    float sample_rate_;
    
    // 访问历史缓冲区
    std::unique_ptr<AccessHistoryBuffer> history_buffer_;
    
    // 访问计数器
    mutable std::shared_mutex counts_mutex_;
    std::unordered_map<BlockId, uint64_t> block_access_counts_;
    std::unordered_map<BlockId, Timestamp> block_last_access_;
    std::unordered_map<LayerId, uint64_t> layer_access_counts_;
    std::unordered_map<HeadId, uint64_t> head_access_counts_;
    
    // 异步处理队列
    std::queue<AccessRecord> access_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::thread processing_thread_;
    
    // 回调
    AccessCallback access_callback_;
    
    // 统计
    std::atomic<uint64_t> total_accesses_;
    std::atomic<uint64_t> read_accesses_;
    std::atomic<uint64_t> write_accesses_;
};

}  // namespace hetkvcache

#endif  // HETKVCACHE_ACCESS_MONITOR_H