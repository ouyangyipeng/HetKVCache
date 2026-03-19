/**
 * @file transfer_scheduler.h
 * @brief 传输调度器
 * 
 * 负责调度和优化数据传输任务，包括：
 * - 优先级调度
 * - 批量传输优化
 * - 带宽均衡
 * - CUDA Stream 管理
 */

#ifndef HETKVCACHE_TRANSFER_SCHEDULER_H
#define HETKVCACHE_TRANSFER_SCHEDULER_H

#include "hetkvcache/types.h"
#include <mutex>
#include <queue>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <algorithm>
#include <map>

// CUDA 前向声明
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;

namespace hetkvcache {

/**
 * @brief 传输请求
 */
struct TransferRequest {
    uint64_t request_id;
    BlockId block_id;
    void* source_ptr;
    void* target_ptr;
    size_t size;
    TierType source_tier;
    TierType target_tier;
    MigrationPriority priority;
    bool is_bidirectional;  ///< 是否双向传输
    
    TransferRequest()
        : request_id(0)
        , block_id(INVALID_BLOCK_ID)
        , source_ptr(nullptr)
        , target_ptr(nullptr)
        , size(0)
        , source_tier(TierType::HOT)
        , target_tier(TierType::WARM)
        , priority(MigrationPriority::NORMAL)
        , is_bidirectional(false)
    {}
};

/**
 * @brief 批量传输任务
 */
struct BatchTransferTask {
    std::vector<TransferRequest> requests;
    size_t total_size;
    Timestamp scheduled_time;
    cudaStream_t stream;
    
    BatchTransferTask() : total_size(0), scheduled_time(0), stream(nullptr) {}
};

/**
 * @brief 调度策略
 */
enum class ScheduleStrategy {
    FIFO,           ///< 先进先出
    PRIORITY,       ///< 优先级调度
    ROUND_ROBIN,    ///< 轮询调度
    BANDWIDTH_AWARE ///< 带宽感知调度
};

/**
 * @brief 传输调度器配置
 */
struct SchedulerConfig {
    ScheduleStrategy strategy;
    size_t max_batch_size;          ///< 最大批量大小
    size_t max_concurrent_batches;  ///< 最大并发批次数
    size_t small_transfer_threshold; ///< 小传输阈值 (合并)
    bool enable_coalescing;         ///< 是否启用合并
    
    SchedulerConfig()
        : strategy(ScheduleStrategy::PRIORITY)
        , max_batch_size(16)
        , max_concurrent_batches(4)
        , small_transfer_threshold(4 * 1024)  // 4KB
        , enable_coalescing(true)
    {}
};

/**
 * @brief 带宽统计键
 */
struct BandwidthKey {
    TierType from_tier;
    TierType to_tier;
    
    bool operator==(const BandwidthKey& other) const {
        return from_tier == other.from_tier && to_tier == other.to_tier;
    }
    
    bool operator<(const BandwidthKey& other) const {
        if (from_tier != other.from_tier) return from_tier < other.from_tier;
        return to_tier < other.to_tier;
    }
};

/**
 * @brief 传输调度器
 */
class TransferScheduler {
public:
    /**
     * @brief 构造函数
     */
    TransferScheduler();
    
    /**
     * @brief 带配置的构造函数
     */
    explicit TransferScheduler(const SchedulerConfig& config);
    
    /**
     * @brief 析构函数
     */
    ~TransferScheduler() = default;
    
    /**
     * @brief 提交传输请求
     */
    uint64_t submitRequest(const TransferRequest& request);
    
    /**
     * @brief 批量提交传输请求
     */
    std::vector<uint64_t> submitRequestBatch(const std::vector<TransferRequest>& requests);
    
    /**
     * @brief 获取下一个要执行的批量传输任务
     */
    BatchTransferTask scheduleNext();
    
    /**
     * @brief 查看下一个任务（不取出）
     */
    const BatchTransferTask* peekNext() const;
    
    /**
     * @brief 是否有待处理任务
     */
    bool hasPendingTasks() const;
    
    /**
     * @brief 获取待处理任务数量
     */
    size_t getPendingCount() const;
    
    /**
     * @brief 取消传输请求
     */
    bool cancelRequest(uint64_t request_id);
    
    /**
     * @brief 获取带宽估计
     */
    double getEstimatedBandwidth(TierType from, TierType to) const;
    
    /**
     * @brief 更新带宽统计
     */
    void updateBandwidth(TierType from, TierType to, double bandwidth_mbps);
    
    /**
     * @brief 重新排序待处理任务
     */
    void reorderTasks();
    
    /**
     * @brief 设置 CUDA Stream 池
     */
    void setStreamPool(std::vector<cudaStream_t>& streams);
    
    /**
     * @brief 分配 Stream
     */
    cudaStream_t allocateStream();
    
    /**
     * @brief 释放 Stream
     */
    void releaseStream(cudaStream_t stream);
    
    /**
     * @brief 获取统计信息
     */
    struct SchedulerStats {
        uint64_t total_requests;
        uint64_t total_batches;
        uint64_t total_bytes;
        double avg_batch_size;
        double avg_wait_time_us;
        std::map<BandwidthKey, double> bandwidth_stats;
    };
    
    SchedulerStats getStats() const;
    
    /**
     * @brief 重置统计
     */
    void resetStats();
    
    /**
     * @brief 清空队列
     */
    void clear();

private:
    // 内部请求存储
    struct InternalRequest {
        TransferRequest request;
        Timestamp submit_time;
        int queue_position;
    };
    
    // 批量合并
    BatchTransferTask coalesceRequests(std::vector<InternalRequest>& requests);
    
    // 计算优先级分数
    int computePriorityScore(const InternalRequest& req) const;
    
    // 估计传输时间
    double estimateTransferTime(const TransferRequest& req) const;

private:
    SchedulerConfig config_;
    mutable std::mutex mutex_;
    
    // 请求队列（按优先级）
    std::priority_queue<std::pair<int, uint64_t>> priority_queue_;
    
    // 所有请求
    std::unordered_map<uint64_t, InternalRequest> requests_;
    
    // 批量任务队列
    std::queue<BatchTransferTask> batch_queue_;
    
    // 带宽统计
    std::map<BandwidthKey, double> bandwidth_stats_;
    
    // Stream 池
    std::vector<cudaStream_t> stream_pool_;
    std::queue<cudaStream_t> available_streams_;
    
    // 统计
    std::atomic<uint64_t> total_requests_;
    std::atomic<uint64_t> total_batches_;
    std::atomic<uint64_t> total_bytes_;
    
    // 请求ID计数器
    std::atomic<uint64_t> next_request_id_;
};

}  // namespace hetkvcache

#endif  // HETKVCACHE_TRANSFER_SCHEDULER_H