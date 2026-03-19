/**
 * @file prefetcher.h
 * @brief 预取器
 * 
 * 基于访问模式预测，提前将可能被访问的数据加载到更快的存储层级。
 */

#ifndef HETKVCACHE_PREFETCHER_H
#define HETKVCACHE_PREFETCHER_H

#include "hetkvcache/types.h"
#include "hetkvcache/monitor/access_pattern_analyzer.h"
#include <mutex>
#include <deque>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <memory>

namespace hetkvcache {

// 前向声明
class MigrationEngine;

/**
 * @brief 预取策略
 */
enum class PrefetchStrategy {
    SEQUENTIAL,     ///< 顺序预取
    MARKOV,         ///< 基于马尔可夫链的预测
    ORACLE,         ///< 完美预知（仅用于测试）
    ADAPTIVE        ///< 自适应策略
};

/**
 * @brief 预取请求
 */
struct PrefetchRequest {
    BlockId block_id;
    TierType target_tier;
    float confidence;
    Timestamp deadline;  ///< 预期访问时间
    
    PrefetchRequest()
        : block_id(INVALID_BLOCK_ID)
        , target_tier(TierType::HOT)
        , confidence(0.0f)
        , deadline(0)
    {}
    
    bool operator<(const PrefetchRequest& other) const {
        return confidence < other.confidence;  // 大顶堆
    }
};

/**
 * @brief 预取统计
 */
struct PrefetchStats {
    uint64_t total_prefetches;
    uint64_t useful_prefetches;   ///< 实际被访问的预取
    uint64_t useless_prefetches;  ///< 未被访问的预取
    uint64_t evicted_prefetches;  ///< 预取后被驱逐的数据
    double hit_rate;
    double avg_confidence;
    double avg_latency_reduction_ms;
    
    PrefetchStats()
        : total_prefetches(0)
        , useful_prefetches(0)
        , useless_prefetches(0)
        , evicted_prefetches(0)
        , hit_rate(0.0)
        , avg_confidence(0.0)
        , avg_latency_reduction_ms(0.0)
    {}
};

/**
 * @brief 预取器配置
 */
struct PrefetcherConfig {
    PrefetchStrategy strategy;
    float confidence_threshold;    ///< 预取置信度阈值
    size_t max_prefetch_window;    ///< 最大预取窗口
    size_t max_prefetch_bytes;     ///< 最大预取字节数
    bool enable_background_prefetch;
    uint64_t prefetch_lookahead;   ///< 预取前瞻步数
    
    PrefetcherConfig()
        : strategy(PrefetchStrategy::ADAPTIVE)
        , confidence_threshold(0.6f)
        , max_prefetch_window(10)
        , max_prefetch_bytes(64 * 1024 * 1024)  // 64MB
        , enable_background_prefetch(true)
        , prefetch_lookahead(3)
    {}
};

/**
 * @brief 预取器
 */
class Prefetcher {
public:
    /**
     * @brief 构造函数
     */
    Prefetcher();
    
    /**
     * @brief 带配置的构造函数
     */
    explicit Prefetcher(const PrefetcherConfig& config);
    
    /**
     * @brief 析构函数
     */
    ~Prefetcher();
    
    /**
     * @brief 初始化
     */
    void initialize(MigrationEngine* migration_engine,
                   AccessPatternAnalyzer* pattern_analyzer);
    
    /**
     * @brief 更新预测模型
     */
    void updateModel(const AccessRecord& record);
    
    /**
     * @brief 更新预测模型（批量）
     */
    void updateModelBatch(const std::vector<AccessRecord>& records);
    
    /**
     * @brief 触发预取
     * @param current_block 当前访问的块
     * @param current_time 当前时间
     * @return 预取的块列表
     */
    std::vector<BlockId> triggerPrefetch(BlockId current_block, Timestamp current_time);
    
    /**
     * @brief 获取预取建议
     */
    std::vector<PrefetchRequest> getPrefetchSuggestions(BlockId current_block);
    
    /**
     * @brief 记录预取命中
     */
    void recordPrefetchHit(BlockId block_id);
    
    /**
     * @brief 记录预取未命中
     */
    void recordPrefetchMiss(BlockId block_id);
    
    /**
     * @brief 取消预取
     */
    void cancelPrefetch(BlockId block_id);
    
    /**
     * @brief 获取统计信息
     */
    PrefetchStats getStats() const;
    
    /**
     * @brief 重置统计
     */
    void resetStats();
    
    /**
     * @brief 清空预取队列
     */
    void clearQueue();
    
    /**
     * @brief 设置配置
     */
    void setConfig(const PrefetcherConfig& config);
    
    /**
     * @brief 获取配置
     */
    const PrefetcherConfig& getConfig() const { return config_; }

private:
    // 预测下一个访问块
    std::vector<BlockId> predictNextBlocks(BlockId current_block);
    
    // 计算预取优先级
    float computePrefetchPriority(BlockId block_id, Timestamp current_time);
    
    // 执行预取
    void executePrefetch(const PrefetchRequest& request);
    
    // 清理过期的预取
    void cleanupExpiredPrefetches(Timestamp current_time);
    
    // 马尔可夫预测
    BlockId markovPredict(BlockId current_block);

private:
    PrefetcherConfig config_;
    mutable std::mutex mutex_;
    
    // 关联组件
    MigrationEngine* migration_engine_;
    AccessPatternAnalyzer* pattern_analyzer_;
    
    // 预取队列
    std::priority_queue<PrefetchRequest> prefetch_queue_;
    
    // 已预取的块
    std::unordered_map<BlockId, Timestamp> prefetched_blocks_;
    
    // 马尔可夫转移矩阵（简化版）
    std::unordered_map<BlockId, std::unordered_map<BlockId, uint64_t>> transition_matrix_;
    
    // 访问历史
    std::deque<BlockId> recent_accesses_;
    
    // 统计
    PrefetchStats stats_;
    std::atomic<uint64_t> total_prefetches_;
    std::atomic<uint64_t> useful_prefetches_;
    
    // 初始化状态
    std::atomic<bool> initialized_;
};

}  // namespace hetkvcache

#endif  // HETKVCACHE_PREFETCHER_H