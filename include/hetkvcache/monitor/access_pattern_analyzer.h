/**
 * @file access_pattern_analyzer.h
 * @brief 访问模式分析器
 * 
 * 分析 KV Cache 的访问模式，包括：
 * - 层级访问分布
 * - 注意力头访问分布
 * - 时间序列模式
 * - 预测下一次访问
 */

#ifndef HETKVCACHE_ACCESS_PATTERN_ANALYZER_H
#define HETKVCACHE_ACCESS_PATTERN_ANALYZER_H

#include "hetkvcache/types.h"
#include "hetkvcache/monitor/access_monitor.h"
#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <deque>
#include <cmath>
#include <functional>

namespace hetkvcache {

// Hash function for pair
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

/**
 * @brief 访问模式类型
 */
enum class AccessPatternType {
    SEQUENTIAL,     ///< 顺序访问
    RANDOM,         ///< 随机访问
    STRIDE,         ///< 步长访问
    CLUSTERED,      ///< 聚类访问
    REPEATING,      ///< 重复访问
    UNKNOWN         ///< 未知模式
};

/**
 * @brief 层访问模式
 */
struct LayerAccessPattern {
    LayerId layer_id;
    AccessPatternType pattern_type;
    float access_ratio;         ///< 占总访问的比例
    uint64_t total_accesses;
    std::vector<HeadId> hot_heads;  ///< 访问频繁的头
    std::vector<HeadId> cold_heads; ///< 访问稀疏的头
    double avg_interval_ms;     ///< 平均访问间隔
    double std_interval_ms;     ///< 访问间隔标准差
    
    LayerAccessPattern()
        : layer_id(0)
        , pattern_type(AccessPatternType::UNKNOWN)
        , access_ratio(0.0f)
        , total_accesses(0)
        , avg_interval_ms(0.0)
        , std_interval_ms(0.0)
    {}
};

/**
 * @brief 时间窗口统计
 */
struct TimeWindowStats {
    Timestamp window_start;
    Timestamp window_end;
    uint64_t access_count;
    uint64_t unique_blocks;
    double avg_heat_score;
    std::vector<BlockId> top_blocks;  ///< 访问最多的块
    
    TimeWindowStats()
        : window_start(0)
        , window_end(0)
        , access_count(0)
        , unique_blocks(0)
        , avg_heat_score(0.0)
    {}
};

/**
 * @brief 访问预测结果
 */
struct AccessPrediction {
    BlockId predicted_block;
    float confidence;           ///< 预测置信度 [0.0, 1.0]
    Timestamp predicted_time;   ///< 预测访问时间
    TierType recommended_prefetch_tier;  ///< 推荐预取层级
    
    AccessPrediction()
        : predicted_block(INVALID_BLOCK_ID)
        , confidence(0.0f)
        , predicted_time(0)
        , recommended_prefetch_tier(TierType::WARM)
    {}
};

/**
 * @brief 访问模式分析器
 */
class AccessPatternAnalyzer {
public:
    /**
     * @brief 构造函数
     */
    AccessPatternAnalyzer();
    
    /**
     * @brief 析构函数
     */
    ~AccessPatternAnalyzer() = default;
    
    /**
     * @brief 初始化
     */
    void initialize(uint32_t num_layers, uint32_t num_heads);
    
    /**
     * @brief 处理访问记录
     */
    void processAccess(const AccessRecord& record);
    
    /**
     * @brief 批量处理访问记录
     */
    void processAccessBatch(const std::vector<AccessRecord>& records);
    
    /**
     * @brief 分析层级访问模式
     */
    std::vector<LayerAccessPattern> analyzeLayerPatterns();
    
    /**
     * @brief 分析注意力头访问模式
     */
    std::unordered_map<LayerId, std::vector<HeadId>> analyzeHeadPatterns();
    
    /**
     * @brief 分析时间窗口模式
     */
    std::vector<TimeWindowStats> analyzeTimePattern(
        Timestamp start_time, 
        Timestamp end_time,
        uint64_t window_size_ms = 100);
    
    /**
     * @brief 预测下一个访问
     * @param current_block 当前块ID
     * @param current_time 当前时间
     * @return 预测结果列表
     */
    std::vector<AccessPrediction> predictNextAccess(
        BlockId current_block,
        Timestamp current_time);
    
    /**
     * @brief 获取预取建议
     * @param max_count 最大建议数量
     * @return 预取建议列表
     */
    std::vector<AccessPrediction> getPrefetchSuggestions(size_t max_count = 10);
    
    /**
     * @brief 检测访问模式类型
     */
    AccessPatternType detectPatternType(const std::vector<BlockId>& block_sequence);
    
    /**
     * @brief 获取层访问统计
     */
    const std::unordered_map<LayerId, uint64_t>& getLayerAccessCounts() const {
        return layer_access_counts_;
    }
    
    /**
     * @brief 获取头访问统计
     */
    const std::unordered_map<std::pair<LayerId, HeadId>, uint64_t, pair_hash>& 
    getHeadAccessCounts() const {
        return head_access_counts_;
    }
    
    /**
     * @brief 生成分析报告
     */
    PatternReport generateReport();
    
    /**
     * @brief 重置分析器
     */
    void reset();
    
    /**
     * @brief 设置历史窗口大小
     */
    void setHistoryWindowSize(size_t size) { history_window_size_ = size; }
    
    /**
     * @brief 设置预测窗口大小
     */
    void setPredictionWindowSize(size_t size) { prediction_window_size_ = size; }

private:
    // 更新马尔可夫转移概率
    void updateTransitionProb(BlockId from, BlockId to);
    
    // 计算访问间隔统计
    void computeIntervalStats(LayerId layer_id);
    
    // 计算熵
    double computeEntropy(const std::vector<uint64_t>& counts) const;

private:
    mutable std::mutex mutex_;
    
    // 模型配置
    uint32_t num_layers_;
    uint32_t num_heads_;
    
    // 访问历史
    std::deque<AccessRecord> access_history_;
    size_t history_window_size_;
    size_t prediction_window_size_;
    
    // 层访问计数
    std::unordered_map<LayerId, uint64_t> layer_access_counts_;
    
    // 头访问计数 [layer, head] -> count
    std::unordered_map<std::pair<LayerId, HeadId>, uint64_t, pair_hash> head_access_counts_;
    
    // 块访问序列
    std::deque<BlockId> block_sequence_;
    
    // 转移概率矩阵 (简化版)
    std::unordered_map<BlockId, std::unordered_map<BlockId, uint64_t>> transition_counts_;
    
    // 层访问间隔记录
    std::unordered_map<LayerId, std::deque<Timestamp>> layer_access_times_;
    
    // 块访问时间记录
    std::unordered_map<BlockId, std::deque<Timestamp>> block_access_times_;
    
    // 分析结果缓存
    std::vector<LayerAccessPattern> cached_layer_patterns_;
    bool patterns_dirty_;
    
    // 总访问计数
    uint64_t total_accesses_;
};

}  // namespace hetkvcache

#endif  // HETKVCACHE_ACCESS_PATTERN_ANALYZER_H