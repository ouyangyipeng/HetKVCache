/**
 * @file heat_evaluator.h
 * @brief KV Cache 热度评估器
 * 
 * 基于访问模式评估每个 KV Block 的热度分数，
 * 决定数据应在哪个存储层级（Hot/Warm/Cold）。
 * 
 * 热度评估算法：
 * heat_score(t) = α * frequency_score + β * recency_score + γ * layer_importance
 */

#ifndef HETKVCACHE_HEAT_EVALUATOR_H
#define HETKVCACHE_HEAT_EVALUATOR_H

#include "hetkvcache/types.h"
#include "hetkvcache/monitor/access_monitor.h"
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>

namespace hetkvcache {

/**
 * @brief 热度评估参数
 */
struct HeatEvalParams {
    float alpha;        ///< 访问频率权重 (默认 0.5)
    float beta;         ///< 最近访问时间权重 (默认 0.3)
    float gamma;        ///< 层重要性权重 (默认 0.2)
    float lambda;       ///< 指数衰减系数 (默认 0.01)
    float hot_threshold;  ///< 热数据阈值 (默认 0.7)
    float warm_threshold; ///< 温数据阈值 (默认 0.3)
    
    HeatEvalParams()
        : alpha(0.5f)
        , beta(0.3f)
        , gamma(0.2f)
        , lambda(0.01f)
        , hot_threshold(0.7f)
        , warm_threshold(0.3f)
    {}
};

/**
 * @brief 块热度信息
 */
struct BlockHeatInfo {
    BlockId block_id;
    float heat_score;           ///< 综合热度分数 [0.0, 1.0]
    float frequency_score;      ///< 频率分数
    float recency_score;        ///< 新近度分数
    float layer_importance;     ///< 层重要性分数
    TierType current_tier;      ///< 当前层级
    TierType recommended_tier;  ///< 推荐层级
    Timestamp last_access_time;
    uint64_t access_count;
    Timestamp last_eval_time;   ///< 最后评估时间
    
    BlockHeatInfo()
        : block_id(INVALID_BLOCK_ID)
        , heat_score(0.0f)
        , frequency_score(0.0f)
        , recency_score(0.0f)
        , layer_importance(1.0f)
        , current_tier(TierType::HOT)
        , recommended_tier(TierType::HOT)
        , last_access_time(0)
        , access_count(0)
        , last_eval_time(0)
    {}
};

/**
 * @brief 迁移建议
 */
struct MigrationRecommendation {
    BlockId block_id;
    TierType from_tier;
    TierType to_tier;
    float heat_score;
    int priority;  ///< 优先级，数值越大越优先
    
    MigrationRecommendation()
        : block_id(INVALID_BLOCK_ID)
        , from_tier(TierType::HOT)
        , to_tier(TierType::WARM)
        , heat_score(0.0f)
        , priority(0)
    {}
    
    bool operator<(const MigrationRecommendation& other) const {
        return priority < other.priority;  // 大顶堆
    }
};

/**
 * @brief 热度评估器类
 * 
 * 实现基于多维度热度评分的 KV Block 分类。
 */
class HeatEvaluator {
public:
    /**
     * @brief 构造函数
     */
    HeatEvaluator();
    
    /**
     * @brief 带参数的构造函数
     */
    explicit HeatEvaluator(const HeatEvalParams& params);
    
    /**
     * @brief 析构函数
     */
    ~HeatEvaluator() = default;
    
    /**
     * @brief 设置评估参数
     */
    void setParams(const HeatEvalParams& params);
    
    /**
     * @brief 获取评估参数
     */
    const HeatEvalParams& getParams() const { return params_; }
    
    /**
     * @brief 处理访问记录
     * @param record 访问记录
     */
    void processAccess(const AccessRecord& record);
    
    /**
     * @brief 批量处理访问记录
     */
    void processAccessBatch(const std::vector<AccessRecord>& records);
    
    /**
     * @brief 更新指定块的热度分数
     * @param block_id 块ID
     * @param current_time 当前时间戳
     */
    void updateBlockHeat(BlockId block_id, Timestamp current_time);
    
    /**
     * @brief 更新所有块的热度分数
     * @param current_time 当前时间戳
     */
    void updateAllHeatScores(Timestamp current_time);
    
    /**
     * @brief 获取块的热度分数
     * @param block_id 块ID
     * @return 热度分数，如果块不存在返回 0.0
     */
    float getHeatScore(BlockId block_id) const;
    
    /**
     * @brief 获取块的完整热度信息
     */
    BlockHeatInfo getBlockHeatInfo(BlockId block_id) const;
    
    /**
     * @brief 根据热度分数分类层级
     * @param heat_score 热度分数
     * @return 推荐的存储层级
     */
    TierType classifyTier(float heat_score) const;
    
    /**
     * @brief 获取块的推荐层级
     * @param block_id 块ID
     * @return 推荐的存储层级
     */
    TierType getRecommendedTier(BlockId block_id) const;
    
    /**
     * @brief 获取需要迁移的块列表
     * @param current_tier 当前层级
     * @param target_tier 目标层级
     * @param max_count 最大数量
     * @return 迁移建议列表，按优先级排序
     */
    std::vector<MigrationRecommendation> getBlocksForMigration(
        TierType current_tier, 
        TierType target_tier,
        size_t max_count = 100);
    
    /**
     * @brief 获取所有热块
     */
    std::vector<BlockId> getHotBlocks() const;
    
    /**
     * @brief 获取所有温块
     */
    std::vector<BlockId> getWarmBlocks() const;
    
    /**
     * @brief 获取所有冷块
     */
    std::vector<BlockId> getColdBlocks() const;
    
    /**
     * @brief 设置块当前所在层级
     */
    void setBlockTier(BlockId block_id, TierType tier);
    
    /**
     * @brief 注册新块
     * @param block_id 块ID
     * @param layer_id Transformer 层ID
     */
    void registerBlock(BlockId block_id, LayerId layer_id);
    
    /**
     * @brief 注销块
     */
    void unregisterBlock(BlockId block_id);
    
    /**
     * @brief 清除所有热度信息
     */
    void clear();
    
    /**
     * @brief 获取热度分布统计
     */
    struct HeatDistribution {
        uint64_t total_blocks;
        uint64_t hot_blocks;
        uint64_t warm_blocks;
        uint64_t cold_blocks;
        float avg_heat_score;
        float min_heat_score;
        float max_heat_score;
        float std_dev;
    };
    
    HeatDistribution getHeatDistribution() const;
    
    /**
     * @brief 设置层的最大数量（用于计算层重要性）
     */
    void setNumLayers(uint32_t num_layers) { num_layers_ = num_layers; }
    
    /**
     * @brief 设置头的最大数量
     */
    void setNumHeads(uint32_t num_heads) { num_heads_ = num_heads; }

private:
    // 计算频率分数
    float computeFrequencyScore(uint64_t access_count, Timestamp current_time) const;
    
    // 计算新近度分数
    float computeRecencyScore(Timestamp last_access, Timestamp current_time) const;
    
    // 计算层重要性分数
    float computeLayerImportance(LayerId layer_id) const;
    
    // 应用指数衰减
    float applyDecay(float score, Timestamp time_diff_ns) const;
    
    // 更新迁移优先级
    int computeMigrationPriority(const BlockHeatInfo& info) const;

private:
    HeatEvalParams params_;
    mutable std::shared_mutex heat_mutex_;
    
    // 块热度信息
    std::unordered_map<BlockId, BlockHeatInfo> block_heat_info_;
    
    // 块到层的映射 - O(1)快速查找
    std::unordered_map<BlockId, LayerId> block_to_layer_;
    
    // 层到块的映射
    std::unordered_map<LayerId, std::vector<BlockId>> layer_blocks_;
    
    // 层级到块的映射
    std::unordered_map<TierType, std::vector<BlockId>> tier_blocks_;
    
    // 模型配置
    uint32_t num_layers_;
    uint32_t num_heads_;
    
    // 统计信息
    std::atomic<uint64_t> total_evaluations_;
};

// ============================================================================
// 内联实现
// ============================================================================

inline float HeatEvaluator::computeFrequencyScore(
    uint64_t access_count, Timestamp current_time) const {
    // 使用对数缩放避免极端值
    return std::log1p(static_cast<float>(access_count)) / std::log(100.0f);
}

inline float HeatEvaluator::computeRecencyScore(
    Timestamp last_access, Timestamp current_time) const {
    if (last_access == 0) return 0.0f;
    
    Timestamp time_diff_ns = current_time - last_access;
    // 转换为秒
    double time_diff_sec = static_cast<double>(time_diff_ns) / 1e9;
    // 指数衰减，半衰期约 10 秒
    return std::exp(-params_.lambda * time_diff_sec);
}

inline float HeatEvaluator::computeLayerImportance(LayerId layer_id) const {
    // 浅层更重要（靠近输入的层）
    // 使用逆序衰减
    if (num_layers_ == 0) return 1.0f;
    return 1.0f - static_cast<float>(layer_id) / static_cast<float>(num_layers_);
}

inline float HeatEvaluator::applyDecay(float score, Timestamp time_diff_ns) const {
    double time_diff_sec = static_cast<double>(time_diff_ns) / 1e9;
    return score * std::exp(-params_.lambda * time_diff_sec);
}

inline TierType HeatEvaluator::classifyTier(float heat_score) const {
    if (heat_score >= params_.hot_threshold) {
        return TierType::HOT;
    } else if (heat_score >= params_.warm_threshold) {
        return TierType::WARM;
    } else {
        return TierType::COLD;
    }
}

}  // namespace hetkvcache

#endif  // HETKVCACHE_HEAT_EVALUATOR_H