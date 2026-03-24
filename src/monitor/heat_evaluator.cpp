/**
 * @file heat_evaluator.cpp
 * @brief KV Cache 热度评估器实现
 */

#include "hetkvcache/monitor/heat_evaluator.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <omp.h>  // OpenMP并行化

namespace hetkvcache {

HeatEvaluator::HeatEvaluator()
    : params_()
    , num_layers_(0)
    , num_heads_(0)
    , total_evaluations_(0)
{
}

HeatEvaluator::HeatEvaluator(const HeatEvalParams& params)
    : params_(params)
    , num_layers_(0)
    , num_heads_(0)
    , total_evaluations_(0)
{
}

void HeatEvaluator::setParams(const HeatEvalParams& params) {
    params_ = params;
}

void HeatEvaluator::processAccess(const AccessRecord& record) {
    std::unique_lock<std::shared_mutex> lock(heat_mutex_);
    
    auto& info = block_heat_info_[record.block_id];
    info.block_id = record.block_id;
    info.last_access_time = record.timestamp;
    info.access_count++;
    
    // 记录层ID用于层重要性计算 - 使用哈希表快速查找
    block_to_layer_[record.block_id] = record.layer_id;
}

void HeatEvaluator::processAccessBatch(const std::vector<AccessRecord>& records) {
    std::unique_lock<std::shared_mutex> lock(heat_mutex_);
    
    for (const auto& record : records) {
        auto& info = block_heat_info_[record.block_id];
        info.block_id = record.block_id;
        info.last_access_time = record.timestamp;
        info.access_count++;
        block_to_layer_[record.block_id] = record.layer_id;
    }
}

void HeatEvaluator::updateBlockHeat(BlockId block_id, Timestamp current_time) {
    std::unique_lock<std::shared_mutex> lock(heat_mutex_);
    
    auto it = block_heat_info_.find(block_id);
    if (it == block_heat_info_.end()) {
        return;
    }
    
    auto& info = it->second;
    
    // 计算各维度分数
    info.frequency_score = computeFrequencyScore(info.access_count, current_time);
    info.recency_score = computeRecencyScore(info.last_access_time, current_time);
    
    // 使用哈希表快速查找层ID - O(1) 而非 O(n)
    LayerId layer_id = 0;
    auto layer_it = block_to_layer_.find(block_id);
    if (layer_it != block_to_layer_.end()) {
        layer_id = layer_it->second;
    }
    info.layer_importance = computeLayerImportance(layer_id);
    
    // 综合热度分数
    info.heat_score = params_.alpha * info.frequency_score +
                      params_.beta * info.recency_score +
                      params_.gamma * info.layer_importance;
    
    // 限制在 [0, 1] 范围
    info.heat_score = std::max(0.0f, std::min(1.0f, info.heat_score));
    
    // 分类层级
    info.recommended_tier = classifyTier(info.heat_score);
    info.last_eval_time = current_time;
    
    total_evaluations_++;
}

void HeatEvaluator::updateAllHeatScores(Timestamp current_time) {
    // 第一阶段：收集需要更新的块ID（读锁）
    std::vector<BlockId> blocks_to_update;
    std::vector<std::pair<BlockId, BlockHeatInfo*>> block_info_ptrs;
    
    {
        std::shared_lock<std::shared_mutex> lock(heat_mutex_);
        blocks_to_update.reserve(block_heat_info_.size());
        for (auto& [block_id, info] : block_heat_info_) {
            blocks_to_update.push_back(block_id);
        }
    }
    
    // 第二阶段：并行计算热度分数（无锁）
    // 使用OpenMP并行处理
    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < blocks_to_update.size(); i++) {
        BlockId block_id = blocks_to_update[i];
        
        // 获取块信息（需要锁）
        BlockHeatInfo local_info;
        LayerId layer_id = 0;
        {
            std::shared_lock<std::shared_mutex> lock(heat_mutex_);
            auto it = block_heat_info_.find(block_id);
            if (it == block_heat_info_.end()) continue;
            local_info = it->second;
            
            auto layer_it = block_to_layer_.find(block_id);
            if (layer_it != block_to_layer_.end()) {
                layer_id = layer_it->second;
            }
        }
        
        // 计算热度分数（无锁，纯计算）
        float frequency_score = computeFrequencyScore(local_info.access_count, current_time);
        float recency_score = computeRecencyScore(local_info.last_access_time, current_time);
        float layer_importance = computeLayerImportance(layer_id);
        
        float heat_score = params_.alpha * frequency_score +
                          params_.beta * recency_score +
                          params_.gamma * layer_importance;
        heat_score = std::max(0.0f, std::min(1.0f, heat_score));
        TierType recommended_tier = classifyTier(heat_score);
        
        // 写回结果（需要锁）
        {
            std::unique_lock<std::shared_mutex> lock(heat_mutex_);
            auto it = block_heat_info_.find(block_id);
            if (it != block_heat_info_.end()) {
                it->second.frequency_score = frequency_score;
                it->second.recency_score = recency_score;
                it->second.layer_importance = layer_importance;
                it->second.heat_score = heat_score;
                it->second.recommended_tier = recommended_tier;
                it->second.last_eval_time = current_time;
            }
        }
    }
    
    total_evaluations_ += blocks_to_update.size();
}

float HeatEvaluator::getHeatScore(BlockId block_id) const {
    std::shared_lock<std::shared_mutex> lock(heat_mutex_);
    auto it = block_heat_info_.find(block_id);
    return (it != block_heat_info_.end()) ? it->second.heat_score : 0.0f;
}

BlockHeatInfo HeatEvaluator::getBlockHeatInfo(BlockId block_id) const {
    std::shared_lock<std::shared_mutex> lock(heat_mutex_);
    auto it = block_heat_info_.find(block_id);
    return (it != block_heat_info_.end()) ? it->second : BlockHeatInfo();
}

TierType HeatEvaluator::getRecommendedTier(BlockId block_id) const {
    std::shared_lock<std::shared_mutex> lock(heat_mutex_);
    auto it = block_heat_info_.find(block_id);
    return (it != block_heat_info_.end()) ? it->second.recommended_tier : TierType::WARM;
}

std::vector<MigrationRecommendation> HeatEvaluator::getBlocksForMigration(
    TierType current_tier, TierType target_tier, size_t max_count) {
    
    std::vector<MigrationRecommendation> recommendations;
    
    std::shared_lock<std::shared_mutex> lock(heat_mutex_);
    
    for (const auto& [block_id, info] : block_heat_info_) {
        if (info.current_tier == current_tier && 
            info.recommended_tier == target_tier) {
            
            MigrationRecommendation rec;
            rec.block_id = block_id;
            rec.from_tier = current_tier;
            rec.to_tier = target_tier;
            rec.heat_score = info.heat_score;
            rec.priority = computeMigrationPriority(info);
            
            recommendations.push_back(rec);
        }
    }
    
    // 按优先级排序
    std::sort(recommendations.begin(), recommendations.end(),
              [](const MigrationRecommendation& a, const MigrationRecommendation& b) {
                  return a.priority > b.priority;
              });
    
    if (recommendations.size() > max_count) {
        recommendations.resize(max_count);
    }
    
    return recommendations;
}

std::vector<BlockId> HeatEvaluator::getHotBlocks() const {
    std::vector<BlockId> blocks;
    std::shared_lock<std::shared_mutex> lock(heat_mutex_);
    
    for (const auto& [block_id, info] : block_heat_info_) {
        if (info.heat_score >= params_.hot_threshold) {
            blocks.push_back(block_id);
        }
    }
    return blocks;
}

std::vector<BlockId> HeatEvaluator::getWarmBlocks() const {
    std::vector<BlockId> blocks;
    std::shared_lock<std::shared_mutex> lock(heat_mutex_);
    
    for (const auto& [block_id, info] : block_heat_info_) {
        if (info.heat_score >= params_.warm_threshold &&
            info.heat_score < params_.hot_threshold) {
            blocks.push_back(block_id);
        }
    }
    return blocks;
}

std::vector<BlockId> HeatEvaluator::getColdBlocks() const {
    std::vector<BlockId> blocks;
    std::shared_lock<std::shared_mutex> lock(heat_mutex_);
    
    for (const auto& [block_id, info] : block_heat_info_) {
        if (info.heat_score < params_.warm_threshold) {
            blocks.push_back(block_id);
        }
    }
    return blocks;
}

void HeatEvaluator::setBlockTier(BlockId block_id, TierType tier) {
    std::unique_lock<std::shared_mutex> lock(heat_mutex_);
    
    auto it = block_heat_info_.find(block_id);
    if (it != block_heat_info_.end()) {
        it->second.current_tier = tier;
        
        // 更新层级映射
        for (auto& [t, blocks] : tier_blocks_) {
            blocks.erase(std::remove(blocks.begin(), blocks.end(), block_id), blocks.end());
        }
        tier_blocks_[tier].push_back(block_id);
    }
}

void HeatEvaluator::registerBlock(BlockId block_id, LayerId layer_id) {
    std::unique_lock<std::shared_mutex> lock(heat_mutex_);
    
    BlockHeatInfo info;
    info.block_id = block_id;
    info.layer_importance = computeLayerImportance(layer_id);
    info.current_tier = TierType::HOT;
    info.recommended_tier = TierType::HOT;
    
    block_heat_info_[block_id] = info;
    layer_blocks_[layer_id].push_back(block_id);
    tier_blocks_[TierType::HOT].push_back(block_id);
}

void HeatEvaluator::unregisterBlock(BlockId block_id) {
    std::unique_lock<std::shared_mutex> lock(heat_mutex_);
    
    block_heat_info_.erase(block_id);
    
    for (auto& [_, blocks] : layer_blocks_) {
        blocks.erase(std::remove(blocks.begin(), blocks.end(), block_id), blocks.end());
    }
    
    for (auto& [_, blocks] : tier_blocks_) {
        blocks.erase(std::remove(blocks.begin(), blocks.end(), block_id), blocks.end());
    }
}

void HeatEvaluator::clear() {
    std::unique_lock<std::shared_mutex> lock(heat_mutex_);
    block_heat_info_.clear();
    layer_blocks_.clear();
    tier_blocks_.clear();
    total_evaluations_ = 0;
}

HeatEvaluator::HeatDistribution HeatEvaluator::getHeatDistribution() const {
    HeatDistribution dist;
    std::shared_lock<std::shared_mutex> lock(heat_mutex_);
    
    dist.total_blocks = block_heat_info_.size();
    if (dist.total_blocks == 0) {
        return dist;
    }
    
    std::vector<float> scores;
    scores.reserve(block_heat_info_.size());
    
    for (const auto& [_, info] : block_heat_info_) {
        scores.push_back(info.heat_score);
        
        if (info.heat_score >= params_.hot_threshold) {
            dist.hot_blocks++;
        } else if (info.heat_score >= params_.warm_threshold) {
            dist.warm_blocks++;
        } else {
            dist.cold_blocks++;
        }
    }
    
    // 计算统计量
    dist.avg_heat_score = std::accumulate(scores.begin(), scores.end(), 0.0f) / scores.size();
    dist.min_heat_score = *std::min_element(scores.begin(), scores.end());
    dist.max_heat_score = *std::max_element(scores.begin(), scores.end());
    
    // 计算标准差
    float variance = 0.0f;
    for (float score : scores) {
        variance += (score - dist.avg_heat_score) * (score - dist.avg_heat_score);
    }
    dist.std_dev = std::sqrt(variance / scores.size());
    
    return dist;
}

int HeatEvaluator::computeMigrationPriority(const BlockHeatInfo& info) const {
    // 优先级基于热度差异和访问频率
    int priority = 0;
    
    // 热度分数越高，降级优先级越低，升级优先级越高
    if (info.recommended_tier < info.current_tier) {
        // 升级（从冷层到热层）
        priority = static_cast<int>(info.heat_score * 100);
    } else {
        // 降级（从热层到冷层）
        priority = static_cast<int>((1.0f - info.heat_score) * 100);
    }
    
    // 访问次数加成
    priority += std::min(static_cast<int>(info.access_count), 50);
    
    return priority;
}

}  // namespace hetkvcache