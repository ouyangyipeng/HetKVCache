/**
 * @file prefetcher.cpp
 * @brief 预取器实现
 */

#include "hetkvcache/migration/prefetcher.h"
#include "hetkvcache/migration/migration_engine.h"
#include <algorithm>
#include <random>

namespace hetkvcache {

Prefetcher::Prefetcher()
    : migration_engine_(nullptr)
    , pattern_analyzer_(nullptr)
    , total_prefetches_(0)
    , useful_prefetches_(0)
    , initialized_(false)
{
}

Prefetcher::Prefetcher(const PrefetcherConfig& config)
    : config_(config)
    , migration_engine_(nullptr)
    , pattern_analyzer_(nullptr)
    , total_prefetches_(0)
    , useful_prefetches_(0)
    , initialized_(false)
{
}

Prefetcher::~Prefetcher() {
}

void Prefetcher::initialize(MigrationEngine* migration_engine,
                           AccessPatternAnalyzer* pattern_analyzer) {
    std::lock_guard<std::mutex> lock(mutex_);
    migration_engine_ = migration_engine;
    pattern_analyzer_ = pattern_analyzer;
    initialized_ = true;
}

void Prefetcher::updateModel(const AccessRecord& record) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 更新访问历史
    recent_accesses_.push_back(record.block_id);
    if (recent_accesses_.size() > config_.max_prefetch_window) {
        recent_accesses_.pop_front();
    }
    
    // 更新转移矩阵
    if (recent_accesses_.size() >= 2) {
        BlockId prev_block = recent_accesses_[recent_accesses_.size() - 2];
        transition_matrix_[prev_block][record.block_id]++;
    }
}

void Prefetcher::updateModelBatch(const std::vector<AccessRecord>& records) {
    for (const auto& record : records) {
        updateModel(record);
    }
}

std::vector<BlockId> Prefetcher::triggerPrefetch(BlockId current_block, 
                                                 Timestamp current_time) {
    std::vector<BlockId> prefetched;
    
    auto predictions = predictNextBlocks(current_block);
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (BlockId next_block : predictions) {
        // 检查是否已预取
        if (prefetched_blocks_.count(next_block) > 0) {
            continue;
        }
        
        // 创建预取请求
        PrefetchRequest request;
        request.block_id = next_block;
        request.target_tier = TierType::WARM;
        request.confidence = computePrefetchPriority(next_block, current_time);
        request.deadline = current_time + 10000000;  // 10ms 后
        
        if (request.confidence >= config_.confidence_threshold) {
            executePrefetch(request);
            prefetched.push_back(next_block);
            prefetched_blocks_[next_block] = current_time;
            total_prefetches_++;
        }
    }
    
    // 清理过期预取
    cleanupExpiredPrefetches(current_time);
    
    return prefetched;
}

std::vector<PrefetchRequest> Prefetcher::getPrefetchSuggestions(BlockId current_block) {
    std::vector<PrefetchRequest> suggestions;
    
    auto predictions = predictNextBlocks(current_block);
    Timestamp current_time = getCurrentTimestamp();
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (BlockId next_block : predictions) {
        PrefetchRequest request;
        request.block_id = next_block;
        request.target_tier = TierType::WARM;
        request.confidence = computePrefetchPriority(next_block, current_time);
        request.deadline = current_time + 10000000;
        
        if (request.confidence >= config_.confidence_threshold) {
            suggestions.push_back(request);
        }
    }
    
    return suggestions;
}

void Prefetcher::recordPrefetchHit(BlockId block_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (prefetched_blocks_.count(block_id) > 0) {
        useful_prefetches_++;
        prefetched_blocks_.erase(block_id);
    }
}

void Prefetcher::recordPrefetchMiss(BlockId block_id) {
    // 访问了未预取的块，可能需要调整预测模型
}

void Prefetcher::cancelPrefetch(BlockId block_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    prefetched_blocks_.erase(block_id);
}

PrefetchStats Prefetcher::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    PrefetchStats stats;
    stats.total_prefetches = total_prefetches_;
    stats.useful_prefetches = useful_prefetches_;
    stats.useless_prefetches = stats.total_prefetches - stats.useful_prefetches;
    
    if (stats.total_prefetches > 0) {
        stats.hit_rate = static_cast<double>(stats.useful_prefetches) / stats.total_prefetches;
    }
    
    stats.avg_confidence = 0.0;  // TODO: 跟踪
    stats.avg_latency_reduction_ms = 0.0;  // TODO: 测量
    
    return stats;
}

void Prefetcher::resetStats() {
    total_prefetches_ = 0;
    useful_prefetches_ = 0;
}

void Prefetcher::clearQueue() {
    std::lock_guard<std::mutex> lock(mutex_);
    prefetched_blocks_.clear();
}

void Prefetcher::setConfig(const PrefetcherConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
}

std::vector<BlockId> Prefetcher::predictNextBlocks(BlockId current_block) {
    std::vector<BlockId> predictions;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    switch (config_.strategy) {
        case PrefetchStrategy::SEQUENTIAL: {
            // 顺序预取：预测后续块
            for (size_t i = 1; i <= config_.prefetch_lookahead; i++) {
                predictions.push_back(current_block + i);
            }
            break;
        }
        
        case PrefetchStrategy::MARKOV: {
            // 马尔可夫预测
            BlockId predicted = markovPredict(current_block);
            if (predicted != INVALID_BLOCK_ID) {
                predictions.push_back(predicted);
            }
            break;
        }
        
        case PrefetchStrategy::ADAPTIVE: {
            // 自适应：结合多种策略
            // 首先尝试马尔可夫
            BlockId predicted = markovPredict(current_block);
            if (predicted != INVALID_BLOCK_ID) {
                predictions.push_back(predicted);
            }
            
            // 然后添加顺序预测
            for (size_t i = 1; i <= config_.prefetch_lookahead && 
                 predictions.size() < config_.max_prefetch_window; i++) {
                BlockId seq_block = current_block + i;
                if (std::find(predictions.begin(), predictions.end(), seq_block) == 
                    predictions.end()) {
                    predictions.push_back(seq_block);
                }
            }
            break;
        }
        
        case PrefetchStrategy::ORACLE: {
            // 完美预测（仅用于测试）
            // 需要外部注入预测结果
            break;
        }
        
        default:
            break;
    }
    
    return predictions;
}

float Prefetcher::computePrefetchPriority(BlockId block_id, Timestamp current_time) {
    // 基于转移概率计算优先级
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (recent_accesses_.empty()) {
        return 0.0f;
    }
    
    BlockId last_block = recent_accesses_.back();
    auto trans_it = transition_matrix_.find(last_block);
    if (trans_it == transition_matrix_.end()) {
        return 0.0f;
    }
    
    auto block_it = trans_it->second.find(block_id);
    if (block_it == trans_it->second.end()) {
        return 0.0f;
    }
    
    // 计算转移概率
    uint64_t total_transitions = 0;
    for (const auto& [_, count] : trans_it->second) {
        total_transitions += count;
    }
    
    if (total_transitions == 0) {
        return 0.0f;
    }
    
    return static_cast<float>(block_it->second) / total_transitions;
}

void Prefetcher::executePrefetch(const PrefetchRequest& request) {
    if (migration_engine_) {
        // 确定源层级
        TierType source_tier = TierType::COLD;  // 假设从冷层预取
        
        migration_engine_->submitMigration(
            request.block_id,
            source_tier,
            request.target_tier,
            MigrationPriority::LOW
        );
    }
}

void Prefetcher::cleanupExpiredPrefetches(Timestamp current_time) {
    // 清理超过 1 秒的预取记录
    Timestamp threshold = current_time - 1000000000;  // 1秒
    
    for (auto it = prefetched_blocks_.begin(); it != prefetched_blocks_.end(); ) {
        if (it->second < threshold) {
            it = prefetched_blocks_.erase(it);
        } else {
            ++it;
        }
    }
}

BlockId Prefetcher::markovPredict(BlockId current_block) {
    auto trans_it = transition_matrix_.find(current_block);
    if (trans_it == transition_matrix_.end()) {
        return INVALID_BLOCK_ID;
    }
    
    // 找出转移概率最高的块
    BlockId best_block = INVALID_BLOCK_ID;
    uint64_t max_count = 0;
    
    for (const auto& [block, count] : trans_it->second) {
        if (count > max_count) {
            max_count = count;
            best_block = block;
        }
    }
    
    return best_block;
}

}  // namespace hetkvcache