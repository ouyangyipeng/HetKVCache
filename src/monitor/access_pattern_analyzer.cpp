/**
 * @file access_pattern_analyzer.cpp
 * @brief 访问模式分析器实现
 */

#include "hetkvcache/monitor/access_pattern_analyzer.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace hetkvcache {

AccessPatternAnalyzer::AccessPatternAnalyzer()
    : num_layers_(0)
    , num_heads_(0)
    , history_window_size_(10000)
    , prediction_window_size_(100)
    , patterns_dirty_(true)
    , total_accesses_(0)
{
}

void AccessPatternAnalyzer::initialize(uint32_t num_layers, uint32_t num_heads) {
    std::lock_guard<std::mutex> lock(mutex_);
    num_layers_ = num_layers;
    num_heads_ = num_heads;
    layer_access_counts_.clear();
    head_access_counts_.clear();
    access_history_.clear();
    block_sequence_.clear();
    transition_counts_.clear();
    total_accesses_ = 0;
    patterns_dirty_ = true;
}

void AccessPatternAnalyzer::processAccess(const AccessRecord& record) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 更新访问历史
    access_history_.push_back(record);
    if (access_history_.size() > history_window_size_) {
        access_history_.pop_front();
    }
    
    // 更新块序列
    if (!block_sequence_.empty()) {
        updateTransitionProb(block_sequence_.back(), record.block_id);
    }
    block_sequence_.push_back(record.block_id);
    if (block_sequence_.size() > history_window_size_) {
        block_sequence_.pop_front();
    }
    
    // 更新计数
    layer_access_counts_[record.layer_id]++;
    auto head_key = std::make_pair(record.layer_id, record.head_id);
    head_access_counts_[head_key]++;
    
    // 更新块访问时间
    block_access_times_[record.block_id].push_back(record.timestamp);
    
    // 更新层访问时间
    layer_access_times_[record.layer_id].push_back(record.timestamp);
    
    total_accesses_++;
    patterns_dirty_ = true;
}

void AccessPatternAnalyzer::processAccessBatch(const std::vector<AccessRecord>& records) {
    for (const auto& record : records) {
        processAccess(record);
    }
}

std::vector<LayerAccessPattern> AccessPatternAnalyzer::analyzeLayerPatterns() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!patterns_dirty_ && !cached_layer_patterns_.empty()) {
        return cached_layer_patterns_;
    }
    
    std::vector<LayerAccessPattern> patterns;
    
    for (const auto& [layer_id, count] : layer_access_counts_) {
        LayerAccessPattern pattern;
        pattern.layer_id = layer_id;
        pattern.total_accesses = count;
        pattern.access_ratio = total_accesses_ > 0 ? 
            static_cast<float>(count) / total_accesses_ : 0.0f;
        
        // 计算访问间隔统计
        computeIntervalStats(layer_id);
        
        // 分析模式类型
        std::vector<BlockId> layer_blocks;
        for (const auto& record : access_history_) {
            if (record.layer_id == layer_id) {
                layer_blocks.push_back(record.block_id);
            }
        }
        pattern.pattern_type = detectPatternType(layer_blocks);
        
        // 识别热头和冷头
        std::vector<std::pair<HeadId, uint64_t>> head_counts;
        for (const auto& [key, cnt] : head_access_counts_) {
            if (key.first == layer_id) {
                head_counts.push_back({key.second, cnt});
            }
        }
        
        std::sort(head_counts.begin(), head_counts.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        size_t top_n = std::min(size_t(3), head_counts.size());
        for (size_t i = 0; i < top_n; i++) {
            pattern.hot_heads.push_back(head_counts[i].first);
        }
        
        if (head_counts.size() > top_n) {
            for (size_t i = head_counts.size() - top_n; i < head_counts.size(); i++) {
                pattern.cold_heads.push_back(head_counts[i].first);
            }
        }
        
        patterns.push_back(pattern);
    }
    
    cached_layer_patterns_ = patterns;
    patterns_dirty_ = false;
    return patterns;
}

std::unordered_map<LayerId, std::vector<HeadId>> AccessPatternAnalyzer::analyzeHeadPatterns() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::unordered_map<LayerId, std::vector<HeadId>> result;
    
    for (const auto& [key, count] : head_access_counts_) {
        if (count > total_accesses_ / (num_layers_ * num_heads_ * 2)) {
            result[key.first].push_back(key.second);
        }
    }
    
    return result;
}

std::vector<TimeWindowStats> AccessPatternAnalyzer::analyzeTimePattern(
    Timestamp start_time, Timestamp end_time, uint64_t window_size_ms) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<TimeWindowStats> stats;
    
    uint64_t window_ns = window_size_ms * 1000000ULL;
    Timestamp current_start = start_time;
    
    while (current_start < end_time) {
        Timestamp current_end = std::min(current_start + window_ns, end_time);
        
        TimeWindowStats window_stat;
        window_stat.window_start = current_start;
        window_stat.window_end = current_end;
        
        std::unordered_map<BlockId, uint64_t> block_counts;
        
        for (const auto& record : access_history_) {
            if (record.timestamp >= current_start && record.timestamp < current_end) {
                window_stat.access_count++;
                block_counts[record.block_id]++;
            }
        }
        
        window_stat.unique_blocks = block_counts.size();
        
        if (window_stat.access_count > 0) {
            // 找出访问最多的块
            std::vector<std::pair<BlockId, uint64_t>> sorted_blocks(
                block_counts.begin(), block_counts.end());
            std::sort(sorted_blocks.begin(), sorted_blocks.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
            
            size_t top_n = std::min(size_t(5), sorted_blocks.size());
            for (size_t i = 0; i < top_n; i++) {
                window_stat.top_blocks.push_back(sorted_blocks[i].first);
            }
        }
        
        stats.push_back(window_stat);
        current_start = current_end;
    }
    
    return stats;
}

std::vector<AccessPrediction> AccessPatternAnalyzer::predictNextAccess(
    BlockId current_block, Timestamp current_time) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<AccessPrediction> predictions;
    
    // 使用马尔可夫链预测
    auto it = transition_counts_.find(current_block);
    if (it != transition_counts_.end()) {
        uint64_t total_transitions = 0;
        for (const auto& [_, count] : it->second) {
            total_transitions += count;
        }
        
        if (total_transitions > 0) {
            std::vector<std::pair<BlockId, float>> probs;
            for (const auto& [next_block, count] : it->second) {
                float prob = static_cast<float>(count) / total_transitions;
                probs.push_back({next_block, prob});
            }
            
            std::sort(probs.begin(), probs.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
            
            size_t n = std::min(prediction_window_size_, probs.size());
            for (size_t i = 0; i < n; i++) {
                AccessPrediction pred;
                pred.predicted_block = probs[i].first;
                pred.confidence = probs[i].second;
                pred.predicted_time = current_time + 1000000;  // 预测1ms后
                pred.recommended_prefetch_tier = 
                    probs[i].second > 0.5f ? TierType::HOT : TierType::WARM;
                predictions.push_back(pred);
            }
        }
    }
    
    // 如果预测结果不足，使用序列预测
    if (predictions.size() < prediction_window_size_) {
        // 查找相似的序列模式
        for (size_t i = 0; i < block_sequence_.size() - 1; i++) {
            if (block_sequence_[i] == current_block) {
                AccessPrediction pred;
                pred.predicted_block = block_sequence_[i + 1];
                pred.confidence = 0.3f;
                pred.predicted_time = current_time + 2000000;
                pred.recommended_prefetch_tier = TierType::WARM;
                predictions.push_back(pred);
                
                if (predictions.size() >= prediction_window_size_) {
                    break;
                }
            }
        }
    }
    
    return predictions;
}

std::vector<AccessPrediction> AccessPatternAnalyzer::getPrefetchSuggestions(size_t max_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<AccessPrediction> suggestions;
    
    if (block_sequence_.empty()) {
        return suggestions;
    }
    
    BlockId last_block = block_sequence_.back();
    Timestamp current_time = getCurrentTimestamp();
    
    auto predictions = predictNextAccess(last_block, current_time);
    
    size_t n = std::min(max_count, predictions.size());
    for (size_t i = 0; i < n; i++) {
        suggestions.push_back(predictions[i]);
    }
    
    return suggestions;
}

AccessPatternType AccessPatternAnalyzer::detectPatternType(
    const std::vector<BlockId>& block_sequence) {
    
    if (block_sequence.size() < 3) {
        return AccessPatternType::UNKNOWN;
    }
    
    // 检测顺序模式
    bool sequential = true;
    for (size_t i = 1; i < block_sequence.size(); i++) {
        if (block_sequence[i] != block_sequence[i-1] + 1) {
            sequential = false;
            break;
        }
    }
    if (sequential) {
        return AccessPatternType::SEQUENTIAL;
    }
    
    // 检测步长模式
    int64_t stride = static_cast<int64_t>(block_sequence[1]) - 
                     static_cast<int64_t>(block_sequence[0]);
    bool stride_pattern = true;
    for (size_t i = 2; i < block_sequence.size(); i++) {
        int64_t diff = static_cast<int64_t>(block_sequence[i]) - 
                       static_cast<int64_t>(block_sequence[i-1]);
        if (diff != stride) {
            stride_pattern = false;
            break;
        }
    }
    if (stride_pattern && stride != 0) {
        return AccessPatternType::STRIDE;
    }
    
    // 检测重复模式
    std::unordered_map<BlockId, int> counts;
    for (BlockId block : block_sequence) {
        counts[block]++;
    }
    
    size_t unique_blocks = counts.size();
    if (unique_blocks <= block_sequence.size() / 2) {
        return AccessPatternType::REPEATING;
    }
    
    // 检测聚类模式
    if (unique_blocks <= block_sequence.size() / 4) {
        return AccessPatternType::CLUSTERED;
    }
    
    return AccessPatternType::RANDOM;
}

PatternReport AccessPatternAnalyzer::generateReport() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    PatternReport report;
    report.total_accesses = total_accesses_;
    report.unique_blocks_accessed = block_access_times_.size();
    report.avg_access_per_block = report.unique_blocks_accessed > 0 ?
        static_cast<double>(total_accesses_) / report.unique_blocks_accessed : 0.0;
    
    // 计算熵
    std::vector<uint64_t> counts;
    for (const auto& [_, count] : layer_access_counts_) {
        counts.push_back(count);
    }
    report.entropy = computeEntropy(counts);
    
    // 生成层级访问矩阵
    report.layer_access_counts.resize(num_layers_);
    for (uint32_t l = 0; l < num_layers_; l++) {
        report.layer_access_counts[l].resize(num_heads_, 0);
        for (uint32_t h = 0; h < num_heads_; h++) {
            auto key = std::make_pair(l, h);
            auto it = head_access_counts_.find(key);
            if (it != head_access_counts_.end()) {
                report.layer_access_counts[l][h] = it->second;
            }
        }
    }
    
    // 识别热点块和冷块
    std::vector<std::pair<BlockId, uint64_t>> block_counts;
    for (const auto& [block, times] : block_access_times_) {
        block_counts.push_back({block, times.size()});
    }
    
    std::sort(block_counts.begin(), block_counts.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    size_t hot_n = std::min(size_t(10), block_counts.size());
    for (size_t i = 0; i < hot_n; i++) {
        report.hot_blocks.push_back(block_counts[i].first);
    }
    
    if (block_counts.size() > hot_n) {
        size_t cold_n = std::min(size_t(10), block_counts.size() - hot_n);
        for (size_t i = block_counts.size() - cold_n; i < block_counts.size(); i++) {
            report.cold_blocks.push_back(block_counts[i].first);
        }
    }
    
    // 生成预测
    if (!block_sequence_.empty()) {
        report.predicted_next_access = predictNextAccess(
            block_sequence_.back(), getCurrentTimestamp())
            .size() > 0 ? 
            std::vector<BlockId>{predictNextAccess(
                block_sequence_.back(), getCurrentTimestamp())[0].predicted_block} :
            std::vector<BlockId>{};
    }
    
    return report;
}

void AccessPatternAnalyzer::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    layer_access_counts_.clear();
    head_access_counts_.clear();
    access_history_.clear();
    block_sequence_.clear();
    transition_counts_.clear();
    layer_access_times_.clear();
    block_access_times_.clear();
    cached_layer_patterns_.clear();
    total_accesses_ = 0;
    patterns_dirty_ = true;
}

void AccessPatternAnalyzer::updateTransitionProb(BlockId from, BlockId to) {
    transition_counts_[from][to]++;
}

void AccessPatternAnalyzer::computeIntervalStats(LayerId layer_id) {
    auto it = layer_access_times_.find(layer_id);
    if (it == layer_access_times_.end() || it->second.size() < 2) {
        return;
    }
    
    const auto& times = it->second;
    std::vector<double> intervals;
    intervals.reserve(times.size() - 1);
    
    for (size_t i = 1; i < times.size(); i++) {
        double interval_ns = static_cast<double>(times[i] - times[i-1]);
        intervals.push_back(interval_ns / 1e6);  // 转换为毫秒
    }
    
    // 计算平均值和标准差
    // 注：实际使用时应该存储这些统计信息
}

double AccessPatternAnalyzer::computeEntropy(const std::vector<uint64_t>& counts) const {
    if (counts.empty()) {
        return 0.0;
    }
    
    uint64_t total = std::accumulate(counts.begin(), counts.end(), 0ULL);
    if (total == 0) {
        return 0.0;
    }
    
    double entropy = 0.0;
    for (uint64_t count : counts) {
        if (count > 0) {
            double p = static_cast<double>(count) / total;
            entropy -= p * std::log2(p);
        }
    }
    
    return entropy;
}

}  // namespace hetkvcache