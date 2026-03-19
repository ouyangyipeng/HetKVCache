/**
 * @file transfer_scheduler.cpp
 * @brief 传输调度器实现
 */

#include "hetkvcache/migration/transfer_scheduler.h"
#include <algorithm>
#include <numeric>

namespace hetkvcache {

TransferScheduler::TransferScheduler()
    : next_request_id_(1)
    , total_requests_(0)
    , total_batches_(0)
    , total_bytes_(0)
{
}

TransferScheduler::TransferScheduler(const SchedulerConfig& config)
    : config_(config)
    , next_request_id_(1)
    , total_requests_(0)
    , total_batches_(0)
    , total_bytes_(0)
{
}

uint64_t TransferScheduler::submitRequest(const TransferRequest& request) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    uint64_t request_id = next_request_id_.fetch_add(1);
    
    InternalRequest internal_req;
    internal_req.request = request;
    internal_req.request.request_id = request_id;
    internal_req.submit_time = getCurrentTimestamp();
    internal_req.queue_position = static_cast<int>(requests_.size());
    
    requests_[request_id] = internal_req;
    
    int priority_score = computePriorityScore(internal_req);
    priority_queue_.push({priority_score, request_id});
    
    total_requests_++;
    total_bytes_ += request.size;
    
    return request_id;
}

std::vector<uint64_t> TransferScheduler::submitRequestBatch(
    const std::vector<TransferRequest>& requests) {
    
    std::vector<uint64_t> request_ids;
    request_ids.reserve(requests.size());
    
    for (const auto& request : requests) {
        request_ids.push_back(submitRequest(request));
    }
    
    return request_ids;
}

BatchTransferTask TransferScheduler::scheduleNext() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (priority_queue_.empty()) {
        return BatchTransferTask();
    }
    
    // 收集要处理的请求
    std::vector<InternalRequest> batch_requests;
    size_t total_size = 0;
    
    while (!priority_queue_.empty() && 
           batch_requests.size() < config_.max_batch_size) {
        
        auto top = priority_queue_.top();
        priority_queue_.pop();
        
        auto it = requests_.find(top.second);
        if (it != requests_.end()) {
            batch_requests.push_back(it->second);
            total_size += it->second.request.size;
        }
    }
    
    if (batch_requests.empty()) {
        return BatchTransferTask();
    }
    
    // 创建批量任务
    BatchTransferTask batch;
    batch.total_size = total_size;
    batch.scheduled_time = getCurrentTimestamp();
    batch.stream = allocateStream();
    
    for (const auto& internal_req : batch_requests) {
        batch.requests.push_back(internal_req.request);
        requests_.erase(internal_req.request.request_id);
    }
    
    total_batches_++;
    
    return batch;
}

const BatchTransferTask* TransferScheduler::peekNext() const {
    // 简化实现，实际应该缓存下一个批次
    return nullptr;
}

bool TransferScheduler::hasPendingTasks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !priority_queue_.empty();
}

size_t TransferScheduler::getPendingCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return requests_.size();
}

bool TransferScheduler::cancelRequest(uint64_t request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = requests_.find(request_id);
    if (it != requests_.end()) {
        requests_.erase(it);
        return true;
    }
    return false;
}

double TransferScheduler::getEstimatedBandwidth(TierType from, TierType to) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    BandwidthKey key{from, to};
    auto it = bandwidth_stats_.find(key);
    if (it != bandwidth_stats_.end()) {
        return it->second;
    }
    
    // 返回默认带宽估计
    if (from == TierType::HOT && to == TierType::WARM) {
        return 12000.0;  // 12 GB/s PCIe
    } else if (from == TierType::WARM && to == TierType::HOT) {
        return 12000.0;
    } else if (from == TierType::WARM && to == TierType::COLD) {
        return 3000.0;  // 3 GB/s NVMe
    } else if (from == TierType::COLD && to == TierType::WARM) {
        return 3000.0;
    }
    
    return 1000.0;  // 默认 1 GB/s
}

void TransferScheduler::updateBandwidth(TierType from, TierType to, double bandwidth_mbps) {
    std::lock_guard<std::mutex> lock(mutex_);
    BandwidthKey key{from, to};
    bandwidth_stats_[key] = bandwidth_mbps;
}

void TransferScheduler::reorderTasks() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 重新构建优先级队列
    while (!priority_queue_.empty()) {
        priority_queue_.pop();
    }
    
    for (const auto& [id, internal_req] : requests_) {
        int priority_score = computePriorityScore(internal_req);
        priority_queue_.push({priority_score, id});
    }
}

void TransferScheduler::setStreamPool(std::vector<cudaStream_t>& streams) {
    std::lock_guard<std::mutex> lock(mutex_);
    stream_pool_ = streams;
    while (!available_streams_.empty()) {
        available_streams_.pop();
    }
    for (auto stream : streams) {
        available_streams_.push(stream);
    }
}

cudaStream_t TransferScheduler::allocateStream() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (available_streams_.empty()) {
        return nullptr;
    }
    cudaStream_t stream = available_streams_.front();
    available_streams_.pop();
    return stream;
}

void TransferScheduler::releaseStream(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    available_streams_.push(stream);
}

TransferScheduler::SchedulerStats TransferScheduler::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    SchedulerStats stats;
    stats.total_requests = total_requests_;
    stats.total_batches = total_batches_;
    stats.total_bytes = total_bytes_;
    stats.avg_batch_size = total_batches_ > 0 ? 
        static_cast<double>(total_requests_) / total_batches_ : 0.0;
    stats.avg_wait_time_us = 0.0;  // TODO: 实现
    stats.bandwidth_stats = bandwidth_stats_;
    
    return stats;
}

void TransferScheduler::resetStats() {
    total_requests_ = 0;
    total_batches_ = 0;
    total_bytes_ = 0;
}

void TransferScheduler::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!priority_queue_.empty()) {
        priority_queue_.pop();
    }
    requests_.clear();
    batch_queue_ = std::queue<BatchTransferTask>();
}

BatchTransferTask TransferScheduler::coalesceRequests(
    std::vector<InternalRequest>& requests) {
    
    BatchTransferTask batch;
    batch.scheduled_time = getCurrentTimestamp();
    
    // 按层级分组
    std::unordered_map<TierType, std::vector<InternalRequest>> by_tier;
    for (const auto& req : requests) {
        by_tier[req.request.target_tier].push_back(req);
    }
    
    // 合并小请求
    for (auto& [tier, tier_requests] : by_tier) {
        size_t total_size = 0;
        for (const auto& req : tier_requests) {
            if (total_size + req.request.size <= config_.max_batch_size * 1024 * 1024) {
                batch.requests.push_back(req.request);
                total_size += req.request.size;
            }
        }
        batch.total_size += total_size;
    }
    
    return batch;
}

int TransferScheduler::computePriorityScore(const InternalRequest& req) const {
    int score = static_cast<int>(req.request.priority) * 1000;
    
    // 添加等待时间因素
    Timestamp wait_time = getCurrentTimestamp() - req.submit_time;
    score += static_cast<int>(wait_time / 1000000);  // 毫秒
    
    // 添加带宽因素
    double bandwidth = getEstimatedBandwidth(req.request.source_tier, 
                                             req.request.target_tier);
    if (bandwidth > 0) {
        double transfer_time = static_cast<double>(req.request.size) / (bandwidth * 1e6);
        score -= static_cast<int>(transfer_time);
    }
    
    return score;
}

double TransferScheduler::estimateTransferTime(const TransferRequest& req) const {
    double bandwidth = getEstimatedBandwidth(req.source_tier, req.target_tier);
    if (bandwidth <= 0) {
        return 0.0;
    }
    return static_cast<double>(req.size) / (bandwidth * 1e6) * 1000.0;  // 毫秒
}

}  // namespace hetkvcache