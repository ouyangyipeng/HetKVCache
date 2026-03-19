/**
 * @file paged_allocator.cpp
 * @brief 分页式内存分配器实现
 */

#include "hetkvcache/allocator/paged_allocator.h"
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cstring>
#include <algorithm>

namespace hetkvcache {

PagedAllocator::PagedAllocator()
    : vram_pool_(nullptr)
    , ram_pool_(nullptr)
    , ssd_fd_(-1)
    , next_sequence_id_(1)
    , total_allocations_(0)
    , total_deallocations_(0)
    , peak_usage_(0)
    , next_block_id_(1)
    , initialized_(false)
{
    tier_capacities_.fill(0);
    for (size_t i = 0; i < NUM_TIERS; ++i) {
        tier_usages_[i] = 0;
    }
}

PagedAllocator::PagedAllocator(const PagedAllocatorConfig& config)
    : config_(config)
    , vram_pool_(nullptr)
    , ram_pool_(nullptr)
    , ssd_fd_(-1)
    , next_sequence_id_(1)
    , total_allocations_(0)
    , total_deallocations_(0)
    , peak_usage_(0)
    , next_block_id_(1)
    , initialized_(false)
{
    tier_capacities_.fill(0);
    for (int i = 0; i < NUM_TIERS; ++i) {
        tier_usages_[i] = 0;
    }
}

PagedAllocator::~PagedAllocator() {
    shutdown();
}

bool PagedAllocator::initialize() {
    if (initialized_.load()) {
        return true;
    }
    
    if (!initializeMemoryPools()) {
        return false;
    }
    
    if (!initializeSSDSwap()) {
        // SSD 初始化失败不是致命错误
    }
    
    // 计算每个层级的块数量
    size_t blocks_per_tier_vram = config_.vram_pool_size / config_.block_size;
    size_t blocks_per_tier_ram = config_.ram_pool_size / config_.block_size;
    
    tier_capacities_[static_cast<int>(TierType::HOT)] = blocks_per_tier_vram;
    tier_capacities_[static_cast<int>(TierType::WARM)] = blocks_per_tier_ram;
    tier_capacities_[static_cast<int>(TierType::COLD)] = 
        config_.ssd_swap_size / config_.block_size;
    
    // 初始化空闲链表
    for (size_t i = 0; i < blocks_per_tier_vram; i++) {
        BlockId block_id = allocateBlockId();
        free_lists_[static_cast<int>(TierType::HOT)].push(block_id);
        
        BlockMetadata metadata;
        metadata.block_id = block_id;
        metadata.tier = TierType::HOT;
        metadata.state = BlockState::FREE;
        metadata.data_ptr = static_cast<char*>(vram_pool_) + i * config_.block_size;
        block_metadata_[block_id] = metadata;
    }
    
    for (size_t i = 0; i < blocks_per_tier_ram; i++) {
        BlockId block_id = allocateBlockId();
        free_lists_[static_cast<int>(TierType::WARM)].push(block_id);
        
        BlockMetadata metadata;
        metadata.block_id = block_id;
        metadata.tier = TierType::WARM;
        metadata.state = BlockState::FREE;
        metadata.data_ptr = static_cast<char*>(ram_pool_) + i * config_.block_size;
        block_metadata_[block_id] = metadata;
    }
    
    initialized_ = true;
    return true;
}

void PagedAllocator::shutdown() {
    if (!initialized_.load()) {
        return;
    }
    
    // 释放内存池
    if (vram_pool_) {
        cudaFree(vram_pool_);
        vram_pool_ = nullptr;
    }
    
    if (ram_pool_) {
        if (config_.use_pinned_memory) {
            cudaFreeHost(ram_pool_);
        } else {
            free(ram_pool_);
        }
        ram_pool_ = nullptr;
    }
    
    if (ssd_fd_ >= 0) {
        close(ssd_fd_);
        ssd_fd_ = -1;
    }
    
    block_metadata_.clear();
    for (auto& free_list : free_lists_) {
        while (!free_list.empty()) {
            free_list.pop();
        }
    }
    
    initialized_ = false;
}

AllocationResult PagedAllocator::allocateBlock(LayerId layer_id,
                                               HeadId head_id,
                                               TierType preferred_tier) {
    AllocationResult result;
    
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    // 尝试从首选层级分配
    BlockId block_id = allocateFromTier(preferred_tier, layer_id, head_id);
    
    if (block_id == INVALID_BLOCK_ID && preferred_tier != TierType::HOT) {
        // 尝试从 HOT 层分配
        block_id = allocateFromTier(TierType::HOT, layer_id, head_id);
    }
    
    if (block_id == INVALID_BLOCK_ID) {
        // 尝试驱逐并分配
        block_id = tryEvictAndAllocate(preferred_tier, layer_id, head_id);
    }
    
    if (block_id == INVALID_BLOCK_ID) {
        result.success = false;
        result.error_message = "Out of memory";
        return result;
    }
    
    auto it = block_metadata_.find(block_id);
    if (it != block_metadata_.end()) {
        it->second.layer_id = layer_id;
        it->second.head_id = head_id;
        it->second.state = BlockState::ALLOCATED;
        it->second.ref_count = 1;
        it->second.last_access = getCurrentTimestamp();
        
        result.block_id = block_id;
        result.ptr = it->second.data_ptr;
        result.tier = it->second.tier;
        result.success = true;
        
        tier_blocks_[it->second.tier].push_back(block_id);
        layer_blocks_[layer_id].push_back(block_id);
        
        tier_usages_[static_cast<int>(it->second.tier)]++;
        
        total_allocations_++;
        size_t current_usage = getTierUsage(TierType::HOT) + 
                               getTierUsage(TierType::WARM) +
                               getTierUsage(TierType::COLD);
        peak_usage_ = std::max(peak_usage_.load(), current_usage);
    }
    
    return result;
}

std::vector<BlockId> PagedAllocator::allocateSequence(LayerId layer_id,
                                                      HeadId head_id,
                                                      size_t num_blocks) {
    std::vector<BlockId> blocks;
    blocks.reserve(num_blocks);
    
    BlockId prev_block = INVALID_BLOCK_ID;
    
    for (size_t i = 0; i < num_blocks; i++) {
        auto result = allocateBlock(layer_id, head_id);
        if (!result.success) {
            // 回滚已分配的块
            for (BlockId block_id : blocks) {
                deallocateBlock(block_id);
            }
            return {};
        }
        
        blocks.push_back(result.block_id);
        
        // 链接块
        if (prev_block != INVALID_BLOCK_ID) {
            auto it = block_metadata_.find(prev_block);
            if (it != block_metadata_.end()) {
                it->second.next_block = result.block_id;
            }
        }
        prev_block = result.block_id;
    }
    
    return blocks;
}

std::vector<BlockId> PagedAllocator::extendSequence(uint64_t sequence_id,
                                                    size_t num_additional_blocks) {
    std::vector<BlockId> new_blocks;
    
    auto seq_it = sequences_.find(sequence_id);
    if (seq_it == sequences_.end()) {
        return new_blocks;
    }
    
    auto& seq_info = seq_it->second;
    new_blocks = allocateSequence(seq_info.layer_id, seq_info.head_id, num_additional_blocks);
    
    if (!new_blocks.empty()) {
        // 链接到序列末尾
        if (seq_info.tail_block != INVALID_BLOCK_ID) {
            auto it = block_metadata_.find(seq_info.tail_block);
            if (it != block_metadata_.end()) {
                it->second.next_block = new_blocks[0];
            }
        }
        
        seq_info.tail_block = new_blocks.back();
        seq_info.num_blocks += new_blocks.size();
    }
    
    return new_blocks;
}

void PagedAllocator::deallocateBlock(BlockId block_id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    auto it = block_metadata_.find(block_id);
    if (it == block_metadata_.end() || it->second.state != BlockState::ALLOCATED) {
        return;
    }
    
    // 从层级列表中移除
    auto& tier_list = tier_blocks_[it->second.tier];
    tier_list.erase(std::remove(tier_list.begin(), tier_list.end(), block_id), 
                    tier_list.end());
    
    auto& layer_list = layer_blocks_[it->second.layer_id];
    layer_list.erase(std::remove(layer_list.begin(), layer_list.end(), block_id),
                     layer_list.end());
    
    // 重置元数据
    it->second.state = BlockState::FREE;
    it->second.ref_count = 0;
    it->second.layer_id = 0;
    it->second.head_id = 0;
    it->second.next_block = INVALID_BLOCK_ID;
    
    // 添加回空闲链表
    free_lists_[static_cast<int>(it->second.tier)].push(block_id);
    
    tier_usages_[static_cast<int>(it->second.tier)]--;
    
    total_deallocations_++;
}

void PagedAllocator::deallocateSequence(const std::vector<BlockId>& block_ids) {
    for (BlockId block_id : block_ids) {
        deallocateBlock(block_id);
    }
}

void* PagedAllocator::getBlockPtr(BlockId block_id) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = block_metadata_.find(block_id);
    if (it != block_metadata_.end()) {
        return it->second.data_ptr;
    }
    return nullptr;
}

const void* PagedAllocator::getBlockPtr(BlockId block_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = block_metadata_.find(block_id);
    if (it != block_metadata_.end()) {
        return it->second.data_ptr;
    }
    return nullptr;
}

BlockMetadata PagedAllocator::getBlockMetadata(BlockId block_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = block_metadata_.find(block_id);
    if (it != block_metadata_.end()) {
        return it->second;
    }
    return BlockMetadata();
}

void PagedAllocator::updateBlockMetadata(BlockId block_id, const BlockMetadata& metadata) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    block_metadata_[block_id] = metadata;
}

size_t PagedAllocator::getFreeBlocks(TierType tier) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return free_lists_[static_cast<int>(tier)].size();
}

size_t PagedAllocator::getUsedBlocks(TierType tier) const {
    return tier_usages_[static_cast<int>(tier)];
}

double PagedAllocator::getFragmentationRatio() const {
    return computeFragmentation();
}

void PagedAllocator::defragment() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    // 简单的碎片整理：重新组织空闲链表
    // 实际实现应该进行实际的数据移动
    
    for (int t = 0; t < static_cast<int>(NUM_TIERS); t++) {
        std::queue<BlockId> new_free_list;
        auto& old_list = free_lists_[t];
        
        while (!old_list.empty()) {
            BlockId block_id = old_list.front();
            old_list.pop();
            
            auto it = block_metadata_.find(block_id);
            if (it != block_metadata_.end() && it->second.state == BlockState::FREE) {
                new_free_list.push(block_id);
            }
        }
        
        free_lists_[t] = std::move(new_free_list);
    }
}

bool PagedAllocator::migrateBlock(BlockId block_id, TierType target_tier) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    auto it = block_metadata_.find(block_id);
    if (it == block_metadata_.end()) {
        return false;
    }
    
    TierType current_tier = it->second.tier;
    if (current_tier == target_tier) {
        return true;
    }
    
    // 检查目标层级是否有空间
    if (free_lists_[static_cast<int>(target_tier)].empty()) {
        return false;
    }
    
    // 获取目标块
    BlockId target_block_id = free_lists_[static_cast<int>(target_tier)].front();
    free_lists_[static_cast<int>(target_tier)].pop();
    
    auto target_it = block_metadata_.find(target_block_id);
    if (target_it == block_metadata_.end()) {
        return false;
    }
    
    // 复制数据
    void* src_ptr = it->second.data_ptr;
    void* dst_ptr = target_it->second.data_ptr;
    
    if (current_tier == TierType::HOT && target_tier == TierType::WARM) {
        cudaMemcpy(dst_ptr, src_ptr, config_.block_size, cudaMemcpyDeviceToHost);
    } else if (current_tier == TierType::WARM && target_tier == TierType::HOT) {
        cudaMemcpy(dst_ptr, src_ptr, config_.block_size, cudaMemcpyHostToDevice);
    } else if (current_tier == TierType::WARM && target_tier == TierType::COLD) {
        // 写入 SSD
        if (ssd_fd_ >= 0) {
            uint64_t offset = target_block_id * config_.block_size;
            pwrite(ssd_fd_, src_ptr, config_.block_size, offset);
        }
    } else if (current_tier == TierType::COLD && target_tier == TierType::WARM) {
        if (ssd_fd_ >= 0) {
            uint64_t offset = it->second.sequence_start;  // 使用实际偏移
            pread(ssd_fd_, dst_ptr, config_.block_size, offset);
        }
    }
    
    // 更新元数据
    target_it->second.layer_id = it->second.layer_id;
    target_it->second.head_id = it->second.head_id;
    target_it->second.state = it->second.state;
    target_it->second.ref_count = it->second.ref_count;
    
    it->second.state = BlockState::FREE;
    it->second.tier = current_tier;
    free_lists_[static_cast<int>(current_tier)].push(block_id);
    
    // 更新使用量
    tier_usages_[static_cast<int>(current_tier)]--;
    tier_usages_[static_cast<int>(target_tier)]++;
    
    return true;
}

void PagedAllocator::lockBlock(BlockId block_id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = block_metadata_.find(block_id);
    if (it != block_metadata_.end()) {
        it->second.state = BlockState::LOCKED;
    }
}

void PagedAllocator::unlockBlock(BlockId block_id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = block_metadata_.find(block_id);
    if (it != block_metadata_.end() && it->second.state == BlockState::LOCKED) {
        it->second.state = BlockState::ALLOCATED;
    }
}

size_t PagedAllocator::getTierUsage(TierType tier) const {
    return tier_usages_[static_cast<int>(tier)];
}

size_t PagedAllocator::getTierCapacity(TierType tier) const {
    return tier_capacities_[static_cast<int>(tier)];
}

size_t PagedAllocator::getTotalBlocks() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return block_metadata_.size();
}

std::vector<BlockId> PagedAllocator::getAllBlockIds() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::vector<BlockId> ids;
    ids.reserve(block_metadata_.size());
    for (const auto& [id, _] : block_metadata_) {
        ids.push_back(id);
    }
    return ids;
}

std::vector<BlockId> PagedAllocator::getBlocksByLayer(LayerId layer_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = layer_blocks_.find(layer_id);
    return (it != layer_blocks_.end()) ? it->second : std::vector<BlockId>();
}

std::vector<BlockId> PagedAllocator::getBlocksByTier(TierType tier) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = tier_blocks_.find(tier);
    return (it != tier_blocks_.end()) ? it->second : std::vector<BlockId>();
}

PagedAllocator::AllocatorStats PagedAllocator::getStats() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    AllocatorStats stats;
    stats.total_blocks = block_metadata_.size();
    stats.allocated_blocks = total_allocations_ - total_deallocations_;
    stats.locked_blocks = 0;
    stats.vram_blocks = getUsedBlocks(TierType::HOT);
    stats.ram_blocks = getUsedBlocks(TierType::WARM);
    stats.ssd_blocks = getUsedBlocks(TierType::COLD);
    stats.total_allocations = total_allocations_;
    stats.total_deallocations = total_deallocations_;
    stats.fragmentation_ratio = computeFragmentation();
    stats.peak_usage = peak_usage_;
    
    for (const auto& [_, metadata] : block_metadata_) {
        if (metadata.state == BlockState::LOCKED) {
            stats.locked_blocks++;
        }
    }
    
    stats.free_blocks = stats.total_blocks - stats.allocated_blocks;
    
    return stats;
}

void PagedAllocator::resetStats() {
    total_allocations_ = 0;
    total_deallocations_ = 0;
    peak_usage_ = 0;
}

BlockId PagedAllocator::allocateFromTier(TierType tier, LayerId layer_id, HeadId head_id) {
    if (free_lists_[static_cast<int>(tier)].empty()) {
        return INVALID_BLOCK_ID;
    }
    
    BlockId block_id = free_lists_[static_cast<int>(tier)].front();
    free_lists_[static_cast<int>(tier)].pop();
    
    auto it = block_metadata_.find(block_id);
    if (it != block_metadata_.end()) {
        it->second.state = BlockState::ALLOCATED;
        it->second.layer_id = layer_id;
        it->second.head_id = head_id;
        it->second.ref_count = 1;
        it->second.last_access = getCurrentTimestamp();
    }
    
    return block_id;
}

BlockId PagedAllocator::tryEvictAndAllocate(TierType preferred_tier, 
                                            LayerId layer_id, 
                                            HeadId head_id) {
    // 找出可以驱逐的块
    auto candidates = selectEvictionCandidates(preferred_tier, 1);
    
    if (candidates.empty()) {
        return INVALID_BLOCK_ID;
    }
    
    // 驱逐第一个候选
    BlockId victim_id = candidates[0];
    auto victim_it = block_metadata_.find(victim_id);
    
    if (victim_it != block_metadata_.end()) {
        // 降级到下一层级
        TierType target_tier = static_cast<TierType>(
            std::min(static_cast<int>(victim_it->second.tier) + 1, 
                    static_cast<int>(NUM_TIERS) - 1));
        
        if (migrateBlock(victim_id, target_tier)) {
            return allocateFromTier(preferred_tier, layer_id, head_id);
        }
    }
    
    return INVALID_BLOCK_ID;
}

std::vector<BlockId> PagedAllocator::selectEvictionCandidates(TierType tier, 
                                                               size_t num_blocks) {
    std::vector<BlockId> candidates;
    
    auto it = tier_blocks_.find(tier);
    if (it == tier_blocks_.end()) {
        return candidates;
    }
    
    // 按热度排序（最冷的先驱逐）
    std::vector<std::pair<BlockId, float>> block_scores;
    for (BlockId block_id : it->second) {
        auto meta_it = block_metadata_.find(block_id);
        if (meta_it != block_metadata_.end() && 
            meta_it->second.state == BlockState::ALLOCATED) {
            block_scores.push_back({block_id, meta_it->second.heat_score});
        }
    }
    
    std::sort(block_scores.begin(), block_scores.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    size_t n = std::min(num_blocks, block_scores.size());
    for (size_t i = 0; i < n; i++) {
        candidates.push_back(block_scores[i].first);
    }
    
    return candidates;
}

bool PagedAllocator::initializeMemoryPools() {
    // 分配 VRAM
    cudaError_t err = cudaMalloc(&vram_pool_, config_.vram_pool_size);
    if (err != cudaSuccess) {
        vram_pool_ = nullptr;
        return false;
    }
    
    // 分配 RAM（锁页内存以加速传输）
    if (config_.use_pinned_memory) {
        err = cudaMallocHost(&ram_pool_, config_.ram_pool_size);
    } else {
        ram_pool_ = aligned_alloc(4096, config_.ram_pool_size);
    }
    
    if (err != cudaSuccess || ram_pool_ == nullptr) {
        cudaFree(vram_pool_);
        vram_pool_ = nullptr;
        return false;
    }
    
    return true;
}

bool PagedAllocator::initializeSSDSwap() {
    // 创建或打开交换文件
    ssd_fd_ = open(config_.ssd_swap_path.c_str(), 
                   O_RDWR | O_CREAT | O_DIRECT,
                   S_IRUSR | S_IWUSR);
    
    if (ssd_fd_ < 0) {
        return false;
    }
    
    // 预分配空间
    if (ftruncate(ssd_fd_, config_.ssd_swap_size) < 0) {
        close(ssd_fd_);
        ssd_fd_ = -1;
        return false;
    }
    
    return true;
}

BlockId PagedAllocator::allocateBlockId() {
    return next_block_id_.fetch_add(1);
}

double PagedAllocator::computeFragmentation() const {
    size_t total_free = 0;
    size_t total_capacity = 0;
    
    for (int t = 0; t < static_cast<int>(NUM_TIERS); t++) {
        total_free += free_lists_[t].size();
        total_capacity += tier_capacities_[t];
    }
    
    if (total_capacity == 0) {
        return 0.0;
    }
    
    double utilization = 1.0 - static_cast<double>(total_free) / total_capacity;
    
    // 计算外部碎片（空闲块的分散程度）
    size_t max_contiguous_free = 0;
    for (int t = 0; t < static_cast<int>(NUM_TIERS); t++) {
        max_contiguous_free = std::max(max_contiguous_free, free_lists_[t].size());
    }
    
    double fragmentation = 1.0;
    if (total_free > 0) {
        fragmentation = 1.0 - static_cast<double>(max_contiguous_free) / total_free;
    }
    
    return fragmentation * utilization;
}

}  // namespace hetkvcache