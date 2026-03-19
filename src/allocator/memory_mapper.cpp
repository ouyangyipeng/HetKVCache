/**
 * @file memory_mapper.cpp
 * @brief 内存映射器实现
 */

#include "hetkvcache/allocator/memory_mapper.h"
#include <algorithm>

namespace hetkvcache {

MemoryMapper::MemoryMapper()
    : vram_base_(nullptr)
    , ram_base_(nullptr)
    , ssd_fd_(-1)
    , ssd_next_offset_(0)
{
    tier_counts_.fill(0);
}

void MemoryMapper::initialize(void* vram_base, size_t vram_size,
                              void* ram_base, size_t ram_size,
                              int ssd_fd) {
    std::lock_guard<std::mutex> lock(mutex_);
    vram_base_ = vram_base;
    ram_base_ = ram_base;
    ssd_fd_ = ssd_fd;
    mappings_.clear();
    ssd_next_offset_ = 0;
    ssd_free_blocks_.clear();
    tier_counts_.fill(0);
}

void MemoryMapper::registerBlock(BlockId block_id, TierType tier,
                                 void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    AddressMapping& mapping = mappings_[block_id];
    mapping.block_id = block_id;
    mapping.primary_tier = tier;
    
    BlockLocation& loc = mapping.locations[static_cast<int>(tier)];
    loc.tier = tier;
    loc.ptr = ptr;
    loc.size = size;
    loc.valid = true;
    
    tier_counts_[static_cast<int>(tier)]++;
}

void MemoryMapper::unregisterBlock(BlockId block_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = mappings_.find(block_id);
    if (it != mappings_.end()) {
        // 更新层级计数
        for (const auto& loc : it->second.locations) {
            if (loc.valid) {
                tier_counts_[static_cast<int>(loc.tier)]--;
            }
        }
        mappings_.erase(it);
    }
}

void MemoryMapper::updateLocation(BlockId block_id, TierType tier,
                                  void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = mappings_.find(block_id);
    if (it != mappings_.end()) {
        BlockLocation& loc = it->second.locations[static_cast<int>(tier)];
        bool was_valid = loc.valid;
        loc.tier = tier;
        loc.ptr = ptr;
        loc.size = size;
        loc.valid = true;
        
        if (!was_valid) {
            tier_counts_[static_cast<int>(tier)]++;
        }
    }
}

void MemoryMapper::updateSSDLocation(BlockId block_id, uint64_t offset, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = mappings_.find(block_id);
    if (it != mappings_.end()) {
        BlockLocation& loc = it->second.locations[static_cast<int>(TierType::COLD)];
        bool was_valid = loc.valid;
        loc.tier = TierType::COLD;
        loc.file_offset = offset;
        loc.size = size;
        loc.valid = true;
        
        if (!was_valid) {
            tier_counts_[static_cast<int>(TierType::COLD)]++;
        }
    }
}

void* MemoryMapper::getPointer(BlockId block_id, TierType tier) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = mappings_.find(block_id);
    if (it != mappings_.end()) {
        const BlockLocation& loc = it->second.locations[static_cast<int>(tier)];
        if (loc.valid) {
            return loc.ptr;
        }
    }
    return nullptr;
}

BlockLocation MemoryMapper::getLocation(BlockId block_id, TierType tier) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = mappings_.find(block_id);
    if (it != mappings_.end()) {
        return it->second.locations[static_cast<int>(tier)];
    }
    return BlockLocation();
}

BlockLocation MemoryMapper::getPrimaryLocation(BlockId block_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = mappings_.find(block_id);
    if (it != mappings_.end()) {
        return it->second.locations[static_cast<int>(it->second.primary_tier)];
    }
    return BlockLocation();
}

TierType MemoryMapper::getPrimaryTier(BlockId block_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = mappings_.find(block_id);
    if (it != mappings_.end()) {
        return it->second.primary_tier;
    }
    return TierType::HOT;
}

void MemoryMapper::setPrimaryTier(BlockId block_id, TierType tier) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = mappings_.find(block_id);
    if (it != mappings_.end()) {
        it->second.primary_tier = tier;
    }
}

bool MemoryMapper::hasLocation(BlockId block_id, TierType tier) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = mappings_.find(block_id);
    if (it != mappings_.end()) {
        return it->second.locations[static_cast<int>(tier)].valid;
    }
    return false;
}

std::vector<TierType> MemoryMapper::getAvailableTiers(BlockId block_id) const {
    std::vector<TierType> tiers;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = mappings_.find(block_id);
    if (it != mappings_.end()) {
        for (int t = 0; t < static_cast<int>(NUM_TIERS); t++) {
            if (it->second.locations[t].valid) {
                tiers.push_back(static_cast<TierType>(t));
            }
        }
    }
    
    return tiers;
}

uint64_t MemoryMapper::allocateSSDSpace(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 尝试复用已释放的空间
    for (auto it = ssd_free_blocks_.begin(); it != ssd_free_blocks_.end(); ++it) {
        if (it->second >= size) {
            uint64_t offset = it->first;
            if (it->second > size) {
                // 分割剩余空间
                ssd_free_blocks_.push_back({offset + size, it->second - size});
            }
            ssd_free_blocks_.erase(it);
            return offset;
        }
    }
    
    // 分配新空间
    uint64_t offset = ssd_next_offset_;
    ssd_next_offset_ += size;
    return offset;
}

void MemoryMapper::freeSSDSpace(uint64_t offset, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    ssd_free_blocks_.push_back({offset, size});
}

void MemoryMapper::setCompressionInfo(BlockId block_id,
                                      size_t original_size,
                                      size_t compressed_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = mappings_.find(block_id);
    if (it != mappings_.end()) {
        for (auto& loc : it->second.locations) {
            if (loc.valid && loc.compressed) {
                loc.size = original_size;
                loc.compressed_size = compressed_size;
            }
        }
    }
}

size_t MemoryMapper::getMappingCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return mappings_.size();
}

std::array<size_t, NUM_TIERS> MemoryMapper::getTierBlockCounts() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tier_counts_;
}

void MemoryMapper::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    mappings_.clear();
    ssd_free_blocks_.clear();
    ssd_next_offset_ = 0;
    tier_counts_.fill(0);
}

}  // namespace hetkvcache