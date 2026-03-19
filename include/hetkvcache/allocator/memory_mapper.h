/**
 * @file memory_mapper.h
 * @brief 内存映射器
 * 
 * 管理块在不同存储层级中的地址映射和位置跟踪。
 */

#ifndef HETKVCACHE_MEMORY_MAPPER_H
#define HETKVCACHE_MEMORY_MAPPER_H

#include "hetkvcache/types.h"
#include <mutex>
#include <unordered_map>
#include <vector>
#include <array>

namespace hetkvcache {

/**
 * @brief 块位置信息
 */
struct BlockLocation {
    TierType tier;
    void* ptr;              ///< 内存指针 (VRAM/RAM)
    uint64_t file_offset;   ///< SSD 文件偏移
    size_t size;
    bool valid;
    bool compressed;        ///< 是否压缩存储
    size_t compressed_size; ///< 压缩后大小
    
    BlockLocation()
        : tier(TierType::HOT)
        , ptr(nullptr)
        , file_offset(0)
        , size(0)
        , valid(false)
        , compressed(false)
        , compressed_size(0)
    {}
};

/**
 * @brief 地址映射条目
 */
struct AddressMapping {
    BlockId block_id;
    std::array<BlockLocation, NUM_TIERS> locations;  ///< 每个层级的位置
    TierType primary_tier;    ///< 主存储层级
    bool is_replicated;       ///< 是否有副本
    
    AddressMapping()
        : block_id(INVALID_BLOCK_ID)
        , primary_tier(TierType::HOT)
        , is_replicated(false)
    {}
};

/**
 * @brief 内存映射器
 */
class MemoryMapper {
public:
    /**
     * @brief 构造函数
     */
    MemoryMapper();
    
    /**
     * @brief 析构函数
     */
    ~MemoryMapper() = default;
    
    /**
     * @brief 初始化
     */
    void initialize(void* vram_base, size_t vram_size,
                   void* ram_base, size_t ram_size,
                   int ssd_fd);
    
    /**
     * @brief 注册块
     */
    void registerBlock(BlockId block_id, TierType tier, 
                      void* ptr, size_t size);
    
    /**
     * @brief 注销块
     */
    void unregisterBlock(BlockId block_id);
    
    /**
     * @brief 更新块位置
     */
    void updateLocation(BlockId block_id, TierType tier,
                       void* ptr, size_t size);
    
    /**
     * @brief 更新 SSD 位置
     */
    void updateSSDLocation(BlockId block_id, uint64_t offset, size_t size);
    
    /**
     * @brief 获取块的内存指针
     */
    void* getPointer(BlockId block_id, TierType tier) const;
    
    /**
     * @brief 获取块的当前位置
     */
    BlockLocation getLocation(BlockId block_id, TierType tier) const;
    
    /**
     * @brief 获取块的主位置
     */
    BlockLocation getPrimaryLocation(BlockId block_id) const;
    
    /**
     * @brief 获取块的主层级
     */
    TierType getPrimaryTier(BlockId block_id) const;
    
    /**
     * @brief 设置块的主层级
     */
    void setPrimaryTier(BlockId block_id, TierType tier);
    
    /**
     * @brief 检查块是否在指定层级
     */
    bool hasLocation(BlockId block_id, TierType tier) const;
    
    /**
     * @brief 获取所有层级的位置
     */
    std::vector<TierType> getAvailableTiers(BlockId block_id) const;
    
    /**
     * @brief 分配 SSD 空间
     */
    uint64_t allocateSSDSpace(size_t size);
    
    /**
     * @brief 释放 SSD 空间
     */
    void freeSSDSpace(uint64_t offset, size_t size);
    
    /**
     * @brief 设置压缩信息
     */
    void setCompressionInfo(BlockId block_id, 
                           size_t original_size, 
                           size_t compressed_size);
    
    /**
     * @brief 获取映射数量
     */
    size_t getMappingCount() const;
    
    /**
     * @brief 获取各层级的块数量
     */
    std::array<size_t, NUM_TIERS> getTierBlockCounts() const;
    
    /**
     * @brief 清空所有映射
     */
    void clear();

private:
    mutable std::mutex mutex_;
    
    // 地址映射表
    std::unordered_map<BlockId, AddressMapping> mappings_;
    
    // 存储区域基址
    void* vram_base_;
    void* ram_base_;
    int ssd_fd_;
    
    // SSD 空间管理
    uint64_t ssd_next_offset_;
    std::vector<std::pair<uint64_t, size_t>> ssd_free_blocks_;
    
    // 统计
    std::array<size_t, NUM_TIERS> tier_counts_;
};

}  // namespace hetkvcache

#endif  // HETKVCACHE_MEMORY_MAPPER_H