/**
 * @file paged_allocator.h
 * @brief 分页式内存分配器
 * 
 * 参考 PagedAttention 的思想，实现固定块大小的内存池管理。
 * 解决动态序列增长带来的显存碎片化问题。
 */

#ifndef HETKVCACHE_PAGED_ALLOCATOR_H
#define HETKVCACHE_PAGED_ALLOCATOR_H

#include "hetkvcache/types.h"
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>
#include <queue>
#include <atomic>
#include <memory>

namespace hetkvcache {

/**
 * @brief 内存块元数据
 */
struct BlockMetadata {
    BlockId block_id;
    LayerId layer_id;
    HeadId head_id;
    SequencePos sequence_start;
    SequencePos sequence_end;
    TierType tier;
    BlockState state;
    uint32_t ref_count;
    void* data_ptr;
    Timestamp last_access;
    uint32_t access_count;
    float heat_score;
    BlockId next_block;  ///< 序列中下一个块
    
    BlockMetadata()
        : block_id(INVALID_BLOCK_ID)
        , layer_id(0)
        , head_id(0)
        , sequence_start(0)
        , sequence_end(0)
        , tier(TierType::HOT)
        , state(BlockState::FREE)
        , ref_count(0)
        , data_ptr(nullptr)
        , last_access(0)
        , access_count(0)
        , heat_score(0.0f)
        , next_block(INVALID_BLOCK_ID)
    {}
};

/**
 * @brief 序列块列表
 */
struct SequenceBlockList {
    BlockId head_block;
    BlockId tail_block;
    size_t num_blocks;
    LayerId layer_id;
    HeadId head_id;
    uint64_t sequence_id;
    
    SequenceBlockList()
        : head_block(INVALID_BLOCK_ID)
        , tail_block(INVALID_BLOCK_ID)
        , num_blocks(0)
        , layer_id(0)
        , head_id(0)
        , sequence_id(0)
    {}
};

/**
 * @brief 分配结果
 */
struct AllocationResult {
    BlockId block_id;
    void* ptr;
    TierType tier;
    bool success;
    std::string error_message;
    
    AllocationResult()
        : block_id(INVALID_BLOCK_ID)
        , ptr(nullptr)
        , tier(TierType::HOT)
        , success(false)
    {}
};

/**
 * @brief 分配器配置
 */
struct PagedAllocatorConfig {
    size_t block_size;              ///< 块大小 (字节)
    size_t vram_pool_size;          ///< VRAM 池大小
    size_t ram_pool_size;           ///< RAM 池大小
    size_t ssd_swap_size;           ///< SSD 交换区大小
    std::string ssd_swap_path;      ///< SSD 交换文件路径
    bool use_pinned_memory;         ///< 使用锁页内存
    bool enable_compression;        ///< 启用压缩
    float vram_threshold;           ///< VRAM 使用阈值
    float ram_threshold;            ///< RAM 使用阈值
    
    PagedAllocatorConfig()
        : block_size(16 * 1024)     // 16KB
        , vram_pool_size(4ULL * 1024 * 1024 * 1024)  // 4GB
        , ram_pool_size(16ULL * 1024 * 1024 * 1024)  // 16GB
        , ssd_swap_size(100ULL * 1024 * 1024 * 1024) // 100GB
        , ssd_swap_path("/tmp/hetkvcache_swap.bin")
        , use_pinned_memory(true)
        , enable_compression(false)
        , vram_threshold(0.9f)
        , ram_threshold(0.9f)
    {}
};

/**
 * @brief 分页式内存分配器
 * 
 * 管理三级存储层次的内存池，提供块式分配接口。
 */
class PagedAllocator {
public:
    /**
     * @brief 构造函数
     */
    PagedAllocator();
    
    /**
     * @brief 带配置的构造函数
     */
    explicit PagedAllocator(const PagedAllocatorConfig& config);
    
    /**
     * @brief 析构函数
     */
    ~PagedAllocator();
    
    // 禁止拷贝
    PagedAllocator(const PagedAllocator&) = delete;
    PagedAllocator& operator=(const PagedAllocator&) = delete;
    
    /**
     * @brief 初始化分配器
     * @return 成功返回 true
     */
    bool initialize();
    
    /**
     * @brief 关闭分配器
     */
    void shutdown();
    
    /**
     * @brief 分配单个块
     * @param layer_id Transformer 层ID
     * @param head_id 注意力头ID
     * @param preferred_tier 首选层级
     * @return 分配结果
     */
    AllocationResult allocateBlock(LayerId layer_id, 
                                   HeadId head_id,
                                   TierType preferred_tier = TierType::HOT);
    
    /**
     * @brief 分配序列块
     * @param layer_id 层ID
     * @param head_id 头ID
     * @param num_blocks 块数量
     * @return 块ID列表
     */
    std::vector<BlockId> allocateSequence(LayerId layer_id,
                                          HeadId head_id,
                                          size_t num_blocks);
    
    /**
     * @brief 扩展序列
     * @param sequence_id 序列ID
     * @param num_additional_blocks 额外块数量
     * @return 新分配的块ID列表
     */
    std::vector<BlockId> extendSequence(uint64_t sequence_id,
                                        size_t num_additional_blocks);
    
    /**
     * @brief 释放块
     */
    void deallocateBlock(BlockId block_id);
    
    /**
     * @brief 释放序列
     */
    void deallocateSequence(const std::vector<BlockId>& block_ids);
    
    /**
     * @brief 获取块指针
     */
    void* getBlockPtr(BlockId block_id);
    
    /**
     * @brief 获取块指针（const版本）
     */
    const void* getBlockPtr(BlockId block_id) const;
    
    /**
     * @brief 获取块元数据
     */
    BlockMetadata getBlockMetadata(BlockId block_id) const;
    
    /**
     * @brief 更新块元数据
     */
    void updateBlockMetadata(BlockId block_id, const BlockMetadata& metadata);
    
    /**
     * @brief 获取空闲块数量
     */
    size_t getFreeBlocks(TierType tier) const;
    
    /**
     * @brief 获取已用块数量
     */
    size_t getUsedBlocks(TierType tier) const;
    
    /**
     * @brief 获取碎片率
     */
    double getFragmentationRatio() const;
    
    /**
     * @brief 执行碎片整理
     */
    void defragment();
    
    /**
     * @brief 迁移块到其他层级
     */
    bool migrateBlock(BlockId block_id, TierType target_tier);
    
    /**
     * @brief 锁定块（禁止迁移）
     */
    void lockBlock(BlockId block_id);
    
    /**
     * @brief 解锁块
     */
    void unlockBlock(BlockId block_id);
    
    /**
     * @brief 获取层级使用量
     */
    size_t getTierUsage(TierType tier) const;
    
    /**
     * @brief 获取层级容量
     */
    size_t getTierCapacity(TierType tier) const;
    
    /**
     * @brief 获取块大小
     */
    size_t getBlockSize() const { return config_.block_size; }
    
    /**
     * @brief 获取总块数
     */
    size_t getTotalBlocks() const;
    
    /**
     * @brief 获取所有块ID
     */
    std::vector<BlockId> getAllBlockIds() const;
    
    /**
     * @brief 获取指定层的块
     */
    std::vector<BlockId> getBlocksByLayer(LayerId layer_id) const;
    
    /**
     * @brief 获取指定层级的块
     */
    std::vector<BlockId> getBlocksByTier(TierType tier) const;
    
    /**
     * @brief 获取分配统计
     */
    struct AllocatorStats {
        size_t total_blocks;
        size_t free_blocks;
        size_t allocated_blocks;
        size_t locked_blocks;
        size_t vram_blocks;
        size_t ram_blocks;
        size_t ssd_blocks;
        size_t total_allocations;
        size_t total_deallocations;
        double fragmentation_ratio;
        size_t peak_usage;
    };
    
    AllocatorStats getStats() const;
    
    /**
     * @brief 重置统计
     */
    void resetStats();

private:
    // 从指定层级分配块
    BlockId allocateFromTier(TierType tier, LayerId layer_id, HeadId head_id);
    
    // 尝试从其他层级获取空间
    BlockId tryEvictAndAllocate(TierType preferred_tier, LayerId layer_id, HeadId head_id);
    
    // 选择驱逐候选
    std::vector<BlockId> selectEvictionCandidates(TierType tier, size_t num_blocks);
    
    // 初始化内存池
    bool initializeMemoryPools();
    
    // 初始化 SSD 交换区
    bool initializeSSDSwap();
    
    // 分配块ID
    BlockId allocateBlockId();
    
    // 获取块在层级中的偏移
    size_t getBlockOffset(TierType tier, BlockId block_id) const;
    
    // 计算碎片率
    double computeFragmentation() const;

private:
    PagedAllocatorConfig config_;
    mutable std::shared_mutex mutex_;
    
    // 内存池
    void* vram_pool_;
    void* ram_pool_;
    int ssd_fd_;
    
    // 块元数据
    std::unordered_map<BlockId, BlockMetadata> block_metadata_;
    
    // 空闲链表
    std::array<std::queue<BlockId>, NUM_TIERS> free_lists_;
    
    // 层级块映射
    std::unordered_map<TierType, std::vector<BlockId>> tier_blocks_;
    
    // 层块映射
    std::unordered_map<LayerId, std::vector<BlockId>> layer_blocks_;
    
    // 序列管理
    std::unordered_map<uint64_t, SequenceBlockList> sequences_;
    uint64_t next_sequence_id_;
    
    // 统计
    std::atomic<uint64_t> total_allocations_;
    std::atomic<uint64_t> total_deallocations_;
    std::atomic<size_t> peak_usage_;
    
    // 块ID计数器
    std::atomic<BlockId> next_block_id_;
    
    // 层级容量
    std::array<size_t, NUM_TIERS> tier_capacities_;
    size_t tier_usages_[NUM_TIERS];  // 使用普通数组避免atomic的fill问题
    
    // 初始化状态
    std::atomic<bool> initialized_;
};

}  // namespace hetkvcache

#endif  // HETKVCACHE_PAGED_ALLOCATOR_H