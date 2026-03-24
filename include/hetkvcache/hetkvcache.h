/**
 * @file hetkvcache.h
 * @brief HetKVCache 主接口头文件
 * 
 * 提供统一的 KV Cache 管理接口，集成监控、迁移、分配三大模块。
 */

#ifndef HETKVCACHE_H
#define HETKVCACHE_H

#include "hetkvcache/types.h"
#include "hetkvcache/config.h"
#include "hetkvcache/monitor/access_monitor.h"
#include "hetkvcache/monitor/lockfree_monitor.h"
#include "hetkvcache/monitor/heat_evaluator.h"
#include "hetkvcache/monitor/access_pattern_analyzer.h"
#include "hetkvcache/migration/migration_engine.h"
#include "hetkvcache/migration/transfer_scheduler.h"
#include "hetkvcache/migration/prefetcher.h"
#include "hetkvcache/allocator/paged_allocator.h"
#include "hetkvcache/allocator/memory_mapper.h"

#include <memory>
#include <mutex>
#include <atomic>
#include <functional>

namespace hetkvcache {

/**
 * @brief KV Cache 句柄信息
 */
struct KVCacheHandleInfo {
    KVCacheHandle handle;
    uint32_t num_layers;
    uint32_t num_heads;
    size_t head_dim;
    size_t block_size;
    size_t num_blocks;
    std::vector<BlockId> blocks;
    
    KVCacheHandleInfo()
        : handle(INVALID_HANDLE)
        , num_layers(0)
        , num_heads(0)
        , head_dim(0)
        , block_size(0)
        , num_blocks(0)
    {}
};

/**
 * @brief 访问请求
 */
struct AccessRequest {
    uint32_t layer_id;
    uint32_t head_id;
    uint64_t sequence_pos;
    AccessType access_type;
    bool blocking;  ///< 是否阻塞等待
    
    AccessRequest()
        : layer_id(0)
        , head_id(0)
        , sequence_pos(0)
        , access_type(AccessType::READ)
        , blocking(true)
    {}
};

/**
 * @brief HetKVCache 主类
 * 
 * 提供完整的 KV Cache 管理功能，包括分配、访问、迁移和监控。
 */
class HetKVCache {
public:
    /**
     * @brief 创建 HetKVCache 实例
     */
    static HetKVCache* create(const HetKVCacheConfig& config);
    
    /**
     * @brief 销毁 HetKVCache 实例
     */
    static void destroy(HetKVCache* instance);
    
    /**
     * @brief 获取默认配置
     */
    static HetKVCacheConfig getDefaultConfig();
    
    // 禁止拷贝
    HetKVCache(const HetKVCache&) = delete;
    HetKVCache& operator=(const HetKVCache&) = delete;
    
    // ========================================================================
    // KV Cache 生命周期管理
    // ========================================================================
    
    /**
     * @brief 分配 KV Cache
     * @param num_layers Transformer 层数
     * @param num_heads 注意力头数
     * @param head_dim 头维度
     * @param initial_blocks 初始块数量
     * @return KV Cache 句柄
     */
    KVCacheHandle allocateKVCache(uint32_t num_layers,
                                  uint32_t num_heads,
                                  size_t head_dim,
                                  size_t initial_blocks = 1);
    
    /**
     * @brief 释放 KV Cache
     */
    void deallocateKVCache(KVCacheHandle handle);
    
    /**
     * @brief 扩展 KV Cache
     * @param handle KV Cache 句柄
     * @param num_additional_blocks 额外块数量
     * @return 新块ID列表
     */
    std::vector<BlockId> extendKVCache(KVCacheHandle handle,
                                       size_t num_additional_blocks);
    
    /**
     * @brief 获取 KV Cache 信息
     */
    KVCacheHandleInfo getKVCacheInfo(KVCacheHandle handle) const;
    
    // ========================================================================
    // 数据访问接口
    // ========================================================================
    
    /**
     * @brief 访问块数据
     * @param handle KV Cache 句柄
     * @param layer_id 层ID
     * @param head_id 头ID
     * @param sequence_pos 序列位置
     * @param access_type 访问类型
     * @return 数据指针
     */
    void* accessBlock(KVCacheHandle handle,
                     uint32_t layer_id,
                     uint32_t head_id,
                     uint64_t sequence_pos,
                     AccessType access_type = AccessType::READ);
    
    /**
     * @brief 释放块访问（减少引用计数）
     */
    void releaseBlock(KVCacheHandle handle, BlockId block_id);
    
    /**
     * @brief 批量访问块
     */
    std::vector<void*> accessBlocks(KVCacheHandle handle,
                                   const std::vector<AccessRequest>& requests);
    
    /**
     * @brief 获取 K Cache 指针
     */
    float* getKCache(KVCacheHandle handle, uint32_t layer_id, uint64_t block_idx);
    
    /**
     * @brief 获取 V Cache 指针
     */
    float* getVCache(KVCacheHandle handle, uint32_t layer_id, uint64_t block_idx);
    
    // ========================================================================
    // 配置和统计接口
    // ========================================================================
    
    /**
     * @brief 更新配置
     */
    void updateConfig(const HetKVCacheConfig& config);
    
    /**
     * @brief 获取当前配置
     */
    const HetKVCacheConfig& getConfig() const;
    
    /**
     * @brief 获取统计信息
     */
    Statistics getStatistics() const;
    
    /**
     * @brief 重置统计信息
     */
    void resetStatistics();
    
    /**
     * @brief 获取热度分布
     */
    typename HeatEvaluator::HeatDistribution getHeatDistribution() const;
    
    /**
     * @brief 触发手动迁移
     */
    void triggerMigration();
    
    /**
     * @brief 触发碎片整理
     */
    void triggerDefragmentation();
    
    // ========================================================================
    // 回调接口
    // ========================================================================
    
    /**
     * @brief 设置迁移完成回调
     */
    void setMigrationCallback(MigrationCallback callback);
    
    /**
     * @brief 设置访问回调
     */
    void setAccessCallback(AccessCallback callback);
    
    // ========================================================================
    // 调试和诊断接口
    // ========================================================================
    
    /**
     * @brief 打印当前状态
     */
    void printStatus() const;
    
    /**
     * @brief 验证内部一致性
     */
    bool validate() const;
    
    /**
     * @brief 导出诊断信息
     */
    std::string exportDiagnostics() const;

private:
    // 私有构造函数
    HetKVCache();
    ~HetKVCache();
    
    // 初始化
    bool initialize(const HetKVCacheConfig& config);
    
    // 关闭
    void shutdown();
    
    // 后台处理线程
    void backgroundWorker();
    
    // 处理迁移建议
    void processMigrationRecommendations();
    
    // 更新热度分数
    void updateHeatScores();
    
    // 获取块所属句柄
    KVCacheHandle getBlockOwner(BlockId block_id) const;
    
private:
    // 配置
    HetKVCacheConfig config_;
    
    // 模块实例
    std::unique_ptr<AccessMonitor> monitor_;
    std::unique_ptr<LockFreeAccessMonitor> lockfree_monitor_;  // 高性能无锁监控器
    std::unique_ptr<HeatEvaluator> evaluator_;
    std::unique_ptr<AccessPatternAnalyzer> analyzer_;
    std::unique_ptr<MigrationEngine> migrator_;
    std::unique_ptr<TransferScheduler> scheduler_;
    std::unique_ptr<Prefetcher> prefetcher_;
    std::unique_ptr<PagedAllocator> allocator_;
    std::unique_ptr<MemoryMapper> mapper_;
    
    // 使用无锁监控器标志
    bool use_lockfree_monitor_ = true;
    
    // 句柄管理
    mutable std::mutex handle_mutex_;
    std::unordered_map<KVCacheHandle, KVCacheHandleInfo> handles_;
    std::unordered_map<BlockId, KVCacheHandle> block_to_handle_;
    std::atomic<KVCacheHandle> next_handle_;
    
    // 后台线程
    std::thread background_thread_;
    std::atomic<bool> running_;
    
    // 初始化状态
    std::atomic<bool> initialized_;
    
    // 统计
    mutable Statistics stats_;
};

// ============================================================================
// C API (用于框架集成)
// ============================================================================

extern "C" {

/**
 * @brief 创建 HetKVCache 实例
 */
void* hetkvcache_create(const HetKVCacheConfig* config);

/**
 * @brief 销毁 HetKVCache 实例
 */
void hetkvcache_destroy(void* instance);

/**
 * @brief 分配 KV Cache
 */
KVCacheHandle hetkvcache_allocate(void* instance,
                                  uint32_t num_layers,
                                  uint32_t num_heads,
                                  size_t head_dim,
                                  size_t initial_blocks);

/**
 * @brief 释放 KV Cache
 */
void hetkvcache_deallocate(void* instance, KVCacheHandle handle);

/**
 * @brief 访问块
 */
void* hetkvcache_access_block(void* instance,
                             KVCacheHandle handle,
                             uint32_t layer_id,
                             uint32_t head_id,
                             uint64_t sequence_pos,
                             int access_type);

/**
 * @brief 获取统计信息
 */
void hetkvcache_get_statistics(void* instance, Statistics* stats);

/**
 * @brief 触发迁移
 */
void hetkvcache_trigger_migration(void* instance);

// ============================================================================
// vLLM 集成接口
// ============================================================================

/**
 * @brief vLLM 分配回调
 */
void* hetkvcache_vllm_alloc(size_t size, int layer_id, int head_id);

/**
 * @brief vLLM 释放回调
 */
void hetkvcache_vllm_free(void* ptr);

/**
 * @brief vLLM 访问回调
 */
int hetkvcache_vllm_access(void* ptr, int access_type);

// ============================================================================
// HuggingFace Transformers 集成接口
// ============================================================================

/**
 * @brief HF KV Cache 分配
 */
void* hetkvcache_hf_alloc_kv_cache(int batch_size,
                                   int num_layers,
                                   int num_heads,
                                   int head_dim);

/**
 * @brief HF KV Cache 释放
 */
void hetkvcache_hf_free_kv_cache(void* handle);

/**
 * @brief HF 获取 K Cache
 */
float* hetkvcache_hf_get_k_cache(void* handle, int layer_id);

/**
 * @brief HF 获取 V Cache
 */
float* hetkvcache_hf_get_v_cache(void* handle, int layer_id);

}  // extern "C"

}  // namespace hetkvcache

#endif  // HETKVCACHE_H