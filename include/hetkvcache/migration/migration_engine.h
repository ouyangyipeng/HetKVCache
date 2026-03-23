/**
 * @file migration_engine.h
 * @brief 动态多级缓存迁移引擎
 * 
 * 负责 KV Cache 数据在 GPU VRAM <-> Host RAM <-> NVMe SSD 之间的迁移。
 * 利用 CUDA Streams 实现异步传输，最小化 GPU 等待时间。
 */

#ifndef HETKVCACHE_MIGRATION_ENGINE_H
#define HETKVCACHE_MIGRATION_ENGINE_H

#include "hetkvcache/types.h"
#include "hetkvcache/monitor/heat_evaluator.h"
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <atomic>
#include <unordered_map>
#include <vector>
#include <memory>
#include <functional>
#include <future>

// 根据编译后端选择适当的头文件和类型定义
#ifdef USE_ASCEND
#include "hetkvcache/device/cuda_compat.h"
#else
// CUDA 前向声明
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;
#endif

namespace hetkvcache {

// 前向声明
class TransferScheduler;
class Prefetcher;
class StreamManager;

/**
 * @brief 迁移任务状态
 */
enum class MigrationStatus {
    PENDING,        ///< 等待执行
    IN_PROGRESS,    ///< 执行中
    COMPLETED,      ///< 已完成
    FAILED,         ///< 失败
    CANCELLED       ///< 已取消
};

/**
 * @brief 迁移任务信息
 */
struct MigrationTaskInfo {
    uint64_t task_id;
    BlockId block_id;
    TierType source_tier;
    TierType target_tier;
    void* source_ptr;
    void* target_ptr;
    size_t size;
    MigrationPriority priority;
    MigrationStatus status;
    Timestamp submit_time;
    Timestamp start_time;
    Timestamp end_time;
    cudaStream_t stream;
    std::unique_ptr<std::promise<bool>> completion_promise;
    std::string error_message;
    
    MigrationTaskInfo()
        : task_id(0)
        , block_id(INVALID_BLOCK_ID)
        , source_tier(TierType::HOT)
        , target_tier(TierType::WARM)
        , source_ptr(nullptr)
        , target_ptr(nullptr)
        , size(0)
        , priority(MigrationPriority::NORMAL)
        , status(MigrationStatus::PENDING)
        , submit_time(0)
        , start_time(0)
        , end_time(0)
        , stream(nullptr)
        , completion_promise(std::make_unique<std::promise<bool>>())
    {}
    
    MigrationTaskInfo(MigrationTaskInfo&& other) noexcept = default;
    MigrationTaskInfo& operator=(MigrationTaskInfo&& other) noexcept = default;
    MigrationTaskInfo(const MigrationTaskInfo&) = delete;
    MigrationTaskInfo& operator=(const MigrationTaskInfo&) = delete;
};

/**
 * @brief 迁移完成回调
 */
using MigrationCallback = std::function<void(const MigrationTaskInfo& task)>;

/**
 * @brief 迁移引擎配置
 */
struct MigrationEngineConfig {
    size_t max_concurrent_transfers;    ///< 最大并发传输数
    size_t transfer_batch_size;         ///< 传输批大小
    size_t vram_budget;                 ///< VRAM 预算 (字节)
    size_t ram_budget;                  ///< RAM 预算 (字节)
    bool enable_overlap;                ///< 是否启用计算传输重叠
    bool enable_compression;            ///< 是否启用压缩 (SSD)
    float prefetch_threshold;           ///< 预取阈值
    
    MigrationEngineConfig()
        : max_concurrent_transfers(8)
        , transfer_batch_size(16)
        , vram_budget(4ULL * 1024 * 1024 * 1024)  // 4GB
        , ram_budget(16ULL * 1024 * 1024 * 1024)  // 16GB
        , enable_overlap(true)
        , enable_compression(false)
        , prefetch_threshold(0.6f)
    {}
};

/**
 * @brief 迁移引擎
 * 
 * 管理三级存储层次之间的数据迁移。
 */
class MigrationEngine {
public:
    /**
     * @brief 构造函数
     */
    MigrationEngine();
    
    /**
     * @brief 带配置的构造函数
     */
    explicit MigrationEngine(const MigrationEngineConfig& config);
    
    /**
     * @brief 析构函数
     */
    ~MigrationEngine();
    
    // 禁止拷贝
    MigrationEngine(const MigrationEngine&) = delete;
    MigrationEngine& operator=(const MigrationEngine&) = delete;
    
    /**
     * @brief 初始化迁移引擎
     * @param vram_base VRAM 基地址
     * @param vram_size VRAM 大小
     * @param ram_base RAM 基地址
     * @param ram_size RAM 大小
     * @param ssd_fd SSD 文件描述符
     * @return 成功返回 true
     */
    bool initialize(void* vram_base, size_t vram_size,
                   void* ram_base, size_t ram_size,
                   int ssd_fd);
    
    /**
     * @brief 关闭迁移引擎
     */
    void shutdown();
    
    /**
     * @brief 提交迁移任务
     * @param block_id 块ID
     * @param source_tier 源层级
     * @param target_tier 目标层级
     * @param priority 优先级
     * @return 任务ID
     */
    uint64_t submitMigration(BlockId block_id, 
                            TierType source_tier, 
                            TierType target_tier,
                            MigrationPriority priority = MigrationPriority::NORMAL);
    
    /**
     * @brief 提交迁移任务（带地址）
     */
    uint64_t submitMigration(BlockId block_id,
                            void* source_ptr,
                            void* target_ptr,
                            size_t size,
                            TierType source_tier,
                            TierType target_tier,
                            MigrationPriority priority = MigrationPriority::NORMAL);
    
    /**
     * @brief 批量提交迁移任务
     */
    std::vector<uint64_t> submitMigrationBatch(
        const std::vector<std::tuple<BlockId, TierType, TierType>>& tasks);
    
    /**
     * @brief 等待迁移完成
     * @param task_id 任务ID
     * @param timeout_ms 超时时间 (毫秒)，0 表示无限等待
     * @return 成功返回 true
     */
    bool awaitMigration(uint64_t task_id, uint64_t timeout_ms = 0);
    
    /**
     * @brief 等待所有待处理迁移完成
     */
    void awaitAllMigrations();
    
    /**
     * @brief 取消迁移任务
     */
    bool cancelMigration(uint64_t task_id);
    
    /**
     * @brief 设置迁移完成回调
     */
    void setMigrationCallback(MigrationCallback callback);
    
    /**
     * @brief 获取任务状态
     */
    MigrationStatus getTaskStatus(uint64_t task_id) const;
    
    /**
     * @brief 获取迁移统计
     */
    MigrationStats getStats() const;
    
    /**
     * @brief 重置统计
     */
    void resetStats();
    
    /**
     * @brief 获取正在进行的迁移数
     */
    size_t getPendingCount() const;
    
    /**
     * @brief 根据热度建议触发迁移
     */
    void processHeatRecommendations(
        const std::vector<MigrationRecommendation>& recommendations);
    
    /**
     * @brief 设置块的地址映射
     */
    void setBlockAddress(BlockId block_id, 
                        TierType tier, 
                        void* ptr, 
                        size_t size);
    
    /**
     * @brief 获取块的地址
     */
    void* getBlockAddress(BlockId block_id, TierType tier) const;
    
    /**
     * @brief 触发预取
     */
    void triggerPrefetch(const std::vector<BlockId>& blocks);
    
    /**
     * @brief 获取层级使用量
     */
    size_t getTierUsage(TierType tier) const;
    
    /**
     * @brief 获取层级容量
     */
    size_t getTierCapacity(TierType tier) const;

private:
    // 工作线程函数
    void workerThread();
    
    // 执行单个迁移任务
    bool executeMigration(MigrationTaskInfo& task);
    
    // VRAM <-> RAM 传输
    bool transferVRAMToRAM(void* src, void* dst, size_t size, cudaStream_t stream);
    bool transferRAMToVRAM(void* src, void* dst, size_t size, cudaStream_t stream);
    
    // RAM <-> SSD 传输
    bool transferRAMToSSD(void* src, uint64_t offset, size_t size);
    bool transferSSDToRAM(uint64_t offset, void* dst, size_t size);
    
    // VRAM <-> SSD 直接传输 (经过 RAM 缓冲)
    bool transferVRAMToSSD(void* src, uint64_t offset, size_t size, cudaStream_t stream);
    bool transferSSDToVRAM(uint64_t offset, void* dst, size_t size, cudaStream_t stream);
    
    // 获取下一个任务ID
    uint64_t getNextTaskId() { return next_task_id_.fetch_add(1); }
    
    // 分配 CUDA Stream
    cudaStream_t allocateStream();
    void releaseStream(cudaStream_t stream);
    
    // 更新统计
    void updateStats(const MigrationTaskInfo& task, bool success);

private:
    // 配置
    MigrationEngineConfig config_;
    
    // 初始化状态
    std::atomic<bool> initialized_;
    
    // 存储区域
    void* vram_base_;
    size_t vram_size_;
    void* ram_base_;
    size_t ram_size_;
    int ssd_fd_;
    
    // CUDA Streams
    std::vector<cudaStream_t> stream_pool_;
    std::queue<cudaStream_t> available_streams_;
    mutable std::mutex stream_mutex_;
    
    // 任务队列
    std::priority_queue<std::pair<int, uint64_t>> task_priority_queue_;
    std::unordered_map<uint64_t, MigrationTaskInfo> tasks_;
    mutable std::mutex task_mutex_;
    std::condition_variable task_cv_;
    
    // 工作线程
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_;
    
    // 统计
    MigrationStats stats_;
    mutable std::mutex stats_mutex_;
    
    // 任务ID计数器
    std::atomic<uint64_t> next_task_id_;
    
    // 回调
    MigrationCallback callback_;
    
    // 层级使用量跟踪
    std::atomic<size_t> vram_usage_;
    std::atomic<size_t> ram_usage_;
    std::atomic<size_t> ssd_usage_;
    
    // 块地址映射
    mutable std::mutex address_mutex_;
    std::unordered_map<BlockId, std::array<void*, NUM_TIERS>> block_addresses_;
    std::unordered_map<BlockId, size_t> block_sizes_;
};

}  // namespace hetkvcache

#endif  // HETKVCACHE_MIGRATION_ENGINE_H