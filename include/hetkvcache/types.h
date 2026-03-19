/**
 * @file types.h
 * @brief HetKVCache 核心类型定义
 * 
 * 定义了 KV Cache 管理系统中使用的所有核心数据类型、枚举和结构体。
 * 参考 PagedAttention 和 vLLM 的设计思想。
 */

#ifndef HETKVCACHE_TYPES_H
#define HETKVCACHE_TYPES_H

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <chrono>
#include <memory>
#include <cmath>
#include <functional>

namespace hetkvcache {

// ============================================================================
// 基础类型定义
// ============================================================================

/// 块ID类型
using BlockId = uint64_t;

/// 层ID类型
using LayerId = uint32_t;

/// 注意力头ID类型
using HeadId = uint32_t;

/// 序列位置类型
using SequencePos = uint64_t;

/// 时间戳类型 (纳秒)
using Timestamp = uint64_t;

/// 句柄类型
using KVCacheHandle = uint64_t;

/// 无效句柄常量
constexpr KVCacheHandle INVALID_HANDLE = 0;

/// 无效块ID常量
constexpr BlockId INVALID_BLOCK_ID = UINT64_MAX;

// ============================================================================
// 存储层级枚举
// ============================================================================

/**
 * @brief 存储层级类型
 * 
 * 定义三级存储层次结构：
 * - HOT: GPU 显存 (VRAM)，存放最热数据
 * - WARM: 主机内存 (RAM)，存放温数据
 * - COLD: NVMe SSD，存放冷数据
 */
enum class TierType : uint8_t {
    HOT = 0,    ///< GPU VRAM - 热数据
    WARM = 1,   ///< Host RAM - 温数据  
    COLD = 2    ///< NVMe SSD - 冷数据
};

/// 层级数量
constexpr size_t NUM_TIERS = 3;

/// 获取层级名称
inline const char* tierToString(TierType tier) {
    switch (tier) {
        case TierType::HOT: return "HOT (VRAM)";
        case TierType::WARM: return "WARM (RAM)";
        case TierType::COLD: return "COLD (SSD)";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// 块状态枚举
// ============================================================================

/**
 * @brief 内存块状态
 */
enum class BlockState : uint8_t {
    FREE = 0,       ///< 空闲可用
    ALLOCATED = 1,  ///< 已分配
    MIGRATING = 2,  ///< 正在迁移
    LOCKED = 3,     ///< 锁定（不可迁移）
    CORRUPTED = 4   ///< 损坏
};

/// 获取状态名称
inline const char* stateToString(BlockState state) {
    switch (state) {
        case BlockState::FREE: return "FREE";
        case BlockState::ALLOCATED: return "ALLOCATED";
        case BlockState::MIGRATING: return "MIGRATING";
        case BlockState::LOCKED: return "LOCKED";
        case BlockState::CORRUPTED: return "CORRUPTED";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// 访问类型枚举
// ============================================================================

/**
 * @brief 访问类型
 */
enum class AccessType : uint8_t {
    READ = 0,       ///< 读访问
    WRITE = 1,      ///< 写访问
    READ_WRITE = 2  ///< 读写访问
};

// ============================================================================
// 迁移优先级枚举
// ============================================================================

/**
 * @brief 迁移优先级
 */
enum class MigrationPriority : uint8_t {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    URGENT = 3
};

// ============================================================================
// KV Block 访问记录结构
// ============================================================================

/**
 * @brief KV Block 访问记录
 * 
 * 记录单次 KV Cache 块访问的详细信息，用于热度评估。
 */
struct KVBlockAccess {
    BlockId block_id;           ///< 块唯一标识
    LayerId layer_id;           ///< Transformer 层 ID
    HeadId head_id;             ///< 注意力头 ID
    Timestamp timestamp;        ///< 访问时间戳 (纳秒)
    AccessType access_type;     ///< 访问类型
    SequencePos sequence_pos;   ///< 序列位置
    
    KVBlockAccess()
        : block_id(INVALID_BLOCK_ID)
        , layer_id(0)
        , head_id(0)
        , timestamp(0)
        , access_type(AccessType::READ)
        , sequence_pos(0)
    {}
    
    KVBlockAccess(BlockId bid, LayerId lid, HeadId hid, 
                  Timestamp ts, AccessType at, SequencePos sp)
        : block_id(bid)
        , layer_id(lid)
        , head_id(hid)
        , timestamp(ts)
        , access_type(at)
        , sequence_pos(sp)
    {}
};

// ============================================================================
// 热度分数结构
// ============================================================================

/**
 * @brief 热度分数结构
 * 
 * 存储每个 KV Block 的热度评估结果。
 */
struct HeatScore {
    BlockId block_id;           ///< 块ID
    float score;                ///< 热度分数 [0.0, 1.0]
    TierType current_tier;      ///< 当前所在层级
    TierType recommended_tier;  ///< 推荐迁移层级
    Timestamp last_access_ts;   ///< 最后访问时间戳
    uint32_t access_count;      ///< 访问计数
    float decayed_score;        ///< 衰减后的分数
    float frequency_score;      ///< 频率分数
    float recency_score;        ///< 新近度分数
    float layer_importance;      ///< 层重要性分数
    
    HeatScore()
        : block_id(INVALID_BLOCK_ID)
        , score(0.0f)
        , current_tier(TierType::HOT)
        , recommended_tier(TierType::HOT)
        , last_access_ts(0)
        , access_count(0)
        , decayed_score(0.0f)
        , frequency_score(0.0f)
        , recency_score(0.0f)
        , layer_importance(1.0f)
    {}
};

// ============================================================================
// 迁移任务结构
// ============================================================================

/**
 * @brief 迁移任务结构
 * 
 * 描述一次数据迁移操作的所有信息。
 */
struct MigrationTask {
    BlockId block_id;           ///< 块ID
    TierType source_tier;       ///< 源层级
    TierType target_tier;       ///< 目标层级
    void* source_ptr;           ///< 源地址
    void* target_ptr;           ///< 目标地址
    size_t size;                ///< 数据大小
    MigrationPriority priority; ///< 优先级
    Timestamp submit_time;      ///< 提交时间
    Timestamp start_time;       ///< 开始时间
    Timestamp end_time;         ///< 结束时间
    bool completed;             ///< 是否完成
    bool success;               ///< 是否成功
    
    MigrationTask()
        : block_id(INVALID_BLOCK_ID)
        , source_tier(TierType::HOT)
        , target_tier(TierType::WARM)
        , source_ptr(nullptr)
        , target_ptr(nullptr)
        , size(0)
        , priority(MigrationPriority::NORMAL)
        , submit_time(0)
        , start_time(0)
        , end_time(0)
        , completed(false)
        , success(false)
    {}
};

// ============================================================================
// 迁移统计结构
// ============================================================================

/**
 * @brief 迁移统计结构
 */
struct MigrationStats {
    uint64_t total_migrations{0};
    uint64_t vram_to_ram{0};
    uint64_t ram_to_vram{0};
    uint64_t ram_to_ssd{0};
    uint64_t ssd_to_ram{0};
    uint64_t vram_to_ssd{0};
    uint64_t ssd_to_vram{0};
    double total_migration_time_ms{0.0};
    uint64_t total_bytes_transferred{0};
    uint64_t failed_migrations{0};
    
    void reset() {
        total_migrations = 0;
        vram_to_ram = 0;
        ram_to_vram = 0;
        ram_to_ssd = 0;
        ssd_to_ram = 0;
        vram_to_ssd = 0;
        ssd_to_vram = 0;
        total_migration_time_ms = 0.0;
        total_bytes_transferred = 0;
        failed_migrations = 0;
    }
    
    double avgMigrationTimeMs() const {
        return total_migrations > 0 ? total_migration_time_ms / total_migrations : 0.0;
    }
};

// ============================================================================
// 内存块头结构
// ============================================================================

/**
 * @brief 内存块头结构
 *
 * 每个 KV Block 的元数据，优化为紧凑结构。
 */
struct BlockHeader {
    BlockId block_id;           ///< 块唯一标识 (8 bytes)
    uint64_t data_ptr_or_offset;///< 数据指针或文件偏移 (8 bytes)
    SequencePos sequence_start; ///< 序列起始位置 (8 bytes)
    Timestamp last_access;      ///< 最后访问时间 (8 bytes)
    LayerId layer_id;           ///< Transformer 层 ID (4 bytes)
    uint32_t access_count;      ///< 访问计数 (4 bytes)
    SequencePos sequence_end;   ///< 序列结束位置 (8 bytes)
    float heat_score;           ///< 热度分数 (4 bytes)
    uint16_t tier_state;        ///< tier + state packed (2 bytes)
    uint16_t ref_count;         ///< 引用计数 (2 bytes)
    uint32_t padding;           ///< 对齐填充 (4 bytes)
    
    BlockHeader()
        : block_id(INVALID_BLOCK_ID)
        , data_ptr_or_offset(0)
        , sequence_start(0)
        , last_access(0)
        , layer_id(0)
        , access_count(0)
        , sequence_end(0)
        , heat_score(0.0f)
        , tier_state(0)
        , ref_count(0)
        , padding(0)
    {}
    
    TierType getTier() const { return static_cast<TierType>(tier_state & 0xFF); }
    void setTier(TierType t) { tier_state = (tier_state & 0xFF00) | static_cast<uint16_t>(t); }
    BlockState getState() const { return static_cast<BlockState>((tier_state >> 8) & 0xFF); }
    void setState(BlockState s) { tier_state = (tier_state & 0x00FF) | (static_cast<uint16_t>(s) << 8); }
};

static_assert(sizeof(BlockHeader) <= 64, "BlockHeader must be <= 64 bytes");

// ============================================================================
// 层级位置结构
// ============================================================================

/**
 * @brief 层级位置结构
 * 
 * 描述一个块在特定层级中的位置信息。
 */
struct TierLocation {
    TierType tier;          ///< 层级类型
    void* ptr;              ///< 内存指针
    uint64_t offset;        ///< 文件偏移（用于 SSD）
    size_t size;            ///< 数据大小
    bool valid;             ///< 是否有效
    
    TierLocation()
        : tier(TierType::HOT)
        , ptr(nullptr)
        , offset(0)
        , size(0)
        , valid(false)
    {}
};

// ============================================================================
// 内存池配置结构
// ============================================================================

/**
 * @brief 内存池配置结构
 */
struct PoolConfig {
    size_t block_size;              ///< 块大小，默认 16KB
    size_t vram_pool_size;           ///< VRAM 池大小 (字节)
    size_t ram_pool_size;            ///< RAM 池大小 (字节)
    size_t ssd_swap_size;            ///< SSD 交换区大小 (字节)
    std::string ssd_swap_path;       ///< SSD 交换文件路径
    bool use_pinned_memory;          ///< 是否使用锁页内存
    bool enable_compression;         ///< 是否启用压缩
    
    PoolConfig()
        : block_size(16 * 1024)      // 16KB
        , vram_pool_size(4ULL * 1024 * 1024 * 1024)  // 4GB
        , ram_pool_size(16ULL * 1024 * 1024 * 1024)  // 16GB
        , ssd_swap_size(100ULL * 1024 * 1024 * 1024) // 100GB
        , ssd_swap_path("/tmp/hetkvcache_swap.bin")
        , use_pinned_memory(true)
        , enable_compression(false)
    {}
};

// ============================================================================
// 系统配置结构
// ============================================================================

/**
 * @brief HetKVCache 系统配置
 */
struct HetKVCacheConfig {
    // 内存配置
    size_t vram_budget_mb;           ///< VRAM 预算 (MB)
    size_t ram_budget_mb;            ///< RAM 预算 (MB)
    size_t ssd_swap_gb;              ///< SSD 交换区大小 (GB)
    std::string ssd_swap_path;       ///< SSD 交换文件路径
    
    // 热度评估配置
    float hot_threshold;             ///< 热数据阈值 [0.0, 1.0]
    float warm_threshold;            ///< 温数据阈值 [0.0, 1.0]
    float alpha;                     ///< 频率权重
    float beta;                      ///< 新近度权重
    float gamma;                     ///< 层重要性权重
    float lambda;                    ///< 衰减系数
    
    // 迁移配置
    bool enable_prefetch;            ///< 是否启用预取
    bool enable_compression;         ///< 是否启用压缩
    size_t max_concurrent_migrations;///< 最大并发迁移数
    size_t migration_batch_size;     ///< 迁移批大小
    
    // 分配器配置
    size_t block_size;               ///< 块大小 (字节)
    bool use_pinned_memory;          ///< 是否使用锁页内存
    
    // 日志配置
    int log_level;                   ///< 日志级别 (0=off, 1=error, 2=warn, 3=info, 4=debug)
    std::string log_file;            ///< 日志文件路径
    
    HetKVCacheConfig()
        : vram_budget_mb(4096)       // 4GB
        , ram_budget_mb(16384)       // 16GB
        , ssd_swap_gb(100)           // 100GB
        , ssd_swap_path("/tmp/hetkvcache_swap.bin")
        , hot_threshold(0.7f)
        , warm_threshold(0.3f)
        , alpha(0.5f)
        , beta(0.3f)
        , gamma(0.2f)
        , lambda(0.01f)
        , enable_prefetch(true)
        , enable_compression(false)
        , max_concurrent_migrations(8)
        , migration_batch_size(16)
        , block_size(16 * 1024)      // 16KB
        , use_pinned_memory(true)
        , log_level(2)
        , log_file("")
    {}
};

// ============================================================================
// 统计信息结构
// ============================================================================

/**
 * @brief 统计信息结构
 */
struct Statistics {
    // 缓存统计
    uint64_t total_blocks;
    uint64_t free_blocks;
    uint64_t allocated_blocks;
    uint64_t migrating_blocks;
    
    // 层级统计
    uint64_t vram_blocks;
    uint64_t ram_blocks;
    uint64_t ssd_blocks;
    
    // 访问统计
    uint64_t total_accesses;
    uint64_t vram_hits;
    uint64_t ram_hits;
    uint64_t ssd_hits;
    uint64_t misses;
    
    // 迁移统计
    MigrationStats migration_stats;
    
    // 性能统计
    double avg_access_latency_ns;
    double avg_migration_latency_ms;
    double throughput_tokens_per_sec;
    
    // 资源使用
    double cpu_usage_percent;
    size_t memory_used_bytes;
    size_t vram_used_bytes;
    
    Statistics()
        : total_blocks(0)
        , free_blocks(0)
        , allocated_blocks(0)
        , migrating_blocks(0)
        , vram_blocks(0)
        , ram_blocks(0)
        , ssd_blocks(0)
        , total_accesses(0)
        , vram_hits(0)
        , ram_hits(0)
        , ssd_hits(0)
        , misses(0)
        , avg_access_latency_ns(0.0)
        , avg_migration_latency_ms(0.0)
        , throughput_tokens_per_sec(0.0)
        , cpu_usage_percent(0.0)
        , memory_used_bytes(0)
        , vram_used_bytes(0)
    {}
    
    // 拷贝构造函数
    Statistics(const Statistics& other) = default;
    
    // 拷贝赋值运算符
    Statistics& operator=(const Statistics& other) = default;
    
    double cacheHitRate() const {
        uint64_t total = vram_hits + ram_hits + ssd_hits + misses;
        return total > 0 ? static_cast<double>(vram_hits + ram_hits + ssd_hits) / total : 0.0;
    }
    
    double vramHitRate() const {
        uint64_t total = vram_hits + ram_hits + ssd_hits + misses;
        return total > 0 ? static_cast<double>(vram_hits) / total : 0.0;
    }
};

// ============================================================================
// 访问模式分析报告
// ============================================================================

/**
 * @brief 访问模式分析报告
 */
struct PatternReport {
    // 层级访问分布
    std::vector<std::vector<uint64_t>> layer_access_counts;  // [layer][head]
    
    // 时间分布
    std::vector<uint64_t> time_distribution;  // 按时间窗口
    
    // 序列位置分布
    std::vector<uint64_t> sequence_distribution;
    
    // 热点块
    std::vector<BlockId> hot_blocks;
    std::vector<BlockId> cold_blocks;
    
    // 预测信息
    std::vector<BlockId> predicted_next_access;
    
    // 统计摘要
    uint64_t total_accesses;
    uint64_t unique_blocks_accessed;
    double avg_access_per_block;
    double entropy;  // 访问分布熵
    
    PatternReport()
        : total_accesses(0)
        , unique_blocks_accessed(0)
        , avg_access_per_block(0.0)
        , entropy(0.0)
    {}
};

// ============================================================================
// 工具函数
// ============================================================================

/**
 * @brief 获取当前时间戳 (纳秒)
 */
inline Timestamp getCurrentTimestamp() {
    auto now = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch());
    return static_cast<Timestamp>(ns.count());
}

/**
 * @brief 计算指数衰减
 */
inline float exponentialDecay(float value, float lambda, Timestamp time_diff_ns) {
    // 将纳秒转换为秒
    double time_diff_sec = static_cast<double>(time_diff_ns) / 1e9;
    return value * std::exp(-lambda * time_diff_sec);
}

/**
 * @brief 安全的对齐分配
 */
inline size_t alignTo(size_t size, size_t alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
}

}  // namespace hetkvcache

#endif  // HETKVCACHE_TYPES_H