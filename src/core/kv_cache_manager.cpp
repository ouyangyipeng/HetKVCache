/**
 * @file kv_cache_manager.cpp
 * @brief KV Cache 管理器核心实现
 */

#include "hetkvcache/hetkvcache.h"
#include <cuda_runtime.h>

namespace hetkvcache {

// ============================================================================
// HetKVCache 实现
// ============================================================================

HetKVCache::HetKVCache()
    : next_handle_(1)
    , running_(false)
    , initialized_(false)
{
}

HetKVCache::~HetKVCache() {
    shutdown();
}

HetKVCache* HetKVCache::create(const HetKVCacheConfig& config) {
    HetKVCache* instance = new HetKVCache();
    if (!instance->initialize(config)) {
        delete instance;
        return nullptr;
    }
    return instance;
}

void HetKVCache::destroy(HetKVCache* instance) {
    if (instance) {
        delete instance;
    }
}

HetKVCacheConfig HetKVCache::getDefaultConfig() {
    return HetKVCacheConfig();
}

bool HetKVCache::initialize(const HetKVCacheConfig& config) {
    if (initialized_.load()) {
        return true;
    }
    
    config_ = config;
    
    // 初始化各模块
    monitor_ = std::make_unique<AccessMonitor>();
    evaluator_ = std::make_unique<HeatEvaluator>();
    analyzer_ = std::make_unique<AccessPatternAnalyzer>();
    migrator_ = std::make_unique<MigrationEngine>();
    scheduler_ = std::make_unique<TransferScheduler>();
    prefetcher_ = std::make_unique<Prefetcher>();
    allocator_ = std::make_unique<PagedAllocator>();
    mapper_ = std::make_unique<MemoryMapper>();
    
    // 初始化监控器
    if (!monitor_->initialize(AccessMonitor::MonitorMode::FULL, 1.0f, 1000000)) {
        return false;
    }
    
    // 初始化分配器
    PagedAllocatorConfig allocator_config;
    allocator_config.block_size = config.block_size;
    allocator_config.vram_pool_size = config.vram_budget_mb * 1024 * 1024;
    allocator_config.ram_pool_size = config.ram_budget_mb * 1024 * 1024;
    allocator_config.ssd_swap_size = config.ssd_swap_gb * 1024 * 1024 * 1024;
    allocator_config.ssd_swap_path = config.ssd_swap_path;
    allocator_config.use_pinned_memory = config.use_pinned_memory;
    
    if (!allocator_->initialize()) {
        return false;
    }
    
    // 初始化迁移引擎
    MigrationEngineConfig migration_config;
    migration_config.vram_budget = config.vram_budget_mb * 1024 * 1024;
    migration_config.ram_budget = config.ram_budget_mb * 1024 * 1024;
    migration_config.enable_compression = config.enable_compression;
    
    // 启动后台线程
    running_ = true;
    background_thread_ = std::thread(&HetKVCache::backgroundWorker, this);
    
    // 启动监控
    monitor_->startMonitoring();
    
    initialized_ = true;
    return true;
}

void HetKVCache::shutdown() {
    if (!initialized_.load()) {
        return;
    }
    
    running_ = false;
    if (background_thread_.joinable()) {
        background_thread_.join();
    }
    
    monitor_->stopMonitoring();
    monitor_->shutdown();
    allocator_->shutdown();
    
    initialized_ = false;
}

KVCacheHandle HetKVCache::allocateKVCache(uint32_t num_layers,
                                          uint32_t num_heads,
                                          size_t head_dim,
                                          size_t initial_blocks) {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    
    KVCacheHandleInfo info;
    info.handle = next_handle_.fetch_add(1);
    info.num_layers = num_layers;
    info.num_heads = num_heads;
    info.head_dim = head_dim;
    info.block_size = config_.block_size;
    info.num_blocks = initial_blocks;
    
    // 分配初始块
    for (size_t i = 0; i < initial_blocks; i++) {
        for (uint32_t l = 0; l < num_layers; l++) {
            auto result = allocator_->allocateBlock(l, 0);
            if (result.success) {
                info.blocks.push_back(result.block_id);
                block_to_handle_[result.block_id] = info.handle;
                
                // 注册到热度评估器
                evaluator_->registerBlock(result.block_id, l);
            }
        }
    }
    
    handles_[info.handle] = info;
    
    // 设置分析器参数
    evaluator_->setNumLayers(num_layers);
    evaluator_->setNumHeads(num_heads);
    analyzer_->initialize(num_layers, num_heads);
    
    return info.handle;
}

void HetKVCache::deallocateKVCache(KVCacheHandle handle) {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    
    auto it = handles_.find(handle);
    if (it == handles_.end()) {
        return;
    }
    
    // 释放所有块
    for (BlockId block_id : it->second.blocks) {
        allocator_->deallocateBlock(block_id);
        block_to_handle_.erase(block_id);
        evaluator_->unregisterBlock(block_id);
    }
    
    handles_.erase(it);
}

std::vector<BlockId> HetKVCache::extendKVCache(KVCacheHandle handle,
                                               size_t num_additional_blocks) {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    
    auto it = handles_.find(handle);
    if (it == handles_.end()) {
        return {};
    }
    
    std::vector<BlockId> new_blocks;
    for (size_t i = 0; i < num_additional_blocks; i++) {
        for (uint32_t l = 0; l < it->second.num_layers; l++) {
            auto result = allocator_->allocateBlock(l, 0);
            if (result.success) {
                new_blocks.push_back(result.block_id);
                it->second.blocks.push_back(result.block_id);
                block_to_handle_[result.block_id] = handle;
                evaluator_->registerBlock(result.block_id, l);
            }
        }
    }
    
    it->second.num_blocks += new_blocks.size();
    
    return new_blocks;
}

KVCacheHandleInfo HetKVCache::getKVCacheInfo(KVCacheHandle handle) const {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    
    auto it = handles_.find(handle);
    if (it != handles_.end()) {
        return it->second;
    }
    return KVCacheHandleInfo();
}

void* HetKVCache::accessBlock(KVCacheHandle handle,
                              uint32_t layer_id,
                              uint32_t head_id,
                              uint64_t sequence_pos,
                              AccessType access_type) {
    // 记录访问
    monitor_->recordAccess(sequence_pos, access_type, layer_id, head_id, sequence_pos);
    
    // 查找块
    std::lock_guard<std::mutex> lock(handle_mutex_);
    
    auto handle_it = handles_.find(handle);
    if (handle_it == handles_.end()) {
        return nullptr;
    }
    
    // 简化实现：使用序列位置作为块索引
    size_t block_idx = sequence_pos % handle_it->second.blocks.size();
    if (block_idx >= handle_it->second.blocks.size()) {
        return nullptr;
    }
    
    BlockId block_id = handle_it->second.blocks[block_idx];
    void* ptr = allocator_->getBlockPtr(block_id);
    
    // 更新热度
    evaluator_->processAccess(AccessRecord(block_id, getCurrentTimestamp(),
                                          access_type, layer_id, head_id, sequence_pos));
    
    return ptr;
}

void HetKVCache::releaseBlock(KVCacheHandle handle, BlockId block_id) {
    allocator_->unlockBlock(block_id);
}

std::vector<void*> HetKVCache::accessBlocks(KVCacheHandle handle,
                                            const std::vector<AccessRequest>& requests) {
    std::vector<void*> ptrs;
    ptrs.reserve(requests.size());
    
    for (const auto& req : requests) {
        ptrs.push_back(accessBlock(handle, req.layer_id, req.head_id,
                                  req.sequence_pos, req.access_type));
    }
    
    return ptrs;
}

float* HetKVCache::getKCache(KVCacheHandle handle, uint32_t layer_id, uint64_t block_idx) {
    return static_cast<float*>(accessBlock(handle, layer_id, 0, block_idx, AccessType::READ));
}

float* HetKVCache::getVCache(KVCacheHandle handle, uint32_t layer_id, uint64_t block_idx) {
    return static_cast<float*>(accessBlock(handle, layer_id, 1, block_idx, AccessType::READ));
}

void HetKVCache::updateConfig(const HetKVCacheConfig& config) {
    config_ = config;
}

const HetKVCacheConfig& HetKVCache::getConfig() const {
    return config_;
}

Statistics HetKVCache::getStatistics() const {
    Statistics stats;
    
    auto monitor_stats = monitor_->getStats();
    stats.total_accesses = monitor_stats.total_accesses;
    
    auto allocator_stats = allocator_->getStats();
    stats.total_blocks = allocator_stats.total_blocks;
    stats.free_blocks = allocator_stats.free_blocks;
    stats.allocated_blocks = allocator_stats.allocated_blocks;
    stats.vram_blocks = allocator_stats.vram_blocks;
    stats.ram_blocks = allocator_stats.ram_blocks;
    stats.ssd_blocks = allocator_stats.ssd_blocks;
    
    stats.vram_hits = allocator_stats.vram_blocks;
    stats.ram_hits = allocator_stats.ram_blocks;
    stats.ssd_hits = allocator_stats.ssd_blocks;
    
    stats.migration_stats = migrator_->getStats();
    
    stats.vram_used_bytes = allocator_->getTierUsage(TierType::HOT) * config_.block_size;
    stats.memory_used_bytes = allocator_->getTierUsage(TierType::WARM) * config_.block_size;
    
    return stats;
}

void HetKVCache::resetStatistics() {
    monitor_->resetStats();
    migrator_->resetStats();
}

HeatEvaluator::HeatDistribution HetKVCache::getHeatDistribution() const {
    return evaluator_->getHeatDistribution();
}

void HetKVCache::triggerMigration() {
    Timestamp current_time = getCurrentTimestamp();
    evaluator_->updateAllHeatScores(current_time);
    
    // 获取降级建议
    auto vram_to_ram = evaluator_->getBlocksForMigration(TierType::HOT, TierType::WARM);
    auto ram_to_ssd = evaluator_->getBlocksForMigration(TierType::WARM, TierType::COLD);
    
    // 获取升级建议
    auto ram_to_vram = evaluator_->getBlocksForMigration(TierType::WARM, TierType::HOT);
    auto ssd_to_ram = evaluator_->getBlocksForMigration(TierType::COLD, TierType::WARM);
    
    // 提交迁移任务
    migrator_->processHeatRecommendations(vram_to_ram);
    migrator_->processHeatRecommendations(ram_to_ssd);
    migrator_->processHeatRecommendations(ram_to_vram);
    migrator_->processHeatRecommendations(ssd_to_ram);
}

void HetKVCache::triggerDefragmentation() {
    allocator_->defragment();
}

void HetKVCache::setMigrationCallback(MigrationCallback callback) {
    migrator_->setMigrationCallback(callback);
}

void HetKVCache::setAccessCallback(AccessCallback callback) {
    monitor_->setAccessCallback(callback);
}

void HetKVCache::printStatus() const {
    auto stats = getStatistics();
    
    printf("=== HetKVCache Status ===\n");
    printf("Total Blocks: %lu\n", stats.total_blocks);
    printf("Allocated: %lu, Free: %lu\n", stats.allocated_blocks, stats.free_blocks);
    printf("VRAM: %lu, RAM: %lu, SSD: %lu\n", 
           stats.vram_blocks, stats.ram_blocks, stats.ssd_blocks);
    printf("Total Accesses: %lu\n", stats.total_accesses);
    printf("Cache Hit Rate: %.2f%%\n", stats.cacheHitRate() * 100);
    printf("Total Migrations: %lu\n", stats.migration_stats.total_migrations);
}

bool HetKVCache::validate() const {
    return initialized_.load();
}

std::string HetKVCache::exportDiagnostics() const {
    std::string result = "HetKVCache Diagnostics\n";
    
    auto stats = getStatistics();
    result += "Total Blocks: " + std::to_string(stats.total_blocks) + "\n";
    result += "VRAM Usage: " + std::to_string(stats.vram_blocks) + "\n";
    result += "RAM Usage: " + std::to_string(stats.ram_blocks) + "\n";
    result += "SSD Usage: " + std::to_string(stats.ssd_blocks) + "\n";
    
    return result;
}

void HetKVCache::backgroundWorker() {
    while (running_.load()) {
        // 定期更新热度分数
        updateHeatScores();
        
        // 检查是否需要迁移
        processMigrationRecommendations();
        
        // 休眠一段时间
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void HetKVCache::processMigrationRecommendations() {
    auto recommendations = evaluator_->getBlocksForMigration(TierType::HOT, TierType::WARM, 10);
    if (!recommendations.empty()) {
        migrator_->processHeatRecommendations(recommendations);
    }
}

void HetKVCache::updateHeatScores() {
    Timestamp current_time = getCurrentTimestamp();
    evaluator_->updateAllHeatScores(current_time);
}

KVCacheHandle HetKVCache::getBlockOwner(BlockId block_id) const {
    auto it = block_to_handle_.find(block_id);
    return (it != block_to_handle_.end()) ? it->second : INVALID_HANDLE;
}

// ============================================================================
// C API 实现
// ============================================================================

extern "C" {

void* hetkvcache_create(const HetKVCacheConfig* config) {
    if (!config) {
        return nullptr;
    }
    return HetKVCache::create(*config);
}

void hetkvcache_destroy(void* instance) {
    if (instance) {
        HetKVCache::destroy(static_cast<HetKVCache*>(instance));
    }
}

KVCacheHandle hetkvcache_allocate(void* instance,
                                  uint32_t num_layers,
                                  uint32_t num_heads,
                                  size_t head_dim,
                                  size_t initial_blocks) {
    if (!instance) return INVALID_HANDLE;
    return static_cast<HetKVCache*>(instance)->allocateKVCache(
        num_layers, num_heads, head_dim, initial_blocks);
}

void hetkvcache_deallocate(void* instance, KVCacheHandle handle) {
    if (instance) {
        static_cast<HetKVCache*>(instance)->deallocateKVCache(handle);
    }
}

void* hetkvcache_access_block(void* instance,
                             KVCacheHandle handle,
                             uint32_t layer_id,
                             uint32_t head_id,
                             uint64_t sequence_pos,
                             int access_type) {
    if (!instance) return nullptr;
    return static_cast<HetKVCache*>(instance)->accessBlock(
        handle, layer_id, head_id, sequence_pos,
        static_cast<AccessType>(access_type));
}

void hetkvcache_get_statistics(void* instance, Statistics* stats) {
    if (instance && stats) {
        *stats = static_cast<HetKVCache*>(instance)->getStatistics();
    }
}

void hetkvcache_trigger_migration(void* instance) {
    if (instance) {
        static_cast<HetKVCache*>(instance)->triggerMigration();
    }
}

// vLLM 集成存根
void* hetkvcache_vllm_alloc(size_t size, int layer_id, int head_id) {
    // 简化实现
    return malloc(size);
}

void hetkvcache_vllm_free(void* ptr) {
    free(ptr);
}

int hetkvcache_vllm_access(void* ptr, int access_type) {
    return 0;
}

// HuggingFace 集成存根
void* hetkvcache_hf_alloc_kv_cache(int batch_size, int num_layers, int num_heads, int head_dim) {
    return nullptr;
}

void hetkvcache_hf_free_kv_cache(void* handle) {
}

float* hetkvcache_hf_get_k_cache(void* handle, int layer_id) {
    return nullptr;
}

float* hetkvcache_hf_get_v_cache(void* handle, int layer_id) {
    return nullptr;
}

}  // extern "C"

}  // namespace hetkvcache