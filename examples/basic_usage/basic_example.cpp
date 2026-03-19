/**
 * @file basic_example.cpp
 * @brief HetKVCache基础使用示例
 * 
 * 本示例演示如何使用HetKVCache进行基本的KV缓存管理操作：
 * - 初始化缓存管理器
 * - 分配和释放KV Cache
 * - 访问数据块
 * - 监控统计信息
 * - 手动触发迁移
 */

#include <hetkvcache/hetkvcache.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>

namespace hetkvcache {

// 生成模拟的KV缓存数据
std::vector<float> generate_kv_cache_data(size_t num_elements) {
    std::vector<float> data(num_elements);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (auto& val : data) {
        val = dist(rng);
    }
    return data;
}

// 示例1：基本KV Cache操作
void basic_kv_cache_operations() {
    std::cout << "\n=== 示例1：基本KV Cache操作 ===\n";
    
    // 创建配置
    HetKVCacheConfig config;
    config.vram_budget_mb = 1024;   // 1GB VRAM
    config.ram_budget_mb = 4096;    // 4GB RAM
    config.block_size = 16 * 1024;  // 16KB blocks
    
    // 初始化缓存管理器
    auto cache = HetKVCache::create(config);
    if (!cache) {
        std::cerr << "初始化缓存管理器失败！\n";
        return;
    }
    std::cout << "缓存管理器初始化成功\n";
    
    // 分配KV Cache (模拟一个4层、8头的Transformer)
    uint32_t num_layers = 4;
    uint32_t num_heads = 8;
    size_t head_dim = 128;
    size_t initial_blocks = 10;
    
    auto handle = cache->allocateKVCache(num_layers, num_heads, head_dim, initial_blocks);
    if (handle == INVALID_HANDLE) {
        std::cerr << "分配KV Cache失败！\n";
        HetKVCache::destroy(cache);
        return;
    }
    std::cout << "分配KV Cache成功, handle=" << handle << "\n";
    
    // 获取KV Cache信息
    auto info = cache->getKVCacheInfo(handle);
    std::cout << "KV Cache信息:\n";
    std::cout << "  层数: " << info.num_layers << "\n";
    std::cout << "  头数: " << info.num_heads << "\n";
    std::cout << "  头维度: " << info.head_dim << "\n";
    std::cout << "  块大小: " << info.block_size << "\n";
    std::cout << "  块数量: " << info.num_blocks << "\n";
    
    // 扩展KV Cache
    auto new_blocks = cache->extendKVCache(handle, 5);
    std::cout << "扩展了 " << new_blocks.size() << " 个新块\n";
    
    // 访问块数据
    if (!info.blocks.empty()) {
        auto* data_ptr = cache->accessBlock(handle, 0, 0, 0, AccessType::READ);
        if (data_ptr) {
            std::cout << "成功访问块数据\n";
        }
        cache->releaseBlock(handle, info.blocks[0]);
    }
    
    // 获取K/V Cache指针
    float* k_cache = cache->getKCache(handle, 0, 0);
    float* v_cache = cache->getVCache(handle, 0, 0);
    if (k_cache && v_cache) {
        std::cout << "成功获取K/V Cache指针\n";
    }
    
    // 释放KV Cache
    cache->deallocateKVCache(handle);
    std::cout << "释放KV Cache完成\n";
    
    // 清理
    HetKVCache::destroy(cache);
}

// 示例2：统计信息演示
void statistics_demo() {
    std::cout << "\n=== 示例2：统计信息演示 ===\n";
    
    HetKVCacheConfig config;
    config.vram_budget_mb = 512;    // 512MB
    config.ram_budget_mb = 2048;    // 2GB
    config.block_size = 16 * 1024;
    
    auto cache = HetKVCache::create(config);
    if (!cache) {
        std::cerr << "初始化失败\n";
        return;
    }
    
    // 分配多个KV Cache
    std::vector<KVCacheHandle> handles;
    for (int i = 0; i < 5; ++i) {
        auto handle = cache->allocateKVCache(4, 8, 128, 20);
        if (handle != INVALID_HANDLE) {
            handles.push_back(handle);
        }
    }
    std::cout << "分配了 " << handles.size() << " 个KV Cache\n";
    
    // 模拟访问
    for (auto handle : handles) {
        auto info = cache->getKVCacheInfo(handle);
        for (size_t i = 0; i < info.blocks.size() && i < 10; ++i) {
            cache->accessBlock(handle, 0, 0, i, AccessType::READ);
            cache->releaseBlock(handle, info.blocks[i]);
        }
    }
    
    // 获取统计信息
    auto stats = cache->getStatistics();
    std::cout << "统计信息:\n";
    std::cout << "  总块数: " << stats.total_blocks << "\n";
    std::cout << "  已分配块: " << stats.allocated_blocks << "\n";
    std::cout << "  空闲块: " << stats.free_blocks << "\n";
    std::cout << "  VRAM块: " << stats.vram_blocks << "\n";
    std::cout << "  RAM块: " << stats.ram_blocks << "\n";
    std::cout << "  SSD块: " << stats.ssd_blocks << "\n";
    std::cout << "  总访问次数: " << stats.total_accesses << "\n";
    std::cout << "  缓存命中率: " << (stats.cacheHitRate() * 100) << "%\n";
    std::cout << "  VRAM命中率: " << (stats.vramHitRate() * 100) << "%\n";
    
    // 清理
    for (auto handle : handles) {
        cache->deallocateKVCache(handle);
    }
    HetKVCache::destroy(cache);
}

// 示例3：迁移演示
void migration_demo() {
    std::cout << "\n=== 示例3：迁移演示 ===\n";
    
    HetKVCacheConfig config;
    config.vram_budget_mb = 256;    // 256MB VRAM (较小容量以触发迁移)
    config.ram_budget_mb = 1024;    // 1GB RAM
    config.block_size = 16 * 1024;
    config.enable_prefetch = true;
    
    auto cache = HetKVCache::create(config);
    if (!cache) {
        std::cerr << "初始化失败\n";
        return;
    }
    
    // 分配大量KV Cache以触发迁移
    std::vector<KVCacheHandle> handles;
    for (int i = 0; i < 20; ++i) {
        auto handle = cache->allocateKVCache(4, 8, 128, 50);
        if (handle != INVALID_HANDLE) {
            handles.push_back(handle);
        }
    }
    std::cout << "分配了 " << handles.size() << " 个KV Cache\n";
    
    // 模拟差异化访问模式
    // 前一半句柄：热数据
    for (int iter = 0; iter < 50; ++iter) {
        for (size_t i = 0; i < handles.size() / 2; ++i) {
            auto info = cache->getKVCacheInfo(handles[i]);
            for (size_t j = 0; j < info.blocks.size() && j < 5; ++j) {
                cache->accessBlock(handles[i], 0, 0, j, AccessType::READ);
                cache->releaseBlock(handles[i], info.blocks[j]);
            }
        }
    }
    
    // 后一半句柄：冷数据（只访问一次）
    for (size_t i = handles.size() / 2; i < handles.size(); ++i) {
        auto info = cache->getKVCacheInfo(handles[i]);
        if (!info.blocks.empty()) {
            cache->accessBlock(handles[i], 0, 0, 0, AccessType::READ);
            cache->releaseBlock(handles[i], info.blocks[0]);
        }
    }
    
    // 触发迁移
    std::cout << "触发迁移评估...\n";
    cache->triggerMigration();
    
    // 等待迁移完成
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // 获取热度分布
    auto heat_dist = cache->getHeatDistribution();
    std::cout << "热度分布:\n";
    std::cout << "  热块数: " << heat_dist.hot_blocks << "\n";
    std::cout << "  温块数: " << heat_dist.warm_blocks << "\n";
    std::cout << "  冷块数: " << heat_dist.cold_blocks << "\n";
    
    // 获取统计信息
    auto stats = cache->getStatistics();
    std::cout << "迁移统计:\n";
    std::cout << "  迁移中块: " << stats.migrating_blocks << "\n";
    std::cout << "  平均访问延迟: " << stats.avg_access_latency_ns << " ns\n";
    
    // 清理
    for (auto handle : handles) {
        cache->deallocateKVCache(handle);
    }
    HetKVCache::destroy(cache);
}

// 示例4：并发访问演示
void concurrent_access_demo() {
    std::cout << "\n=== 示例4：并发访问演示 ===\n";
    
    HetKVCacheConfig config;
    config.vram_budget_mb = 1024;   // 1GB
    config.ram_budget_mb = 4096;    // 4GB
    config.block_size = 16 * 1024;
    
    auto cache = HetKVCache::create(config);
    if (!cache) {
        std::cerr << "初始化失败\n";
        return;
    }
    
    const int num_threads = 4;
    const int caches_per_thread = 10;
    
    std::vector<std::thread> threads;
    std::atomic<int> total_accesses{0};
    
    auto worker = [&](int thread_id) {
        // 每个线程分配自己的KV Cache
        std::vector<KVCacheHandle> my_handles;
        for (int i = 0; i < caches_per_thread; ++i) {
            auto handle = cache->allocateKVCache(2, 4, 64, 10);
            if (handle != INVALID_HANDLE) {
                my_handles.push_back(handle);
            }
        }
        
        // 访问数据
        for (int iter = 0; iter < 10; ++iter) {
            for (auto handle : my_handles) {
                auto info = cache->getKVCacheInfo(handle);
                for (size_t j = 0; j < info.blocks.size() && j < 3; ++j) {
                    cache->accessBlock(handle, 0, 0, j, AccessType::READ);
                    cache->releaseBlock(handle, info.blocks[j]);
                    total_accesses++;
                }
            }
        }
        
        // 清理
        for (auto handle : my_handles) {
            cache->deallocateKVCache(handle);
        }
    };
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << "并发测试完成:\n";
    std::cout << "  线程数: " << num_threads << "\n";
    std::cout << "  每线程KV Cache数: " << caches_per_thread << "\n";
    std::cout << "  总访问次数: " << total_accesses.load() << "\n";
    std::cout << "  总耗时: " << duration << " ms\n";
    
    // 最终统计
    auto stats = cache->getStatistics();
    std::cout << "  最终缓存命中率: " << (stats.cacheHitRate() * 100) << "%\n";
    
    HetKVCache::destroy(cache);
}

// 示例5：诊断功能演示
void diagnostics_demo() {
    std::cout << "\n=== 示例5：诊断功能演示 ===\n";
    
    HetKVCacheConfig config;
    config.vram_budget_mb = 512;
    config.ram_budget_mb = 2048;
    config.block_size = 16 * 1024;
    config.log_level = 3;  // INFO level
    
    auto cache = HetKVCache::create(config);
    if (!cache) {
        std::cerr << "初始化失败\n";
        return;
    }
    
    // 分配一些KV Cache
    auto handle = cache->allocateKVCache(4, 8, 128, 10);
    
    // 打印状态
    std::cout << "当前状态:\n";
    cache->printStatus();
    
    // 验证内部一致性
    bool valid = cache->validate();
    std::cout << "内部一致性验证: " << (valid ? "通过" : "失败") << "\n";
    
    // 导出诊断信息
    std::string diagnostics = cache->exportDiagnostics();
    std::cout << "诊断信息长度: " << diagnostics.length() << " 字节\n";
    
    // 清理
    cache->deallocateKVCache(handle);
    HetKVCache::destroy(cache);
}

} // namespace hetkvcache

int main() {
    std::cout << "========================================\n";
    std::cout << "    HetKVCache 基础使用示例\n";
    std::cout << "========================================\n";
    
    try {
        hetkvcache::basic_kv_cache_operations();
        hetkvcache::statistics_demo();
        hetkvcache::migration_demo();
        hetkvcache::concurrent_access_demo();
        hetkvcache::diagnostics_demo();
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }
    
    std::cout << "\n========================================\n";
    std::cout << "    所有示例执行完成\n";
    std::cout << "========================================\n";
    
    return 0;
}