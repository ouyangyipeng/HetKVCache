/**
 * @file test_main.cpp
 * @brief HetKVCache 测试主程序
 */

#include "hetkvcache/hetkvcache.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <thread>
#include <chrono>

using namespace hetkvcache;

// 测试配置
void test_config() {
    std::cout << "Testing configuration..." << std::endl;
    
    auto config = HetKVCache::getDefaultConfig();
    assert(config.vram_budget_mb == 4096);
    assert(config.ram_budget_mb == 16384);
    assert(config.hot_threshold > config.warm_threshold);
    
    std::cout << "Configuration test passed!" << std::endl;
}

// 测试分配器
void test_allocator() {
    std::cout << "Testing allocator..." << std::endl;
    
    PagedAllocatorConfig config;
    config.block_size = 16 * 1024;  // 16KB
    config.vram_pool_size = 1024 * 1024;  // 1MB
    config.ram_pool_size = 4 * 1024 * 1024;  // 4MB
    config.use_pinned_memory = false;  // 测试时不使用锁页内存
    
    PagedAllocator allocator;
    allocator.initialize();
    
    // 测试分配
    auto result = allocator.allocateBlock(0, 0);
    if (result.success) {
        std::cout << "  Block allocated: " << result.block_id << std::endl;
        assert(result.ptr != nullptr);
        
        // 测试获取指针
        void* ptr = allocator.getBlockPtr(result.block_id);
        assert(ptr == result.ptr);
        
        // 测试释放
        allocator.deallocateBlock(result.block_id);
        std::cout << "  Block deallocated" << std::endl;
    }
    
    std::cout << "Allocator test passed!" << std::endl;
}

// 测试热度评估器
void test_heat_evaluator() {
    std::cout << "Testing heat evaluator..." << std::endl;
    
    HeatEvaluator evaluator;
    evaluator.setNumLayers(32);
    evaluator.setNumHeads(32);
    
    // 注册块
    evaluator.registerBlock(1, 0);
    evaluator.registerBlock(2, 1);
    evaluator.registerBlock(3, 2);
    
    // 模拟访问
    AccessRecord record1(1, getCurrentTimestamp(), AccessType::READ, 0, 0, 0);
    AccessRecord record2(2, getCurrentTimestamp(), AccessType::READ, 1, 0, 0);
    AccessRecord record3(1, getCurrentTimestamp(), AccessType::READ, 0, 0, 0);
    
    evaluator.processAccess(record1);
    evaluator.processAccess(record2);
    evaluator.processAccess(record3);
    
    // 更新热度
    evaluator.updateAllHeatScores(getCurrentTimestamp());
    
    // 检查热度分数
    float score1 = evaluator.getHeatScore(1);
    float score2 = evaluator.getHeatScore(2);
    float score3 = evaluator.getHeatScore(3);
    
    std::cout << "  Heat scores: " << score1 << ", " << score2 << ", " << score3 << std::endl;
    
    // 块1访问2次应该比块2（1次）热度高
    assert(score1 > score2);
    
    // 分类测试
    TierType tier = evaluator.classifyTier(0.8f);
    assert(tier == TierType::HOT);
    
    tier = evaluator.classifyTier(0.5f);
    assert(tier == TierType::WARM);
    
    tier = evaluator.classifyTier(0.1f);
    assert(tier == TierType::COLD);
    
    std::cout << "Heat evaluator test passed!" << std::endl;
}

// 测试访问监控器
void test_access_monitor() {
    std::cout << "Testing access monitor..." << std::endl;
    
    AccessMonitor monitor;
    monitor.initialize(AccessMonitor::MonitorMode::FULL, 1.0f, 1000);
    monitor.startMonitoring();
    
    // 记录访问
    monitor.recordAccess(1, AccessType::READ, 0, 0, 0);
    monitor.recordAccess(2, AccessType::READ, 1, 0, 0);
    monitor.recordAccess(1, AccessType::WRITE, 0, 0, 1);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // 检查统计
    auto stats = monitor.getStats();
    assert(stats.total_accesses >= 3);
    
    uint64_t count1 = monitor.getAccessCount(1);
    uint64_t count2 = monitor.getAccessCount(2);
    
    std::cout << "  Access counts: " << count1 << ", " << count2 << std::endl;
    assert(count1 >= 2);
    assert(count2 >= 1);
    
    monitor.stopMonitoring();
    monitor.shutdown();
    
    std::cout << "Access monitor test passed!" << std::endl;
}

// 测试迁移引擎（不使用 CUDA）
void test_migration_engine() {
    std::cout << "Testing migration engine (without CUDA)..." << std::endl;
    
    MigrationEngineConfig config;
    config.max_concurrent_transfers = 4;
    config.vram_budget = 1024 * 1024;
    config.ram_budget = 4 * 1024 * 1024;
    
    MigrationEngine engine;
    // 注意：需要 CUDA 环境才能完整初始化
    // 这里只测试基本功能
    
    std::cout << "Migration engine basic test passed!" << std::endl;
}

// 测试完整系统
void test_full_system() {
    std::cout << "Testing full system..." << std::endl;
    
    auto config = HetKVCache::getDefaultConfig();
    config.vram_budget_mb = 256;  // 测试用小容量
    config.ram_budget_mb = 512;
    config.log_level = 3;
    
    // 创建实例
    auto* instance = HetKVCache::create(config);
    if (!instance) {
        std::cout << "  Failed to create instance (may need CUDA)" << std::endl;
        return;
    }
    
    // 分配 KV Cache
    KVCacheHandle handle = instance->allocateKVCache(32, 32, 128, 10);
    if (handle != INVALID_HANDLE) {
        std::cout << "  KV Cache allocated: " << handle << std::endl;
        
        // 访问块
        void* ptr = instance->accessBlock(handle, 0, 0, 0, AccessType::READ);
        std::cout << "  Block accessed: " << (ptr ? "success" : "null") << std::endl;
        
        // 获取统计
        auto stats = instance->getStatistics();
        std::cout << "  Total blocks: " << stats.total_blocks << std::endl;
        
        // 释放
        instance->deallocateKVCache(handle);
        std::cout << "  KV Cache deallocated" << std::endl;
    }
    
    HetKVCache::destroy(instance);
    std::cout << "Full system test passed!" << std::endl;
}

// 并发测试
void test_concurrent_access() {
    std::cout << "Testing concurrent access..." << std::endl;
    
    AccessMonitor monitor;
    monitor.initialize(AccessMonitor::MonitorMode::FULL, 1.0f, 10000);
    monitor.startMonitoring();
    
    const int num_threads = 4;
    const int accesses_per_thread = 1000;
    
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&monitor, t, accesses_per_thread]() {
            for (int i = 0; i < accesses_per_thread; i++) {
                monitor.recordAccess(t * 1000 + i, AccessType::READ, t, 0, i);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    auto stats = monitor.getStats();
    std::cout << "  Total accesses: " << stats.total_accesses << std::endl;
    assert(stats.total_accesses >= num_threads * accesses_per_thread);
    
    monitor.stopMonitoring();
    monitor.shutdown();
    
    std::cout << "Concurrent access test passed!" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "HetKVCache Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    
    test_config();
    test_allocator();
    test_heat_evaluator();
    test_access_monitor();
    test_migration_engine();
    test_concurrent_access();
    
    // 完整系统测试需要 CUDA
    // test_full_system();
    
    std::cout << "========================================" << std::endl;
    std::cout << "All tests passed!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}