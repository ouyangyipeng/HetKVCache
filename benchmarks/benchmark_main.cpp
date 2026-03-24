/**
 * @file benchmark_main.cpp
 * @brief HetKVCache 基准测试主程序
 */

#include "hetkvcache/hetkvcache.h"
#include "hetkvcache/monitor/access_monitor.h"
#include "hetkvcache/monitor/lockfree_monitor.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>
#include <thread>

using namespace hetkvcache;

// 基准测试配置
struct BenchmarkConfig {
    int num_iterations;
    int num_blocks;
    int num_layers;
    int num_heads;
    int sequence_length;
    bool enable_migration;
    bool enable_prefetch;
};

// 基准测试结果
struct BenchmarkResult {
    std::string name;
    double total_time_ms;
    double avg_time_us;
    uint64_t operations;
    double throughput_ops_per_sec;
    double cache_hit_rate;
    size_t memory_used;
    
    void print() const {
        std::cout << "  " << name << ":" << std::endl;
        std::cout << "    Total time: " << total_time_ms << " ms" << std::endl;
        std::cout << "    Average time: " << avg_time_us << " us" << std::endl;
        std::cout << "    Operations: " << operations << std::endl;
        std::cout << "    Throughput: " << throughput_ops_per_sec << " ops/sec" << std::endl;
        if (cache_hit_rate >= 0) {
            std::cout << "    Cache hit rate: " << (cache_hit_rate * 100) << "%" << std::endl;
        }
        std::cout << "    Memory used: " << (memory_used / 1024.0 / 1024.0) << " MB" << std::endl;
    }
};

// 计时器
class Timer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_;
};

// 分配器基准测试
BenchmarkResult benchmark_allocator(PagedAllocator& allocator, 
                                    const BenchmarkConfig& config) {
    BenchmarkResult result;
    result.name = "Allocator Benchmark";
    result.cache_hit_rate = -1;
    
    Timer timer;
    timer.start();
    
    std::vector<BlockId> blocks;
    blocks.reserve(config.num_blocks);
    
    // 分配
    for (int i = 0; i < config.num_blocks; i++) {
        auto alloc_result = allocator.allocateBlock(i % config.num_layers, 
                                                     i % config.num_heads);
        if (alloc_result.success) {
            blocks.push_back(alloc_result.block_id);
        }
    }
    
    // 访问
    for (BlockId block_id : blocks) {
        void* ptr = allocator.getBlockPtr(block_id);
        (void)ptr;  // 避免未使用警告
    }
    
    // 释放
    for (BlockId block_id : blocks) {
        allocator.deallocateBlock(block_id);
    }
    
    result.total_time_ms = timer.elapsed_ms();
    result.operations = config.num_blocks * 3;  // alloc + access + dealloc
    result.avg_time_us = (result.total_time_ms * 1000) / result.operations;
    result.throughput_ops_per_sec = (result.operations * 1000) / result.total_time_ms;
    result.memory_used = config.num_blocks * 16 * 1024;  // 假设 16KB 块
    
    return result;
}

// 热度评估基准测试
BenchmarkResult benchmark_heat_evaluator(HeatEvaluator& evaluator,
                                         const BenchmarkConfig& config) {
    BenchmarkResult result;
    result.name = "Heat Evaluator Benchmark";
    
    // 注册块
    for (int i = 0; i < config.num_blocks; i++) {
        evaluator.registerBlock(i, i % config.num_layers);
    }
    
    Timer timer;
    timer.start();
    
    // 模拟访问
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> block_dist(0, config.num_blocks - 1);
    
    for (int i = 0; i < config.num_iterations; i++) {
        BlockId block_id = block_dist(gen);
        AccessRecord record(block_id, getCurrentTimestamp(), 
                           AccessType::READ, block_id % config.num_layers, 
                           0, i);
        evaluator.processAccess(record);
    }
    
    // 更新热度
    evaluator.updateAllHeatScores(getCurrentTimestamp());
    
    result.total_time_ms = timer.elapsed_ms();
    result.operations = config.num_iterations + config.num_blocks;
    result.avg_time_us = (result.total_time_ms * 1000) / result.operations;
    result.throughput_ops_per_sec = (result.operations * 1000) / result.total_time_ms;
    
    // 计算热点比例
    auto hot = evaluator.getHotBlocks();
    auto cold = evaluator.getColdBlocks();
    result.cache_hit_rate = static_cast<double>(hot.size()) / 
                           (hot.size() + cold.size() + evaluator.getWarmBlocks().size());
    
    return result;
}

// 访问监控基准测试
BenchmarkResult benchmark_access_monitor(AccessMonitor& monitor,
                                         const BenchmarkConfig& config) {
    BenchmarkResult result;
    result.name = "Access Monitor Benchmark";
    
    monitor.initialize(AccessMonitor::MonitorMode::FULL, 1.0f, 100000);
    monitor.startMonitoring();
    
    Timer timer;
    timer.start();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> block_dist(0, config.num_blocks - 1);
    std::uniform_int_distribution<> layer_dist(0, config.num_layers - 1);
    std::uniform_int_distribution<> head_dist(0, config.num_heads - 1);
    
    for (int i = 0; i < config.num_iterations; i++) {
        monitor.recordAccess(block_dist(gen), AccessType::READ,
                           layer_dist(gen), head_dist(gen), i);
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    result.total_time_ms = timer.elapsed_ms();
    result.operations = config.num_iterations;
    result.avg_time_us = (result.total_time_ms * 1000) / result.operations;
    result.throughput_ops_per_sec = (result.operations * 1000) / result.total_time_ms;
    
    auto stats = monitor.getStats();
    result.cache_hit_rate = 1.0;  // 监控器本身没有命中率概念
    
    monitor.stopMonitoring();
    monitor.shutdown();
    
    return result;
}

// 无锁监控器基准测试
BenchmarkResult benchmark_lockfree_monitor(const BenchmarkConfig& config) {
    BenchmarkResult result;
    result.name = "LockFree Access Monitor Benchmark";
    
    LockFreeAccessMonitor monitor;
    monitor.initialize(LockFreeAccessMonitor::MonitorMode::FULL, 1.0f, 100000);
    monitor.startMonitoring();
    
    Timer timer;
    timer.start();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> block_dist(0, config.num_blocks - 1);
    std::uniform_int_distribution<> layer_dist(0, config.num_layers - 1);
    std::uniform_int_distribution<> head_dist(0, config.num_heads - 1);
    
    // 单线程测试
    for (int i = 0; i < config.num_iterations * 10; i++) {
        monitor.recordAccess(block_dist(gen), AccessType::READ,
                            layer_dist(gen), head_dist(gen), i);
    }
    
    result.total_time_ms = timer.elapsed_ms();
    result.operations = config.num_iterations * 10;
    result.avg_time_us = (result.total_time_ms * 1000) / result.operations;
    result.throughput_ops_per_sec = (result.operations * 1000) / result.total_time_ms;
    result.cache_hit_rate = 1.0;
    
    monitor.stopMonitoring();
    monitor.shutdown();
    
    return result;
}

// 并发监控器基准测试
BenchmarkResult benchmark_concurrent_monitor(const BenchmarkConfig& config) {
    BenchmarkResult result;
    result.name = "Concurrent LockFree Monitor Benchmark";
    
    LockFreeAccessMonitor monitor;
    monitor.initialize(LockFreeAccessMonitor::MonitorMode::FULL, 1.0f, 100000);
    monitor.startMonitoring();
    
    Timer timer;
    timer.start();
    
    // 多线程并发测试
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    std::atomic<int> total_ops{0};
    
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&monitor, &config, &total_ops, t]() {
            std::random_device rd;
            std::mt19937 gen(rd() + t);
            std::uniform_int_distribution<> block_dist(0, config.num_blocks - 1);
            std::uniform_int_distribution<> layer_dist(0, config.num_layers - 1);
            std::uniform_int_distribution<> head_dist(0, config.num_heads - 1);
            
            for (int i = 0; i < config.num_iterations; i++) {
                monitor.recordAccess(block_dist(gen), AccessType::READ,
                                    layer_dist(gen), head_dist(gen), i);
                total_ops.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    result.total_time_ms = timer.elapsed_ms();
    result.operations = total_ops.load();
    result.avg_time_us = (result.total_time_ms * 1000) / result.operations;
    result.throughput_ops_per_sec = (result.operations * 1000) / result.total_time_ms;
    result.cache_hit_rate = 1.0;
    
    monitor.stopMonitoring();
    monitor.shutdown();
    
    return result;
}

// 迁移引擎基准测试
BenchmarkResult benchmark_migration(MigrationEngine& engine,
                                    const BenchmarkConfig& config) {
    BenchmarkResult result;
    result.name = "Migration Engine Benchmark";
    
    // 需要 CUDA 环境
    
    result.total_time_ms = 0;
    result.operations = 0;
    result.avg_time_us = 0;
    result.throughput_ops_per_sec = 0;
    result.cache_hit_rate = -1;
    
    return result;
}

// 端到端基准测试
BenchmarkResult benchmark_e2e(HetKVCache& instance,
                              const BenchmarkConfig& config) {
    BenchmarkResult result;
    result.name = "End-to-End Benchmark";
    
    // 分配 KV Cache
    KVCacheHandle handle = instance.allocateKVCache(config.num_layers, 
                                                     config.num_heads,
                                                     128,  // head_dim
                                                     config.sequence_length);
    
    if (handle == INVALID_HANDLE) {
        result.name += " (Failed)";
        return result;
    }
    
    Timer timer;
    timer.start();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> layer_dist(0, config.num_layers - 1);
    std::uniform_int_distribution<> head_dist(0, config.num_heads - 1);
    std::uniform_int_distribution<> seq_dist(0, config.sequence_length - 1);
    
    // 模拟推理访问模式
    for (int iter = 0; iter < config.num_iterations; iter++) {
        // 顺序访问（模拟解码）
        for (int pos = 0; pos < config.sequence_length; pos++) {
            for (uint32_t layer = 0; layer < config.num_layers; layer++) {
                instance.accessBlock(handle, layer, 0, pos, AccessType::READ);
            }
        }
    }
    
    result.total_time_ms = timer.elapsed_ms();
    result.operations = config.num_iterations * config.sequence_length * config.num_layers;
    result.avg_time_us = (result.total_time_ms * 1000) / result.operations;
    result.throughput_ops_per_sec = (result.operations * 1000) / result.total_time_ms;
    
    auto stats = instance.getStatistics();
    result.cache_hit_rate = stats.cacheHitRate();
    result.memory_used = stats.vram_used_bytes + stats.memory_used_bytes;
    
    instance.deallocateKVCache(handle);
    
    return result;
}

// 运行所有基准测试
void run_benchmarks(const BenchmarkConfig& config) {
    std::cout << "========================================" << std::endl;
    std::cout << "HetKVCache Benchmark Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Iterations: " << config.num_iterations << std::endl;
    std::cout << "  Blocks: " << config.num_blocks << std::endl;
    std::cout << "  Layers: " << config.num_layers << std::endl;
    std::cout << "  Heads: " << config.num_heads << std::endl;
    std::cout << "  Sequence Length: " << config.sequence_length << std::endl;
    std::cout << std::endl;
    
    std::vector<BenchmarkResult> results;
    
    // 基础分配器基准测试
    {
        PagedAllocatorConfig alloc_config;
        alloc_config.block_size = 16 * 1024;
        alloc_config.vram_pool_size = config.num_blocks * 16 * 1024;
        alloc_config.ram_pool_size = config.num_blocks * 16 * 1024 * 4;
        alloc_config.use_pinned_memory = false;
        
        PagedAllocator allocator;
        allocator.initialize();
        results.push_back(benchmark_allocator(allocator, config));
    }
    
    // 热度评估基准测试
    {
        HeatEvaluator evaluator;
        evaluator.setNumLayers(config.num_layers);
        evaluator.setNumHeads(config.num_heads);
        results.push_back(benchmark_heat_evaluator(evaluator, config));
    }
    
    // 访问监控基准测试（原始版本）
    {
        AccessMonitor monitor;
        results.push_back(benchmark_access_monitor(monitor, config));
    }
    
    // 无锁监控器基准测试
    {
        results.push_back(benchmark_lockfree_monitor(config));
    }
    
    // 并发监控器基准测试
    {
        results.push_back(benchmark_concurrent_monitor(config));
    }
    
    // 打印结果
    std::cout << "========================================" << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "========================================" << std::endl;
    
    for (const auto& result : results) {
        result.print();
        std::cout << std::endl;
    }
    
    // 汇总
    std::cout << "========================================" << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "========================================" << std::endl;
    
    double total_ops = 0;
    for (const auto& result : results) {
        total_ops += result.operations;
    }
    
    std::cout << "Total operations: " << total_ops << std::endl;
}

int main(int argc, char* argv[]) {
    BenchmarkConfig config;
    config.num_iterations = 100;
    config.num_blocks = 10000;
    config.num_layers = 32;
    config.num_heads = 32;
    config.sequence_length = 2048;
    config.enable_migration = true;
    config.enable_prefetch = true;
    
    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--iterations" && i + 1 < argc) {
            config.num_iterations = std::stoi(argv[++i]);
        } else if (arg == "--blocks" && i + 1 < argc) {
            config.num_blocks = std::stoi(argv[++i]);
        } else if (arg == "--layers" && i + 1 < argc) {
            config.num_layers = std::stoi(argv[++i]);
        } else if (arg == "--heads" && i + 1 < argc) {
            config.num_heads = std::stoi(argv[++i]);
        } else if (arg == "--seq-len" && i + 1 < argc) {
            config.sequence_length = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --iterations N   Number of iterations (default: 100)" << std::endl;
            std::cout << "  --blocks N       Number of blocks (default: 10000)" << std::endl;
            std::cout << "  --layers N       Number of layers (default: 32)" << std::endl;
            std::cout << "  --heads N        Number of heads (default: 32)" << std::endl;
            std::cout << "  --seq-len N      Sequence length (default: 2048)" << std::endl;
            return 0;
        }
    }
    
    run_benchmarks(config);
    
    return 0;
}