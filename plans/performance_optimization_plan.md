# HetKVCache 性能优化计划

## 一、当前性能分析

### 1.1 Benchmark 结果 (昇腾910B4)

| 模块 | 当前性能 | 目标性能 | 瓶颈分析 |
|------|---------|---------|---------|
| AccessMonitor | 998 ops/sec | 500K+ ops/sec | 锁竞争严重 |
| HeatEvaluator | 309K ops/sec | 1M+ ops/sec | 单线程计算 |
| Allocator | 560K ops/sec | 2M+ ops/sec | 锁粒度粗 |

### 1.2 关键瓶颈识别

#### AccessMonitor 瓶颈 (最严重)
```cpp
// 问题代码位置: src/monitor/access_monitor.cpp

// 瓶颈1: 每次访问都加锁 (第88行)
std::lock_guard<std::mutex> lock(queue_mutex_);  // 严重锁竞争
access_queue_.push(record);
queue_cv_.notify_one();  // 每次访问都通知

// 瓶颈2: 单线程处理队列 (第180-206行)
void processQueue() {
    while (running_.load()) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, ...);  // 条件变量等待开销
        // 单线程处理
    }
}

// 瓶颈3: updateCounts使用写锁 (第216行)
std::unique_lock<std::shared_mutex> lock(counts_mutex_);  // 写锁阻塞所有读
```

#### HeatEvaluator 瓶颈
```cpp
// 问题代码位置: src/monitor/heat_evaluator.cpp

// 瓶颈1: updateAllHeatScores串行处理 (第90-100行)
for (BlockId block_id : blocks_to_update) {
    updateBlockHeat(block_id, current_time);  // 逐个更新，无并行
}

// 瓶颈2: 层ID查找效率低 (第66-72行)
for (const auto& [lid, blocks] : layer_blocks_) {
    if (std::find(blocks.begin(), blocks.end(), block_id) != blocks.end()) {
        layer_id = lid;  // O(n) 查找
        break;
    }
}
```

## 二、优化方案

### 2.1 AccessMonitor 无锁化优化 (优先级: 最高)

#### 方案A: 无锁队列 + 批量处理
```cpp
// 使用无锁队列替代互斥锁队列
#include <boost/lockfree/queue.hpp>

class AccessMonitor {
    // 替换 std::queue 为无锁队列
    boost::lockfree::queue<AccessRecord> access_queue_{1024};
    
    void recordAccess(...) {
        // 无锁入队
        access_queue_.push(record);
        // 批量通知，减少条件变量开销
        if (++pending_count_ >= BATCH_SIZE) {
            queue_cv_.notify_one();
        }
    }
};
```

#### 方案B: 线程本地缓存 + 定期刷新
```cpp
class AccessMonitor {
    // 每个线程本地缓存
    static thread_local std::vector<AccessRecord> local_cache_;
    
    void recordAccess(...) {
        local_cache_.push_back(record);
        if (local_cache_.size() >= LOCAL_BATCH_SIZE) {
            flushLocalCache();
        }
    }
    
    void flushLocalCache() {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        for (const auto& r : local_cache_) {
            access_queue_.push(r);
        }
        local_cache_.clear();
    }
};
```

#### 方案C: 原子计数器替代哈希表
```cpp
// 使用原子数组替代哈希表 + 锁
class AccessMonitor {
    // 固定大小的原子计数器数组
    static constexpr size_t MAX_BLOCKS = 1000000;
    std::atomic<uint64_t> block_counts_[MAX_BLOCKS];
    std::atomic<uint64_t> layer_counts_[128];  // 最多128层
    std::atomic<uint64_t> head_counts_[128];   // 最多128头
    
    void updateCounts(const AccessRecord& record) {
        // 无锁更新
        block_counts_[record.block_id].fetch_add(1, std::memory_order_relaxed);
        layer_counts_[record.layer_id].fetch_add(1, std::memory_order_relaxed);
        head_counts_[record.head_id].fetch_add(1, std::memory_order_relaxed);
    }
};
```

### 2.2 HeatEvaluator 并行化优化

#### 方案: OpenMP 并行计算
```cpp
void HeatEvaluator::updateAllHeatScores(Timestamp current_time) {
    std::vector<BlockId> blocks_to_update;
    {
        std::shared_lock<std::shared_mutex> lock(heat_mutex_);
        blocks_to_update.reserve(block_heat_info_.size());
        for (const auto& [block_id, _] : block_heat_info_) {
            blocks_to_update.push_back(block_id);
        }
    }
    
    // 并行计算热度分数
    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < blocks_to_update.size(); i++) {
        updateBlockHeatNoLock(blocks_to_update[i], current_time);
    }
}
```

#### 方案: 层ID快速查找
```cpp
// 使用哈希表替代线性查找
std::unordered_map<BlockId, LayerId> block_to_layer_;

void processAccess(const AccessRecord& record) {
    block_to_layer_[record.block_id] = record.layer_id;  // O(1)
}

LayerId findLayer(BlockId block_id) {
    auto it = block_to_layer_.find(block_id);
    return (it != block_to_layer_.end()) ? it->second : 0;
}
```

### 2.3 Allocator 锁优化

#### 方案: 分片锁
```cpp
class PagedAllocator {
    static constexpr size_t NUM_SHARDS = 16;
    
    struct Shard {
        std::mutex mutex;
        std::queue<BlockId> free_list;
        std::unordered_map<BlockId, BlockMetadata> metadata;
    };
    
    Shard shards_[NUM_SHARDS];
    
    size_t getShard(BlockId block_id) {
        return block_id % NUM_SHARDS;
    }
    
    BlockId allocateBlock() {
        size_t shard = getShard(next_block_id_);
        std::lock_guard<std::mutex> lock(shards_[shard].mutex);
        // 只锁定单个分片
    }
};
```

### 2.4 迁移引擎异步优化

#### 方案: 双缓冲 + 流水线
```cpp
class MigrationEngine {
    // 双缓冲
    std::vector<MigrationTask> task_buffer_a_;
    std::vector<MigrationTask> task_buffer_b_;
    std::atomic<int> active_buffer_{0};
    
    void submitTask(MigrationTask task) {
        auto& buffer = (active_buffer_.load() == 0) ? task_buffer_a_ : task_buffer_b_;
        buffer.push_back(task);
    }
    
    void processTasks() {
        // 切换缓冲区
        int inactive = 1 - active_buffer_.exchange(1 - active_buffer_.load());
        auto& buffer = (inactive == 0) ? task_buffer_a_ : task_buffer_b_;
        
        // 批量处理
        for (auto& task : buffer) {
            executeMigration(task);
        }
        buffer.clear();
    }
};
```

### 2.5 预取策略优化

#### 方案: 基于马尔可夫链的预测
```cpp
class MarkovPrefetcher {
    // 状态转移矩阵
    static constexpr size_t STATE_SIZE = 1024;
    std::atomic<uint32_t> transition_matrix_[STATE_SIZE][STATE_SIZE];
    
    // 训练
    void train(BlockId prev, BlockId current) {
        transition_matrix_[prev % STATE_SIZE][current % STATE_SIZE]
            .fetch_add(1, std::memory_order_relaxed);
    }
    
    // 预测
    std::vector<BlockId> predict(BlockId current, int top_k = 3) {
        uint32_t scores[STATE_SIZE];
        // 找出转移概率最高的k个块
        // ...
    }
};
```

### 2.6 昇腾NPU专用优化

#### 方案: ACL流池化
```cpp
class AscendStreamPool {
    static constexpr size_t POOL_SIZE = 16;
    aclrtStream streams_[POOL_SIZE];
    std::atomic<size_t> next_stream_{0};
    
    aclrtStream getStream() {
        return streams_[next_stream_.fetch_add(1) % POOL_SIZE];
    }
};
```

## 三、实施计划

### 阶段1: 核心瓶颈优化 (优先级最高)
1. AccessMonitor 无锁化
2. 原子计数器替代哈希表

### 阶段2: 并行化优化
3. HeatEvaluator OpenMP并行
4. Allocator 分片锁

### 阶段3: 高级优化
5. 迁移引擎双缓冲
6. 马尔可夫预取

### 阶段4: 硬件特定优化
7. 昇腾NPU流池化
8. 鲲鹏CPU NEON优化

## 四、预期效果

| 优化项 | 预期提升 | 置信度 |
|--------|---------|--------|
| AccessMonitor无锁化 | 500x (998 -> 500K) | 高 |
| HeatEvaluator并行化 | 3x (309K -> 1M) | 高 |
| Allocator分片锁 | 4x (560K -> 2M) | 中 |
| 迁移引擎双缓冲 | 2x | 中 |
| 马尔可夫预取 | 10%命中率提升 | 中 |

## 五、风险评估

1. **无锁算法复杂度** - 需要仔细处理内存序
2. **并行正确性** - 需要验证数据一致性
3. **硬件兼容性** - 需要在CUDA和Ascend双平台测试