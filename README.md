# HetKVCache - 异构存储KV缓存管理系统

<div align="center">

**面向LLM推理的异构存储KV缓存优化系统**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-green.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

</div>

## 📖 项目简介

HetKVCache（Heterogeneous KV Cache）是专为大规模语言模型（LLM）推理设计的异构存储管理系统。通过智能化的多级缓存策略，实现GPU显存、主机内存和NVMe SSD之间的高效数据迁移，显著提升推理吞吐量。

### 核心特性

- **🔍 智能监控**：实时追踪KV缓存块的访问模式，精确评估数据热度
- **⚡ 动态迁移**：基于热度评分的异步数据迁移，最小化推理延迟
- **📦 无碎片分配**：采用PagedAttention风格的块分配策略，消除内存碎片
- **🚀 预取优化**：基于马尔可夫链的访问预测，提前加载热点数据
- **🔧 透明集成**：提供C/C++和Python API，无缝对接vLLM和HuggingFace

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      LLM Inference Engine                        │
│                    (vLLM / HuggingFace)                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      HetKVCache API Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  C/C++ API  │  │ Python API  │  │    Integration Hooks    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Core Modules                                │
├────────────────┬────────────────┬───────────────────────────────┤
│   Monitor &    │   Migration    │      Paged Allocator          │
│   Profiler     │    Engine      │                               │
│                │                │                               │
│ • Access Trace │ • Async Transfer│ • Block Pool Management      │
│ • Heat Score   │ • CUDA Streams │ • Tier-aware Allocation      │
│ • Pattern Ana. │ • Prefetching  │ • Zero Fragmentation          │
└────────────────┴────────────────┴───────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Storage Hierarchy                              │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   GPU VRAM      │   Host RAM      │       NVMe SSD               │
│   (Hot Tier)    │   (Warm Tier)   │      (Cold Tier)             │
│                 │                 │                              │
│  • < 1ms access │  • 1-10ms access│  • 0.1-1ms access           │
│  • High bandwidth│ • Pinned mem   │  • Large capacity            │
│  • Limited size │  • Medium size  │  • Persistent storage        │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

## 🚀 快速开始

### 环境要求

- **操作系统**：Linux (Ubuntu 20.04+ 推荐)
- **编译器**：GCC 9+ 或 Clang 10+
- **CUDA**：11.0 或更高版本
- **CMake**：3.18+
- **GPU**：NVIDIA GPU (计算能力 7.0+)

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/your-repo/HetKVCache.git
cd HetKVCache

# 2. 创建构建目录
mkdir build && cd build

# 3. 配置和编译
cmake ..
make -j$(nproc)

# 4. 安装 (可选)
sudo make install
```

### 快速示例

```cpp
#include <hetkvcache/hetkvcache.h>

int main() {
    // 创建配置
    HetKVCache::Config config;
    config.vram_capacity = 16ULL * 1024 * 1024 * 1024;  // 16GB VRAM
    config.ram_capacity = 64ULL * 1024 * 1024 * 1024;    // 64GB RAM
    config.block_size = 16 * 1024;                        // 16KB blocks
    
    // 初始化缓存管理器
    auto cache = HetKVCache::KVCacheManager::create(config);
    
    // 分配缓存块
    auto block_id = cache->allocate_block(1024, 32);  // layer 0, 32 blocks
    
    // 写入数据
    float* data = ...; // 你的KV缓存数据
    cache->write_block(block_id, data, data_size);
    
    // 读取数据 (自动触发迁移如果数据不在VRAM)
    cache->read_block(block_id, data);
    
    // 释放块
    cache->release_block(block_id);
    
    return 0;
}
```

## 📚 详细文档

### 模块说明

#### 1. 监控与热度评估器 (Monitor & Profiler)

监控模块负责追踪所有KV缓存块的访问模式，并计算热度评分。

**热度评分算法**：
```
heat_score = α × frequency + β × recency + γ × layer_importance
```

其中：
- `frequency`: 访问频率（指数衰减）
- `recency`: 最近访问时间
- `layer_importance`: 层级重要性权重

**分层策略**：
| 热度等级 | 分数范围 | 存储位置 | 访问延迟 |
|---------|---------|---------|---------|
| Hot | ≥ 0.7 | GPU VRAM | < 1ms |
| Warm | 0.3 - 0.7 | Host RAM | 1-10ms |
| Cold | < 0.3 | NVMe SSD | 0.1-1s |

#### 2. 迁移引擎 (Migration Engine)

迁移引擎负责在不同存储层级之间异步移动数据。

**核心特性**：
- 基于CUDA Streams的异步传输
- 优先级队列管理迁移任务
- 智能预取减少等待延迟

**迁移路径**：
```
VRAM (Hot) ←→ RAM (Warm) ←→ SSD (Cold)
     ↑              ↑             ↑
     └──────────────┴─────────────┘
            异步迁移引擎
```

#### 3. 无碎片分配器 (Paged Allocator)

采用PagedAttention风格的块分配策略，完全消除内存碎片。

**设计要点**：
- 固定16KB块大小
- 块池预分配
- 引用计数管理
- 层级感知分配

**内存布局**：
```
┌─────────────────────────────────────────┐
│              Block Pool                  │
├─────────────────────────────────────────┤
│ Block 0 │ Block 1 │ Block 2 │ ... │ Block N │
│  16KB   │  16KB   │  16KB   │     │  16KB   │
└─────────────────────────────────────────┘
```

### 配置选项

```cpp
struct Config {
    // 容量配置
    size_t vram_capacity;        // GPU显存容量
    size_t ram_capacity;         // 主机内存容量
    size_t ssd_capacity;         // SSD容量（0表示无限制）
    
    // 块配置
    size_t block_size;           // 块大小（默认16KB）
    size_t max_blocks_per_layer; // 每层最大块数
    
    // 热度评估参数
    float alpha;                  // 频率权重（默认0.5）
    float beta;                   // 新近度权重（默认0.3）
    float gamma;                  // 层级权重（默认0.2）
    
    // 迁移参数
    float hot_threshold;          // 热数据阈值（默认0.7）
    float cold_threshold;         // 冷数据阈值（默认0.3）
    size_t migration_batch_size;  // 迁移批次大小
    
    // 预取参数
    bool enable_prefetch;         // 启用预取
    size_t prefetch_window;       // 预取窗口大小
};
```

### API 参考

#### C/C++ API

```cpp
namespace HetKVCache {

class KVCacheManager {
public:
    // 创建管理器实例
    static std::unique_ptr<KVCacheManager> create(const Config& config);
    
    // 块管理
    BlockId allocate_block(uint32_t layer, uint32_t num_blocks);
    void release_block(BlockId block_id);
    
    // 数据访问
    void read_block(BlockId block_id, void* buffer);
    void write_block(BlockId block_id, const void* data, size_t size);
    
    // 迁移控制
    void migrate_to_vram(BlockId block_id);
    void migrate_to_ram(BlockId block_id);
    void migrate_to_ssd(BlockId block_id);
    
    // 监控接口
    HeatScore get_heat_score(BlockId block_id);
    TierType get_current_tier(BlockId block_id);
    Statistics get_statistics();
};

}  // namespace HetKVCache
```

#### Python API

```python
import hetkvcache

# 创建管理器
config = hetkvcache.Config(
    vram_capacity=16 * 1024**3,  # 16GB
    ram_capacity=64 * 1024**3,   # 64GB
)
cache = hetkvcache.KVCacheManager(config)

# 分配和访问
block_id = cache.allocate_block(layer=0, num_blocks=32)
cache.write_block(block_id, data)
cache.read_block(block_id, buffer)
cache.release_block(block_id)
```

## 🧪 测试与基准测试

### 运行单元测试

```bash
cd build
./tests/hetkvcache_tests
```

### 运行基准测试

```bash
# 一键运行完整基准测试
./scripts/run_benchmark.sh

# 或手动运行
cd build
./benchmarks/hetkvcache_benchmark --scenario=long_seq --duration=60
```

### 基准测试场景

| 场景 | 序列长度 | 批次大小 | 描述 |
|------|---------|---------|------|
| short_seq | 512 | 32 | 短序列，高并发 |
| medium_seq | 2048 | 16 | 中等序列 |
| long_seq | 8192 | 8 | 长序列，高显存压力 |
| ultra_long | 32768 | 2 | 超长序列，极端场景 |

## 📊 性能指标

### 目标性能

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 吞吐量提升 | ≥ 30% | 相比无优化基线 |
| 缓存命中率 | ≥ 90% | 热数据命中率 |
| 迁移开销 | < 5% | 迁移时间占总时间比 |
| 内存碎片 | 0% | 完全无碎片 |

### 性能调优建议

1. **调整块大小**：根据模型hidden_size调整，推荐设置为`hidden_size * num_heads * sizeof(float16)`
2. **热度阈值**：根据工作负载调整，高并发场景可降低hot_threshold
3. **预取窗口**：长序列场景增大prefetch_window
4. **批次大小**：根据GPU显存和迁移带宽调整migration_batch_size

## 🔧 集成指南

### 与vLLM集成

```python
from vllm import LLM
from hetkvcache import vLLMIntegration

# 创建HetKVCache后端
cache_backend = vLLMIntegration.create_backend(
    vram_capacity=16 * 1024**3,
    ram_capacity=64 * 1024**3,
)

# 配置vLLM使用HetKVCache
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    kv_cache_backend=cache_backend,
)
```

### 与HuggingFace集成

```python
from transformers import AutoModelForCausalLM
from hetkvcache import HuggingFaceIntegration

# 加载模型并启用HetKVCache
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map="auto",
)
model = HuggingFaceIntegration.wrap_model(model)
```

## 📁 项目结构

```
HetKVCache/
├── CMakeLists.txt              # 构建配置
├── README.md                   # 本文档
├── include/hetkvcache/         # 公共头文件
│   ├── types.h                 # 类型定义
│   ├── config.h                # 配置结构
│   ├── hetkvcache.h            # 主API
│   ├── monitor/                # 监控模块头文件
│   ├── migration/              # 迁移模块头文件
│   ├── allocator/              # 分配器头文件
│   └── utils/                  # 工具头文件
├── src/                        # 源文件
│   ├── core/                   # 核心实现
│   ├── monitor/                # 监控实现
│   ├── migration/              # 迁移实现
│   ├── allocator/             # 分配器实现
│   ├── cuda/                   # CUDA内核
│   └── utils/                  # 工具实现
├── tests/                      # 单元测试
├── benchmarks/                 # 基准测试
├── examples/                   # 示例代码
│   ├── basic_usage/            # 基础用法
│   ├── vllm_integration/       # vLLM集成
│   └── huggingface_integration/ # HuggingFace集成
└── scripts/                    # 脚本
    └── run_benchmark.sh        # 自动化测试脚本
```

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

### 代码规范

- C++代码遵循[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- 使用clang-format格式化代码
- 确保所有测试通过

## 📄 许可证

本项目采用Apache 2.0许可证 - 详见[LICENSE](LICENSE)文件。

## 🙏 致谢

本项目为2026年全国大学生计算机系统能力大赛OS功能挑战赛道参赛作品。

感谢以下开源项目的启发：
- [vLLM](https://github.com/vllm-project/vllm) - PagedAttention设计理念
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - 模型生态
- [FlashAttention](https://github.com/dao-scu-lab/flash-attention) - 高效注意力实现

## 📧 联系方式

- 项目主页：https://github.com/your-repo/HetKVCache
- 问题反馈：https://github.com/your-repo/HetKVCache/issues
- 邮箱：your-team@example.com

---

<div align="center">
Made with ❤️ for the 2026 National Student Computer System Capability Competition
</div>