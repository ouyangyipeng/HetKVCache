#!/usr/bin/env python3
"""
HetKVCache 端到端LLM推理测试脚本
使用Qwen模型在昇腾NPU上进行推理测试
"""

import os
import sys
import time
import json
import argparse
import statistics
from datetime import datetime
from typing import List, Dict, Any

# 设置昇腾环境
os.environ['ASCEND_DEVICE_ID'] = '0'

try:
    import torch
    import torch_npu  # 昇腾PyTorch扩展
    from torch_npu.contrib import transfer_to_npu
    print(f"PyTorch version: {torch.__version__}")
    print(f"torch_npu available: True")
except ImportError as e:
    print(f"Warning: torch_npu not available: {e}")
    print("Falling back to CPU mode...")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available")


class LLMInferenceBenchmark:
    """LLM推理性能基准测试"""
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-0.5B", device: str = "npu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """加载模型"""
        print(f"\nLoading model from {self.model_path}...")
        
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library not available")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # 加载模型 - 不使用device_map，手动设置设备
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        if self.device == "npu" and torch.npu.is_available():
            self.model = self.model.to("npu:0")
            print(f"Model loaded on NPU:0")
        else:
            self.model = self.model.to("cpu")
            print(f"Model loaded on CPU")
        
        self.model.eval()
        print("Model loaded successfully!")
        
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """生成文本并记录性能指标"""
        
        # 编码输入
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_length = input_ids.shape[1]
        
        # 移动到设备
        if self.device == "npu" and torch.npu.is_available():
            input_ids = input_ids.npu()
        
        # 记录开始时间
        start_time = time.perf_counter()
        
        # 生成 - 使用更简单的参数避免昇腾兼容性问题
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            except RuntimeError as e:
                # 如果NPU生成失败，回退到CPU
                print(f"NPU generation failed, falling back to CPU: {e}")
                self.model = self.model.cpu()
                input_ids = input_ids.cpu()
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                self.model = self.model.to("npu:0")
        
        # 记录结束时间
        end_time = time.perf_counter()
        
        # 解码输出
        generated_text = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )
        
        # 计算性能指标
        total_time = end_time - start_time
        output_length = outputs.shape[1] - input_length
        throughput = output_length / total_time if total_time > 0 else 0
        
        return {
            "generated_text": generated_text,
            "input_length": input_length,
            "output_length": output_length,
            "total_time": total_time,
            "throughput": throughput,
            "time_to_first_token": total_time / output_length if output_length > 0 else 0,
        }
    
    def benchmark(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        warmup: int = 2,
    ) -> Dict[str, Any]:
        """运行基准测试"""
        
        results = []
        throughputs = []
        latencies = []
        
        print(f"\n{'='*60}")
        print(f"Running benchmark with {len(prompts)} prompts")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"Warmup iterations: {warmup}")
        print(f"{'='*60}\n")
        
        # 预热
        print("Warming up...")
        for i in range(warmup):
            _ = self.generate(prompts[i % len(prompts)], max_new_tokens=max_new_tokens)
            print(f"  Warmup {i+1}/{warmup} done")
        
        # 正式测试
        print("\nRunning benchmark...")
        for i, prompt in enumerate(prompts):
            result = self.generate(prompt, max_new_tokens=max_new_tokens)
            results.append(result)
            throughputs.append(result["throughput"])
            latencies.append(result["total_time"])
            
            print(f"  [{i+1}/{len(prompts)}] "
                  f"Throughput: {result['throughput']:.2f} tokens/s, "
                  f"Latency: {result['total_time']*1000:.2f} ms")
        
        # 计算统计数据
        stats = {
            "num_requests": len(prompts),
            "max_new_tokens": max_new_tokens,
            "throughput": {
                "mean": statistics.mean(throughputs),
                "median": statistics.median(throughputs),
                "stdev": statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
                "min": min(throughputs),
                "max": max(throughputs),
            },
            "latency": {
                "mean": statistics.mean(latencies) * 1000,  # ms
                "median": statistics.median(latencies) * 1000,
                "stdev": statistics.stdev(latencies) * 1000 if len(latencies) > 1 else 0,
                "min": min(latencies) * 1000,
                "max": max(latencies) * 1000,
                "p50": sorted(latencies)[len(latencies)//2] * 1000 if latencies else 0,
                "p99": sorted(latencies)[int(len(latencies)*0.99)] * 1000 if len(latencies) > 1 else 0,
            },
            "total_tokens": sum(r["output_length"] for r in results),
            "total_time": sum(latencies),
        }
        
        return stats
    
    def print_stats(self, stats: Dict[str, Any]):
        """打印统计信息"""
        print(f"\n{'='*60}")
        print("Benchmark Results")
        print(f"{'='*60}")
        print(f"Number of requests: {stats['num_requests']}")
        print(f"Max new tokens: {stats['max_new_tokens']}")
        print(f"Total tokens generated: {stats['total_tokens']}")
        print(f"Total time: {stats['total_time']:.2f} s")
        print(f"\nThroughput (tokens/s):")
        print(f"  Mean:   {stats['throughput']['mean']:.2f}")
        print(f"  Median: {stats['throughput']['median']:.2f}")
        print(f"  StdDev: {stats['throughput']['stdev']:.2f}")
        print(f"  Min:    {stats['throughput']['min']:.2f}")
        print(f"  Max:    {stats['throughput']['max']:.2f}")
        print(f"\nLatency (ms):")
        print(f"  Mean:   {stats['latency']['mean']:.2f}")
        print(f"  Median: {stats['latency']['median']:.2f}")
        print(f"  P50:    {stats['latency']['p50']:.2f}")
        print(f"  P99:    {stats['latency']['p99']:.2f}")
        print(f"  StdDev: {stats['latency']['stdev']:.2f}")
        print(f"{'='*60}\n")


def generate_test_prompts(num_prompts: int = 10, seq_length: str = "medium") -> List[str]:
    """生成测试提示"""
    
    short_prompts = [
        "你好",
        "介绍一下自己",
        "什么是AI？",
        "写一首诗",
        "1+1等于几？",
    ]
    
    medium_prompts = [
        "请详细介绍一下大语言模型的发展历史和主要技术突破。",
        "解释一下Transformer架构中的自注意力机制是如何工作的。",
        "请写一个Python函数，实现快速排序算法，并解释其时间复杂度。",
        "分析一下当前人工智能领域的主要挑战和未来发展方向。",
        "请用通俗易懂的语言解释什么是深度学习，它和传统机器学习有什么区别？",
        "写一篇关于气候变化对全球经济影响的短文，大约200字。",
        "请解释一下KV Cache在LLM推理中的作用，以及为什么需要优化它。",
        "设计一个简单的神经网络架构，用于图像分类任务。",
        "比较一下GPT和BERT模型的主要区别和应用场景。",
        "请写一段代码，使用PyTorch实现一个简单的线性回归模型。",
    ]
    
    long_prompts = [
        "请详细分析一下当前大语言模型推理优化的主要技术方向，包括但不限于：量化、剪枝、知识蒸馏、KV Cache优化、投机解码等。对于每种技术，请说明其原理、优缺点和适用场景。",
        "假设你是一个系统架构师，需要为一个大规模LLM推理服务设计架构。请详细描述你会如何设计这个系统，包括：硬件选型、模型部署策略、负载均衡、缓存策略、监控告警等方面。",
        "请深入分析一下异构计算在AI推理中的应用。具体讨论GPU、NPU、CPU各自的特点，以及如何根据不同的推理场景选择合适的硬件。同时讨论一下内存层次结构对推理性能的影响。",
    ]
    
    if seq_length == "short":
        base_prompts = short_prompts
    elif seq_length == "long":
        base_prompts = long_prompts
    else:
        base_prompts = medium_prompts
    
    # 生成指定数量的提示
    prompts = []
    for i in range(num_prompts):
        prompts.append(base_prompts[i % len(base_prompts)])
    
    return prompts


def main():
    parser = argparse.ArgumentParser(description="HetKVCache LLM Inference Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="Model path or name")
    parser.add_argument("--num-prompts", type=int, default=10,
                        help="Number of test prompts")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Maximum new tokens to generate")
    parser.add_argument("--seq-length", type=str, default="medium",
                        choices=["short", "medium", "long"],
                        help="Sequence length type")
    parser.add_argument("--device", type=str, default="npu",
                        choices=["npu", "cpu"],
                        help="Device to use")
    parser.add_argument("--output", type=str, default="",
                        help="Output file for results")
    
    args = parser.parse_args()
    
    # 检查NPU可用性
    if args.device == "npu":
        try:
            if torch.npu.is_available():
                print(f"NPU available: {torch.npu.device_count()} devices")
                print(f"Current NPU: {torch.npu.current_device()}")
                print(f"NPU name: {torch.npu.get_device_name(0)}")
            else:
                print("NPU not available, falling back to CPU")
                args.device = "cpu"
        except Exception as e:
            print(f"Error checking NPU: {e}")
            args.device = "cpu"
    
    # 创建benchmark实例
    benchmark = LLMInferenceBenchmark(
        model_path=args.model,
        device=args.device
    )
    
    # 加载模型
    benchmark.load_model()
    
    # 生成测试提示
    prompts = generate_test_prompts(
        num_prompts=args.num_prompts,
        seq_length=args.seq_length
    )
    
    # 运行benchmark
    stats = benchmark.benchmark(
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
    )
    
    # 打印结果
    benchmark.print_stats(stats)
    
    # 保存结果
    if args.output:
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model": args.model,
                "device": args.device,
                "num_prompts": args.num_prompts,
                "max_new_tokens": args.max_new_tokens,
                "seq_length": args.seq_length,
            },
            "stats": stats,
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    
    return stats


if __name__ == "__main__":
    main()