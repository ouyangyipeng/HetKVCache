"""
vLLM Integration Example for HetKVCache

This example demonstrates how to integrate HetKVCache with vLLM
for optimized KV cache management in LLM inference.

Prerequisites:
- vLLM installed (pip install vllm)
- HetKVCache Python bindings built

Usage:
    python vllm_example.py --model meta-llama/Llama-2-7b-hf --prompt "Hello, world!"
"""

import argparse
import time
from typing import List, Optional
import numpy as np

# Note: This is a mock integration example. In production, you would need:
# 1. HetKVCache Python bindings (via pybind11)
# 2. vLLM installed with custom KV cache backend support

try:
    import hetkvcache
    HETKVCACHE_AVAILABLE = True
except ImportError:
    HETKVCACHE_AVAILABLE = False
    print("Warning: HetKVCache Python bindings not available. Using mock implementation.")


class MockHetKVCache:
    """Mock implementation for demonstration when HetKVCache is not available."""
    
    def __init__(self, vram_capacity: int, ram_capacity: int, block_size: int = 16 * 1024):
        self.vram_capacity = vram_capacity
        self.ram_capacity = ram_capacity
        self.block_size = block_size
        self.blocks = {}
        self.next_block_id = 0
        self.access_counts = {}
        
    def allocate_block(self, layer: int, num_blocks: int) -> int:
        block_id = self.next_block_id
        self.next_block_id += 1
        self.blocks[block_id] = {
            'layer': layer,
            'num_blocks': num_blocks,
            'data': np.zeros(num_blocks * self.block_size, dtype=np.float16),
            'tier': 'VRAM',  # Start in VRAM
        }
        return block_id
    
    def write_block(self, block_id: int, data: np.ndarray):
        if block_id in self.blocks:
            self.blocks[block_id]['data'] = data.copy()
            
    def read_block(self, block_id: int) -> np.ndarray:
        if block_id in self.blocks:
            self.record_access(block_id, 'READ')
            return self.blocks[block_id]['data'].copy()
        return None
    
    def release_block(self, block_id: int):
        if block_id in self.blocks:
            del self.blocks[block_id]
            
    def record_access(self, block_id: int, access_type: str):
        self.access_counts[block_id] = self.access_counts.get(block_id, 0) + 1
        
    def get_heat_score(self, block_id: int) -> float:
        total_accesses = sum(self.access_counts.values())
        if total_accesses == 0:
            return 0.0
        return self.access_counts.get(block_id, 0) / total_accesses
    
    def get_current_tier(self, block_id: int) -> str:
        if block_id in self.blocks:
            return self.blocks[block_id]['tier']
        return 'UNKNOWN'
    
    def get_statistics(self) -> dict:
        return {
            'total_blocks': len(self.blocks),
            'total_accesses': sum(self.access_counts.values()),
        }


class HetKVCacheBackend:
    """
    HetKVCache backend adapter for vLLM integration.
    
    This class provides the interface expected by vLLM's KV cache manager,
    backed by HetKVCache's hierarchical storage management.
    """
    
    def __init__(
        self,
        vram_capacity: int = 16 * 1024**3,  # 16GB
        ram_capacity: int = 64 * 1024**3,    # 64GB
        block_size: int = 16 * 1024,          # 16KB
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
    ):
        """
        Initialize HetKVCache backend.
        
        Args:
            vram_capacity: GPU VRAM capacity in bytes
            ram_capacity: Host RAM capacity in bytes
            block_size: Size of each cache block in bytes
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
        """
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Calculate tokens per block
        # Each block stores: 2 * num_layers * 2 * num_heads * head_dim * dtype_size
        # For K and V cache, factor of 2 for key and value
        dtype_size = 2  # float16
        kv_size_per_token = 2 * num_heads * head_dim * dtype_size
        self.tokens_per_block = block_size // (2 * kv_size_per_token)
        
        print(f"HetKVCache Configuration:")
        print(f"  Block size: {block_size / 1024} KB")
        print(f"  Tokens per block: {self.tokens_per_block}")
        print(f"  VRAM capacity: {vram_capacity / 1024**3:.1f} GB")
        print(f"  RAM capacity: {ram_capacity / 1024**3:.1f} GB")
        
        # Initialize HetKVCache
        if HETKVCACHE_AVAILABLE:
            config = hetkvcache.Config()
            config.vram_capacity = vram_capacity
            config.ram_capacity = ram_capacity
            config.block_size = block_size
            config.enable_prefetch = True
            config.hot_threshold = 0.7
            config.cold_threshold = 0.3
            self.cache = hetkvcache.KVCacheManager(config)
        else:
            self.cache = MockHetKVCache(vram_capacity, ram_capacity, block_size)
        
        # Block tracking
        self.sequence_blocks = {}  # seq_id -> list of block_ids
        self.block_info = {}  # block_id -> (layer, token_range)
        
    def allocate_slots(
        self,
        seq_id: int,
        num_tokens: int,
    ) -> List[int]:
        """
        Allocate cache slots for a sequence.
        
        Args:
            seq_id: Sequence identifier
            num_tokens: Number of tokens to allocate
            
        Returns:
            List of block IDs allocated
        """
        num_blocks_needed = (num_tokens + self.tokens_per_block - 1) // self.tokens_per_block
        
        block_ids = []
        for layer in range(self.num_layers):
            layer_block_id = self.cache.allocate_block(layer, num_blocks_needed)
            if layer_block_id < 0:  # Allocation failed
                # Rollback
                for bid in block_ids:
                    self.cache.release_block(bid)
                raise RuntimeError(f"Failed to allocate blocks for sequence {seq_id}")
            block_ids.append(layer_block_id)
            self.block_info[layer_block_id] = (layer, (0, num_tokens))
        
        self.sequence_blocks[seq_id] = block_ids
        return block_ids
    
    def write_kv_cache(
        self,
        seq_id: int,
        layer_idx: int,
        key_cache: np.ndarray,
        value_cache: np.ndarray,
    ):
        """
        Write KV cache for a specific layer.
        
        Args:
            seq_id: Sequence identifier
            layer_idx: Layer index
            key_cache: Key cache tensor [num_tokens, num_heads, head_dim]
            value_cache: Value cache tensor [num_tokens, num_heads, head_dim]
        """
        if seq_id not in self.sequence_blocks:
            raise ValueError(f"Sequence {seq_id} not found")
        
        block_id = self.sequence_blocks[seq_id][layer_idx]
        
        # Combine K and V cache
        kv_data = np.concatenate([key_cache, value_cache], axis=-1)
        kv_data = kv_data.astype(np.float16).flatten()
        
        self.cache.write_block(block_id, kv_data)
        
    def read_kv_cache(
        self,
        seq_id: int,
        layer_idx: int,
    ) -> tuple:
        """
        Read KV cache for a specific layer.
        
        Args:
            seq_id: Sequence identifier
            layer_idx: Layer index
            
        Returns:
            Tuple of (key_cache, value_cache)
        """
        if seq_id not in self.sequence_blocks:
            raise ValueError(f"Sequence {seq_id} not found")
        
        block_id = self.sequence_blocks[seq_id][layer_idx]
        
        kv_data = self.cache.read_block(block_id)
        
        if kv_data is None:
            return None, None
        
        # Split K and V cache
        half_size = kv_data.shape[0] // 2
        key_cache = kv_data[:half_size].reshape(-1, self.num_heads, self.head_dim)
        value_cache = kv_data[half_size:].reshape(-1, self.num_heads, self.head_dim)
        
        return key_cache, value_cache
    
    def free_sequence(self, seq_id: int):
        """
        Free all cache blocks for a sequence.
        
        Args:
            seq_id: Sequence identifier
        """
        if seq_id in self.sequence_blocks:
            for block_id in self.sequence_blocks[seq_id]:
                self.cache.release_block(block_id)
            del self.sequence_blocks[seq_id]
    
    def record_access(self, seq_id: int, layer_idx: int, access_type: str = 'READ'):
        """
        Record an access for heat evaluation.
        
        Args:
            seq_id: Sequence identifier
            layer_idx: Layer index
            access_type: 'READ' or 'WRITE'
        """
        if seq_id in self.sequence_blocks:
            block_id = self.sequence_blocks[seq_id][layer_idx]
            self.cache.record_access(block_id, access_type)
    
    def get_heat_scores(self, seq_id: int) -> dict:
        """
        Get heat scores for all layers of a sequence.
        
        Args:
            seq_id: Sequence identifier
            
        Returns:
            Dictionary mapping layer index to heat score
        """
        if seq_id not in self.sequence_blocks:
            return {}
        
        scores = {}
        for layer_idx, block_id in enumerate(self.sequence_blocks[seq_id]):
            scores[layer_idx] = self.cache.get_heat_score(block_id)
        return scores
    
    def get_statistics(self) -> dict:
        """Get cache statistics."""
        stats = self.cache.get_statistics()
        return {
            'total_blocks': stats.get('total_blocks', 0),
            'vram_blocks': stats.get('vram_blocks', 0),
            'ram_blocks': stats.get('ram_blocks', 0),
            'ssd_blocks': stats.get('ssd_blocks', 0),
            'cache_hit_rate': stats.get('cache_hit_rate', 0),
            'migration_count': stats.get('migration_count', 0),
        }


def simulate_llm_inference(
    backend: HetKVCacheBackend,
    num_sequences: int = 4,
    sequence_length: int = 2048,
    num_layers: int = 32,
):
    """
    Simulate LLM inference with HetKVCache backend.
    
    This function simulates the KV cache access patterns during
    typical LLM inference to demonstrate HetKVCache's benefits.
    """
    print(f"\n{'='*60}")
    print(f"Simulating LLM Inference")
    print(f"{'='*60}")
    print(f"Sequences: {num_sequences}")
    print(f"Sequence Length: {sequence_length}")
    print(f"Layers: {num_layers}")
    print(f"{'='*60}\n")
    
    # Simulate prompt phase (all tokens at once)
    print("Phase 1: Prompt Processing")
    print("-" * 40)
    
    seq_ids = []
    for i in range(num_sequences):
        seq_id = i
        
        # Allocate cache slots
        start_time = time.time()
        block_ids = backend.allocate_slots(seq_id, sequence_length)
        allocate_time = time.time() - start_time
        
        seq_ids.append(seq_id)
        print(f"  Seq {seq_id}: Allocated {len(block_ids)} blocks in {allocate_time*1000:.2f}ms")
    
    # Simulate writing KV cache for prompt phase
    print("\nWriting KV cache (prompt phase)...")
    for seq_id in seq_ids:
        for layer_idx in range(num_layers):
            # Simulate KV cache data
            key_cache = np.random.randn(sequence_length, backend.num_heads, backend.head_dim).astype(np.float16)
            value_cache = np.random.randn(sequence_length, backend.num_heads, backend.head_dim).astype(np.float16)
            
            backend.write_kv_cache(seq_id, layer_idx, key_cache, value_cache)
        
        if (seq_id + 1) % 4 == 0:
            print(f"  Processed {seq_id + 1} sequences...")
    
    print(f"  KV cache written for all sequences")
    
    # Simulate generation phase (one token at a time)
    print("\nPhase 2: Token Generation")
    print("-" * 40)
    
    num_generate_tokens = 128
    total_reads = 0
    total_hits = 0
    
    for gen_idx in range(num_generate_tokens):
        for seq_id in seq_ids:
            for layer_idx in range(num_layers):
                # Read existing KV cache (this is where migration matters)
                start_time = time.time()
                key_cache, value_cache = backend.read_kv_cache(seq_id, layer_idx)
                read_time = time.time() - start_time
                
                # Record access for heat evaluation
                backend.record_access(seq_id, layer_idx, 'READ')
                
                total_reads += 1
                if read_time < 0.001:  # Assuming VRAM access
                    total_hits += 1
        
        if (gen_idx + 1) % 32 == 0:
            print(f"  Generated {gen_idx + 1} tokens...")
    
    # Print statistics
    print("\n" + "="*60)
    print("Inference Statistics")
    print("="*60)
    
    stats = backend.get_statistics()
    print(f"Total blocks: {stats['total_blocks']}")
    print(f"Cache hit rate: {total_hits / total_reads * 100:.1f}%")
    print(f"Migration count: {stats.get('migration_count', 0)}")
    
    # Show heat scores for first sequence
    if seq_ids:
        heat_scores = backend.get_heat_scores(seq_ids[0])
        print(f"\nHeat scores for sequence 0 (first 5 layers):")
        for layer_idx in range(min(5, len(heat_scores))):
            score = heat_scores.get(layer_idx, 0)
            print(f"  Layer {layer_idx}: {score:.3f}")
    
    # Cleanup
    print("\nCleaning up...")
    for seq_id in seq_ids:
        backend.free_sequence(seq_id)
    
    print("Done!")


def benchmark_comparison():
    """
    Compare HetKVCache with baseline (no migration) performance.
    """
    print("\n" + "="*60)
    print("Benchmark: HetKVCache vs Baseline")
    print("="*60)
    
    # Test parameters
    test_configs = [
        {"name": "Short sequences", "num_seq": 16, "seq_len": 512, "layers": 32},
        {"name": "Medium sequences", "num_seq": 8, "seq_len": 2048, "layers": 32},
        {"name": "Long sequences", "num_seq": 4, "seq_len": 8192, "layers": 32},
        {"name": "Ultra-long sequences", "num_seq": 2, "seq_len": 32768, "layers": 32},
    ]
    
    for config in test_configs:
        print(f"\n{config['name']}:")
        print("-" * 40)
        
        # Create backend with limited VRAM to force migration
        backend = HetKVCacheBackend(
            vram_capacity=2 * 1024**3,   # 2GB VRAM (limited)
            ram_capacity=16 * 1024**3,   # 16GB RAM
            block_size=16 * 1024,
            num_layers=config['layers'],
        )
        
        start_time = time.time()
        simulate_llm_inference(
            backend,
            num_sequences=config['num_seq'],
            sequence_length=config['seq_len'],
            num_layers=config['layers'],
        )
        total_time = time.time() - start_time
        
        print(f"\nTotal time: {total_time:.2f}s")
        print(f"Throughput: {config['num_seq'] * config['seq_len'] / total_time:.0f} tokens/s")


def main():
    parser = argparse.ArgumentParser(description="HetKVCache vLLM Integration Example")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Model name or path")
    parser.add_argument("--prompt", type=str, default="Hello, world!",
                        help="Input prompt")
    parser.add_argument("--vram", type=int, default=16,
                        help="VRAM capacity in GB")
    parser.add_argument("--ram", type=int, default=64,
                        help="RAM capacity in GB")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark comparison")
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_comparison()
    else:
        # Standard inference simulation
        backend = HetKVCacheBackend(
            vram_capacity=args.vram * 1024**3,
            ram_capacity=args.ram * 1024**3,
            num_layers=32,
        )
        
        simulate_llm_inference(backend)


if __name__ == "__main__":
    main()