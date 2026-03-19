"""
HuggingFace Transformers Integration Example for HetKVCache

This example demonstrates how to integrate HetKVCache with HuggingFace
Transformers for optimized KV cache management during inference.

Prerequisites:
- transformers library installed (pip install transformers)
- torch installed
- HetKVCache Python bindings built

Usage:
    python hf_example.py --model gpt2 --prompt "The quick brown fox" --max_length 100
"""

import argparse
import time
from typing import Optional, Tuple, List
import numpy as np

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers/torch not available. Using mock implementation.")

try:
    import hetkvcache
    HETKVCACHE_AVAILABLE = True
except ImportError:
    HETKVCACHE_AVAILABLE = False
    print("Warning: HetKVCache Python bindings not available. Using mock implementation.")


class MockHetKVCacheForHF:
    """Mock HetKVCache for demonstration without actual bindings."""
    
    def __init__(self, config: dict):
        self.config = config
        self.blocks = {}
        self.next_block_id = 0
        self.stats = {'hits': 0, 'misses': 0, 'migrations': 0}
        
    def allocate(self, num_blocks: int, layer_idx: int) -> int:
        block_id = self.next_block_id
        self.next_block_id += 1
        self.blocks[block_id] = {
            'layer': layer_idx,
            'size': num_blocks,
            'data': None,
            'tier': 'VRAM',
        }
        return block_id
    
    def write(self, block_id: int, data: np.ndarray):
        self.blocks[block_id]['data'] = data.copy()
        
    def read(self, block_id: int) -> Optional[np.ndarray]:
        if block_id in self.blocks:
            self.stats['hits'] += 1
            return self.blocks[block_id]['data']
        self.stats['misses'] += 1
        return None
    
    def free(self, block_id: int):
        if block_id in self.blocks:
            del self.blocks[block_id]
            
    def get_stats(self) -> dict:
        total = self.stats['hits'] + self.stats['misses']
        return {
            'hit_rate': self.stats['hits'] / total if total > 0 else 0,
            'migrations': self.stats['migrations'],
            'total_blocks': len(self.blocks),
        }


class HetKVCacheKVManager:
    """
    KV Cache manager using HetKVCache backend.
    
    This class manages KV caches for HuggingFace models with
    automatic tier management and migration.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int = 4096,
        vram_capacity: int = 16 * 1024**3,
        ram_capacity: int = 64 * 1024**3,
        block_size: int = 16 * 1024,
    ):
        """
        Initialize KV cache manager.
        
        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            max_seq_len: Maximum sequence length
            vram_capacity: GPU VRAM capacity in bytes
            ram_capacity: Host RAM capacity in bytes
            block_size: Size of each cache block in bytes
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        
        # Calculate tokens per block
        dtype_size = 2  # float16
        kv_size_per_token = 2 * num_heads * head_dim * dtype_size  # K + V
        self.tokens_per_block = block_size // kv_size_per_token
        
        # Initialize HetKVCache
        if HETKVCACHE_AVAILABLE:
            config = hetkvcache.Config()
            config.vram_capacity = vram_capacity
            config.ram_capacity = ram_capacity
            config.block_size = block_size
            self.cache = hetkvcache.KVCacheManager(config)
        else:
            self.cache = MockHetKVCacheForHF({
                'vram_capacity': vram_capacity,
                'ram_capacity': ram_capacity,
                'block_size': block_size,
            })
        
        # KV cache storage
        # Maps: layer_idx -> block_ids for each sequence
        self.layer_blocks = {}  # layer_idx -> list of block_ids
        self.sequence_lengths = {}  # seq_id -> current_length
        
        print(f"HetKVCache KV Manager initialized:")
        print(f"  Layers: {num_layers}")
        print(f"  Heads: {num_heads}")
        print(f"  Head dim: {head_dim}")
        print(f"  Tokens per block: {self.tokens_per_block}")
        
    def allocate_for_sequence(self, seq_id: int, init_length: int = 0):
        """
        Allocate KV cache for a new sequence.
        
        Args:
            seq_id: Sequence identifier
            init_length: Initial sequence length (for prompt)
        """
        self.sequence_lengths[seq_id] = init_length
        
        if seq_id not in self.layer_blocks:
            self.layer_blocks[seq_id] = {}
        
        # Allocate blocks for each layer
        num_blocks_needed = max(1, (init_length + self.tokens_per_block - 1) // self.tokens_per_block)
        
        for layer_idx in range(self.num_layers):
            if HETKVCACHE_AVAILABLE:
                block_id = self.cache.allocate_block(layer_idx, num_blocks_needed)
            else:
                block_id = self.cache.allocate(num_blocks_needed, layer_idx)
            
            self.layer_blocks[seq_id][layer_idx] = [block_id]
    
    def update_kv_cache(
        self,
        seq_id: int,
        layer_idx: int,
        key_states: np.ndarray,
        value_states: np.ndarray,
        position_offset: int = 0,
    ):
        """
        Update KV cache for a specific layer.
        
        Args:
            seq_id: Sequence identifier
            layer_idx: Layer index
            key_states: Key states [batch, num_heads, seq_len, head_dim]
            value_states: Value states [batch, num_heads, seq_len, head_dim]
            position_offset: Offset position for incremental generation
        """
        if seq_id not in self.layer_blocks:
            self.allocate_for_sequence(seq_id)
        
        # Handle different tensor shapes
        if len(key_states.shape) == 4:
            key_states = key_states.squeeze(0)  # Remove batch dim
            value_states = value_states.squeeze(0)
        
        # Convert to numpy if needed
        if hasattr(key_states, 'numpy'):
            key_states = key_states.detach().cpu().numpy()
            value_states = value_states.detach().cpu().numpy()
        
        # Flatten and convert to float16
        key_flat = key_states.astype(np.float16).flatten()
        value_flat = value_states.astype(np.float16).flatten()
        
        # Combine K and V
        kv_data = np.concatenate([key_flat, value_flat])
        
        # Write to cache
        block_ids = self.layer_blocks[seq_id][layer_idx]
        if block_ids:
            if HETKVCACHE_AVAILABLE:
                self.cache.write_block(block_ids[0], kv_data, len(kv_data))
            else:
                self.cache.write(block_ids[0], kv_data)
    
    def get_kv_cache(
        self,
        seq_id: int,
        layer_idx: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Retrieve KV cache for a specific layer.
        
        Args:
            seq_id: Sequence identifier
            layer_idx: Layer index
            
        Returns:
            Tuple of (key_states, value_states)
        """
        if seq_id not in self.layer_blocks:
            return None, None
        
        block_ids = self.layer_blocks[seq_id].get(layer_idx, [])
        if not block_ids:
            return None, None
        
        # Read from cache
        if HETKVCACHE_AVAILABLE:
            kv_data = self.cache.read_block(block_ids[0])
        else:
            kv_data = self.cache.read(block_ids[0])
        
        if kv_data is None:
            return None, None
        
        # Split K and V
        half = len(kv_data) // 2
        key_flat = kv_data[:half]
        value_flat = kv_data[half:]
        
        # Reshape to expected dimensions
        seq_len = len(key_flat) // (self.num_heads * self.head_dim)
        key_states = key_flat.reshape(self.num_heads, seq_len, self.head_dim)
        value_states = value_flat.reshape(self.num_heads, seq_len, self.head_dim)
        
        return key_states, value_states
    
    def extend_sequence(self, seq_id: int, new_length: int):
        """
        Extend cache for growing sequence.
        
        Args:
            seq_id: Sequence identifier
            new_length: New sequence length
        """
        if seq_id not in self.sequence_lengths:
            self.allocate_for_sequence(seq_id, new_length)
            return
        
        old_length = self.sequence_lengths[seq_id]
        if new_length <= old_length:
            return
        
        # Check if we need more blocks
        old_blocks = (old_length + self.tokens_per_block - 1) // self.tokens_per_block
        new_blocks = (new_length + self.tokens_per_block - 1) // self.tokens_per_block
        
        if new_blocks > old_blocks:
            # Allocate additional blocks
            additional_blocks = new_blocks - old_blocks
            for layer_idx in range(self.num_layers):
                if HETKVCACHE_AVAILABLE:
                    block_id = self.cache.allocate_block(layer_idx, additional_blocks)
                else:
                    block_id = self.cache.allocate(additional_blocks, layer_idx)
                
                self.layer_blocks[seq_id][layer_idx].append(block_id)
        
        self.sequence_lengths[seq_id] = new_length
    
    def free_sequence(self, seq_id: int):
        """
        Free all cache blocks for a sequence.
        
        Args:
            seq_id: Sequence identifier
        """
        if seq_id not in self.layer_blocks:
            return
        
        for layer_idx, block_ids in self.layer_blocks[seq_id].items():
            for block_id in block_ids:
                if HETKVCACHE_AVAILABLE:
                    self.cache.release_block(block_id)
                else:
                    self.cache.free(block_id)
        
        del self.layer_blocks[seq_id]
        if seq_id in self.sequence_lengths:
            del self.sequence_lengths[seq_id]
    
    def get_statistics(self) -> dict:
        """Get cache statistics."""
        if HETKVCACHE_AVAILABLE:
            return self.cache.get_statistics()
        else:
            return self.cache.get_stats()


class HuggingFaceKVCacheWrapper:
    """
    Wrapper for HuggingFace models to use HetKVCache.
    
    This wrapper intercepts the KV cache operations during model inference
    and routes them through HetKVCache for optimized memory management.
    """
    
    def __init__(
        self,
        model_name: str,
        vram_capacity: int = 16 * 1024**3,
        ram_capacity: int = 64 * 1024**3,
        device: str = "cuda",
    ):
        """
        Initialize wrapper.
        
        Args:
            model_name: HuggingFace model name or path
            vram_capacity: GPU VRAM capacity in bytes
            ram_capacity: Host RAM capacity in bytes
            device: Device to run on ("cuda" or "cpu")
        """
        self.device = device
        self.model_name = model_name
        
        if TRANSFORMERS_AVAILABLE:
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
            
            # Get model dimensions
            self.num_layers = self.config.num_hidden_layers
            self.num_heads = self.config.num_attention_heads
            self.head_dim = self.config.hidden_size // self.config.num_attention_heads
            
            print(f"Loaded model: {model_name}")
            print(f"  Layers: {self.num_layers}")
            print(f"  Heads: {self.num_heads}")
            print(f"  Head dim: {self.head_dim}")
        else:
            # Mock model configuration
            self.tokenizer = None
            self.config = None
            self.model = None
            self.num_layers = 32
            self.num_heads = 32
            self.head_dim = 128
        
        # Initialize KV cache manager
        self.kv_manager = HetKVCacheKVManager(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            vram_capacity=vram_capacity,
            ram_capacity=ram_capacity,
        )
        
        # Sequence tracking
        self.active_sequences = {}
        self.next_seq_id = 0
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text using HetKVCache-optimized inference.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated text
        """
        if not TRANSFORMERS_AVAILABLE:
            return self._mock_generate(prompt, max_length)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]
        
        # Allocate sequence
        seq_id = self.next_seq_id
        self.next_seq_id += 1
        
        start_time = time.time()
        
        # Prefill phase
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                use_cache=True,
                return_dict=True,
            )
        
        past_key_values = outputs.past_key_values
        self._store_kv_cache(seq_id, past_key_values, input_length)
        
        # Generation phase
        generated_ids = []
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        generated_ids.append(next_token.item())
        
        for step in range(max_length - 1):
            # Update KV cache for new token
            self.kv_manager.extend_sequence(seq_id, input_length + step + 1)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=next_token.unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
            generated_ids.append(next_token.item())
            
            # Stop on EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        generation_time = time.time() - start_time
        
        # Decode
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Cleanup
        self.kv_manager.free_sequence(seq_id)
        
        stats = self.kv_manager.get_statistics()
        print(f"\nGeneration Stats:")
        print(f"  Time: {generation_time:.2f}s")
        print(f"  Tokens: {len(generated_ids)}")
        print(f"  Rate: {len(generated_ids) / generation_time:.1f} tokens/s")
        print(f"  Cache hit rate: {stats.get('hit_rate', 0) * 100:.1f}%")
        
        return generated_text
    
    def _store_kv_cache(self, seq_id: int, past_key_values: tuple, seq_len: int):
        """
        Store past key values in HetKVCache.
        
        Args:
            seq_id: Sequence identifier
            past_key_values: Tuple of (key, value) for each layer
            seq_len: Sequence length
        """
        self.kv_manager.allocate_for_sequence(seq_id, seq_len)
        
        for layer_idx, (key, value) in enumerate(past_key_values):
            self.kv_manager.update_kv_cache(
                seq_id,
                layer_idx,
                key.cpu().numpy(),
                value.cpu().numpy(),
            )
    
    def _mock_generate(self, prompt: str, max_length: int) -> str:
        """Mock generation for demonstration without transformers."""
        print("Running mock generation (transformers not available)")
        
        seq_id = self.next_seq_id
        self.next_seq_id += 1
        
        # Simulate prefill
        self.kv_manager.allocate_for_sequence(seq_id, len(prompt.split()))
        
        # Simulate generation
        words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
        generated = []
        
        for i in range(max_length):
            self.kv_manager.extend_sequence(seq_id, len(prompt.split()) + i + 1)
            generated.append(words[i % len(words)])
        
        # Cleanup
        self.kv_manager.free_sequence(seq_id)
        
        return " ".join(generated)


def benchmark_model(model_name: str, prompt_lengths: List[int]):
    """
    Benchmark a model with different prompt lengths.
    
    Args:
        model_name: HuggingFace model name
        prompt_lengths: List of prompt lengths to test
    """
    print("\n" + "="*60)
    print(f"Benchmarking: {model_name}")
    print("="*60)
    
    wrapper = HuggingFaceKVCacheWrapper(
        model_name=model_name,
        vram_capacity=8 * 1024**3,   # 8GB VRAM
        ram_capacity=32 * 1024**3,    # 32GB RAM
    )
    
    results = []
    
    for prompt_len in prompt_lengths:
        # Generate a prompt of the desired length
        prompt_words = ["token"] * prompt_len
        prompt = " ".join(prompt_words)
        
        print(f"\nPrompt length: {prompt_len}")
        print("-" * 40)
        
        start_time = time.time()
        generated = wrapper.generate(prompt, max_length=50)
        elapsed = time.time() - start_time
        
        results.append({
            'prompt_length': prompt_len,
            'time': elapsed,
            'generated_length': len(generated.split()),
        })
    
    print("\n" + "="*60)
    print("Benchmark Results")
    print("="*60)
    print(f"{'Prompt Len':<12} {'Time (s)':<12} {'Throughput':<15}")
    print("-" * 40)
    
    for r in results:
        throughput = (r['prompt_length'] + r['generated_length']) / r['time']
        print(f"{r['prompt_length']:<12} {r['time']:<12.2f} {throughput:<15.1f} tok/s")


def main():
    parser = argparse.ArgumentParser(description="HetKVCache HuggingFace Integration Example")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="HuggingFace model name")
    parser.add_argument("--prompt", type=str, default="The quick brown fox",
                        help="Input prompt")
    parser.add_argument("--max_length", type=int, default=50,
                        help="Maximum generation length")
    parser.add_argument("--vram", type=int, default=16,
                        help="VRAM capacity in GB")
    parser.add_argument("--ram", type=int, default=64,
                        help="RAM capacity in GB")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark with various prompt lengths")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to use")
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_model(args.model, [128, 256, 512, 1024, 2048])
    else:
        wrapper = HuggingFaceKVCacheWrapper(
            model_name=args.model,
            vram_capacity=args.vram * 1024**3,
            ram_capacity=args.ram * 1024**3,
            device=args.device,
        )
        
        print(f"\nGenerating text...")
        print(f"Prompt: {args.prompt}")
        print("-" * 40)
        
        generated = wrapper.generate(args.prompt, max_length=args.max_length)
        
        print(f"\nGenerated text:")
        print(generated)


if __name__ == "__main__":
    main()