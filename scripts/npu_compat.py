#!/usr/bin/env python3
"""
NPU兼容性补丁模块
解决torch_npu不支持部分操作的问题
"""

import torch
import warnings

# 保存原始函数的模块级变量
_TORCH_ANY_ORIGINAL = None
_TORCH_ALL_ORIGINAL = None
_TORCH_ISIN_ORIGINAL = None
_TENSOR_ANY_ORIGINAL = None
_TENSOR_ALL_ORIGINAL = None


def _create_npu_any_patch(original_any):
    """创建NPU兼容的any补丁"""
    def npu_fallback_any(input, dim=None, keepdim=False, *, out=None):
        # 检查是否是Tensor
        if not isinstance(input, torch.Tensor):
            if input is None:
                return torch.tensor(False)
            if dim is None:
                return original_any(input)
            return original_any(input, dim=dim, keepdim=keepdim, out=out)
        
        # NPU tensor强制CPU回退
        if input.device.type == 'npu':
            input_cpu = input.cpu()
            if dim is None:
                result = original_any(input_cpu)
            else:
                result = original_any(input_cpu, dim=dim, keepdim=keepdim)
            return result
        
        # 非NPU tensor正常处理
        if dim is None:
            return original_any(input)
        return original_any(input, dim=dim, keepdim=keepdim)
    
    return npu_fallback_any


def _create_npu_all_patch(original_all):
    """创建NPU兼容的all补丁"""
    def npu_fallback_all(input, dim=None, keepdim=False, *, out=None):
        if not isinstance(input, torch.Tensor):
            if input is None:
                return torch.tensor(True)
            if dim is None:
                return original_all(input)
            return original_all(input, dim=dim, keepdim=keepdim, out=out)
        
        if input.device.type == 'npu':
            input_cpu = input.cpu()
            if dim is None:
                result = original_all(input_cpu)
            else:
                result = original_all(input_cpu, dim=dim, keepdim=keepdim)
            return result
        
        if dim is None:
            return original_all(input)
        return original_all(input, dim=dim, keepdim=keepdim)
    
    return npu_fallback_all


def _create_npu_isin_patch(original_isin):
    """创建NPU兼容的isin补丁"""
    def npu_fallback_isin(elements, test_elements, assume_unique=False, invert=False):
        # NPU tensor强制CPU回退
        if isinstance(elements, torch.Tensor) and elements.device.type == 'npu':
            elements_cpu = elements.cpu()
            test_cpu = test_elements.cpu() if isinstance(test_elements, torch.Tensor) and test_elements.device.type == 'npu' else test_elements
            result = original_isin(elements_cpu, test_cpu, assume_unique=assume_unique, invert=invert)
            return result.to(elements.device)
        
        return original_isin(elements, test_elements, assume_unique=assume_unique, invert=invert)
    
    return npu_fallback_isin


def _create_tensor_any_patch(original_tensor_any):
    """创建Tensor.any()的NPU兼容补丁"""
    def tensor_any_npu_patch(self, dim=None, keepdim=False):
        if self.device.type == 'npu':
            self_cpu = self.cpu()
            if dim is None:
                return original_tensor_any(self_cpu)
            return original_tensor_any(self_cpu, dim=dim, keepdim=keepdim)
        if dim is None:
            return original_tensor_any(self)
        return original_tensor_any(self, dim=dim, keepdim=keepdim)
    
    return tensor_any_npu_patch


def _create_tensor_all_patch(original_tensor_all):
    """创建Tensor.all()的NPU兼容补丁"""
    def tensor_all_npu_patch(self, dim=None, keepdim=False):
        if self.device.type == 'npu':
            self_cpu = self.cpu()
            if dim is None:
                return original_tensor_all(self_cpu)
            return original_tensor_all(self_cpu, dim=dim, keepdim=keepdim)
        if dim is None:
            return original_tensor_all(self)
        return original_tensor_all(self, dim=dim, keepdim=keepdim)
    
    return tensor_all_npu_patch


def patch_torch_for_npu():
    """
    为torch_npu打补丁，解决兼容性问题
    """
    global _TORCH_ANY_ORIGINAL, _TORCH_ALL_ORIGINAL, _TORCH_ISIN_ORIGINAL
    global _TENSOR_ANY_ORIGINAL, _TENSOR_ALL_ORIGINAL
    
    # 保存原始函数（只保存一次）
    if _TORCH_ANY_ORIGINAL is None:
        _TORCH_ANY_ORIGINAL = torch.any
        _TORCH_ALL_ORIGINAL = torch.all
        _TORCH_ISIN_ORIGINAL = torch.isin
        _TENSOR_ANY_ORIGINAL = torch.Tensor.any
        _TENSOR_ALL_ORIGINAL = torch.Tensor.all
    
    # 创建补丁函数（使用保存的原始函数）
    npu_any = _create_npu_any_patch(_TORCH_ANY_ORIGINAL)
    npu_all = _create_npu_all_patch(_TORCH_ALL_ORIGINAL)
    npu_isin = _create_npu_isin_patch(_TORCH_ISIN_ORIGINAL)
    tensor_any = _create_tensor_any_patch(_TENSOR_ANY_ORIGINAL)
    tensor_all = _create_tensor_all_patch(_TENSOR_ALL_ORIGINAL)
    
    # 应用补丁 - 替换torch函数
    torch.any = npu_any
    torch.all = npu_all
    torch.isin = npu_isin
    
    # 替换Tensor方法
    torch.Tensor.any = tensor_any
    torch.Tensor.all = tensor_all
    
    # 也修补torch.functional中的版本
    if hasattr(torch, 'functional'):
        torch.functional.any = npu_any
        torch.functional.all = npu_all
    
    print("NPU compatibility patches applied:")
    print("  - torch.isin: CPU fallback for NPU tensors")
    print("  - torch.any: CPU fallback for NPU tensors")
    print("  - torch.all: CPU fallback for NPU tensors")
    print("  - torch.Tensor.any: CPU fallback for NPU tensors")
    print("  - torch.Tensor.all: CPU fallback for NPU tensors")


def patch_transformers_for_npu():
    """
    为transformers库打补丁，解决NPU兼容性问题
    """
    try:
        import transformers
        from transformers.generation.utils import GenerationMixin
        
        # 保存原始方法
        original_prepare_logits_for_warper = getattr(GenerationMixin, '_prepare_logits_for_warper', None)
        
        if original_prepare_logits_for_warper:
            def patched_prepare_logits_for_warper(self, logits, next_token_scores):
                """NPU兼容的logits准备"""
                if logits.device.type == 'npu':
                    logits = logits.cpu()
                    next_token_scores = next_token_scores.cpu()
                    result = original_prepare_logits_for_warper(self, logits, next_token_scores)
                    return result.to('npu')
                return original_prepare_logits_for_warper(self, logits, next_token_scores)
            
            GenerationMixin._prepare_logits_for_warper = patched_prepare_logits_for_warper
            print("  - transformers.GenerationMixin._prepare_logits_for_warper: patched")
        
        print("Transformers NPU compatibility patches applied")
    except ImportError:
        print("Transformers not available, skipping patches")


def apply_all_npu_patches():
    """应用所有NPU兼容性补丁"""
    print("\n" + "="*50)
    print("Applying NPU compatibility patches...")
    print("="*50)
    
    patch_torch_for_npu()
    patch_transformers_for_npu()
    
    print("="*50)
    print("All NPU compatibility patches applied successfully!")
    print("="*50 + "\n")


if __name__ == "__main__":
    # 测试补丁
    apply_all_npu_patches()
    
    # 测试isin
    import torch
    try:
        import torch_npu
        print(f"torch_npu version: {torch_npu.__version__}")
        print(f"NPU available: {torch.npu.is_available()}")
        if torch.npu.is_available():
            print(f"NPU count: {torch.npu.device_count()}")
            print(f"Current NPU: {torch.npu.current_device()}")
            
            # 测试isin
            x = torch.tensor([1, 2, 3, 4, 5]).npu()
            y = torch.tensor([2, 4]).npu()
            result = torch.isin(x, y)
            print(f"torch.isin test: {result.cpu().tolist()}")
            assert result.cpu().tolist() == [False, True, False, True, False]
            print("torch.isin test passed!")
            
            # 测试any
            t = torch.tensor([False, True, False]).npu()
            result = torch.any(t)
            print(f"torch.any test: {result.cpu().item()}")
            assert result.cpu().item() == True
            print("torch.any test passed!")
            
            # 测试all
            t = torch.tensor([True, True, True]).npu()
            result = torch.all(t)
            print(f"torch.all test: {result.cpu().item()}")
            assert result.cpu().item() == True
            print("torch.all test passed!")
    except ImportError:
        print("torch_npu not available, skipping NPU tests")