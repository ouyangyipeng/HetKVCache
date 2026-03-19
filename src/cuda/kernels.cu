/**
 * @file kernels.cu
 * @brief CUDA 内核函数实现
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

namespace hetkvcache {

// ============================================================================
// 数学常量
// ============================================================================

constexpr float PI = 3.14159265358979323846f;

// ============================================================================
// 简单的量化/反量化内核
// ============================================================================

/**
 * @brief FP32 到 FP16 量化内核
 */
__global__ void quantizeFP32ToFP16Kernel(
    const float* input,
    half* output,
    size_t n) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2half(input[idx]);
    }
}

/**
 * @brief FP16 到 FP32 反量化内核
 */
__global__ void dequantizeFP16ToFP32Kernel(
    const half* input,
    float* output,
    size_t n) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __half2float(input[idx]);
    }
}

// ============================================================================
// INT8 量化内核
// ============================================================================

/**
 * @brief FP32 到 INT8 量化内核
 */
__global__ void quantizeFP32ToINT8Kernel(
    const float* input,
    int8_t* output,
    float scale,
    size_t n) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx] / scale;
        val = fmaxf(-128.0f, fminf(127.0f, val));
        output[idx] = static_cast<int8_t>(roundf(val));
    }
}

/**
 * @brief INT8 到 FP32 反量化内核
 */
__global__ void dequantizeINT8ToFP32Kernel(
    const int8_t* input,
    float* output,
    float scale,
    size_t n) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = static_cast<float>(input[idx]) * scale;
    }
}

// ============================================================================
// 简单的压缩内核（差分编码）
// ============================================================================

/**
 * @brief 差分编码内核
 */
__global__ void differentialEncodeKernel(
    const float* input,
    float* output,
    float base,
    size_t n) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (idx == 0) {
            output[idx] = input[idx] - base;
        } else {
            output[idx] = input[idx] - input[idx - 1];
        }
    }
}

/**
 * @brief 差分解码内核
 */
__global__ void differentialDecodeKernel(
    const float* input,
    float* output,
    float base,
    size_t n) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (idx == 0) {
            output[idx] = input[idx] + base;
        } else {
            output[idx] = input[idx] + output[idx - 1];
        }
    }
}

// ============================================================================
// 数据处理内核
// ============================================================================

/**
 * @brief 归一化内核
 */
__global__ void normalizeKernel(
    float* data,
    float mean,
    float std,
    size_t n) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = (data[idx] - mean) / std;
    }
}

/**
 * @brief 缩放内核
 */
__global__ void scaleKernel(
    float* data,
    float scale,
    size_t n) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

// ============================================================================
// 聚合内核
// ============================================================================

/**
 * @brief 向量加法内核
 */
__global__ void vectorAddKernel(
    const float* a,
    const float* b,
    float* c,
    size_t n) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * @brief 向量点积内核（简化版）
 */
__global__ void dotProductKernel(
    const float* a,
    const float* b,
    float* result,
    size_t n) {
    
    extern __shared__ float shared[];
    
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    shared[tid] = 0.0f;
    
    if (idx < n) {
        shared[tid] = a[idx] * b[idx];
    }
    
    __syncthreads();
    
    // 归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, shared[0]);
    }
}

// ============================================================================
// 内存操作内核
// ============================================================================

/**
 * @brief 批量复制内核
 */
__global__ void batchCopyKernel(
    const float* const* srcs,
    float** dsts,
    const size_t* sizes,
    int num_arrays) {
    
    int array_idx = blockIdx.y;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (array_idx < num_arrays && idx < sizes[array_idx]) {
        dsts[array_idx][idx] = srcs[array_idx][idx];
    }
}

/**
 * @brief 批量填充内核
 */
__global__ void batchFillKernel(
    float** dsts,
    float value,
    const size_t* sizes,
    int num_arrays) {
    
    int array_idx = blockIdx.y;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (array_idx < num_arrays && idx < sizes[array_idx]) {
        dsts[array_idx][idx] = value;
    }
}

// ============================================================================
// 内核启动包装函数
// ============================================================================

extern "C" void launchQuantizeFP32ToFP16(
    const float* input,
    void* output,
    size_t n,
    cudaStream_t stream) {
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    quantizeFP32ToFP16Kernel<<<numBlocks, blockSize, 0, stream>>>(
        input, reinterpret_cast<half*>(output), n);
}

extern "C" void launchDequantizeFP16ToFP32(
    const void* input,
    float* output,
    size_t n,
    cudaStream_t stream) {
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    dequantizeFP16ToFP32Kernel<<<numBlocks, blockSize, 0, stream>>>(
        reinterpret_cast<const half*>(input), output, n);
}

extern "C" void launchQuantizeFP32ToINT8(
    const float* input,
    int8_t* output,
    float scale,
    size_t n,
    cudaStream_t stream) {
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    quantizeFP32ToINT8Kernel<<<numBlocks, blockSize, 0, stream>>>(
        input, output, scale, n);
}

extern "C" void launchDequantizeINT8ToFP32(
    const int8_t* input,
    float* output,
    float scale,
    size_t n,
    cudaStream_t stream) {
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    dequantizeINT8ToFP32Kernel<<<numBlocks, blockSize, 0, stream>>>(
        input, output, scale, n);
}

extern "C" void launchScaleKernel(
    float* data,
    float scale,
    size_t n,
    cudaStream_t stream) {
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    scaleKernel<<<numBlocks, blockSize, 0, stream>>>(data, scale, n);
}

extern "C" void launchVectorAdd(
    const float* a,
    const float* b,
    float* c,
    size_t n,
    cudaStream_t stream) {
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    vectorAddKernel<<<numBlocks, blockSize, 0, stream>>>(a, b, c, n);
}

}  // namespace hetkvcache