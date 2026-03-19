/**
 * @file memory_transfer.cu
 * @brief CUDA 内存传输实现
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

namespace hetkvcache {

// CUDA 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while(0)

/**
 * @brief 异步内存传输
 */
extern "C" bool asyncMemoryTransfer(
    void* dst, const void* src, size_t size,
    cudaStream_t stream, int direction) {
    
    cudaError_t err;
    
    switch (direction) {
        case 0:  // Host to Device
            err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
            break;
        case 1:  // Device to Host
            err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
            break;
        case 2:  // Device to Device
            err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
            break;
        default:
            return false;
    }
    
    return err == cudaSuccess;
}

/**
 * @brief 批量内存传输
 */
extern "C" bool batchMemoryTransfer(
    void** dsts, const void** srcs, size_t* sizes,
    int count, cudaStream_t stream, int direction) {
    
    for (int i = 0; i < count; i++) {
        if (!asyncMemoryTransfer(dsts[i], srcs[i], sizes[i], stream, direction)) {
            return false;
        }
    }
    
    return cudaStreamSynchronize(stream) == cudaSuccess;
}

/**
 * @brief 分配锁页内存
 */
extern "C" bool allocatePinnedMemory(void** ptr, size_t size) {
    cudaError_t err = cudaMallocHost(ptr, size);
    return err == cudaSuccess;
}

/**
 * @brief 释放锁页内存
 */
extern "C" void freePinnedMemory(void* ptr) {
    cudaFreeHost(ptr);
}

/**
 * @brief 分配设备内存
 */
extern "C" bool allocateDeviceMemory(void** ptr, size_t size) {
    cudaError_t err = cudaMalloc(ptr, size);
    return err == cudaSuccess;
}

/**
 * @brief 释放设备内存
 */
extern "C" void freeDeviceMemory(void* ptr) {
    cudaFree(ptr);
}

/**
 * @brief 创建 CUDA Stream
 */
extern "C" bool createCudaStream(cudaStream_t* stream) {
    cudaError_t err = cudaStreamCreate(stream);
    return err == cudaSuccess;
}

/**
 * @brief 销毁 CUDA Stream
 */
extern "C" void destroyCudaStream(cudaStream_t stream) {
    cudaStreamDestroy(stream);
}

/**
 * @brief 同步 Stream
 */
extern "C" bool synchronizeStream(cudaStream_t stream) {
    cudaError_t err = cudaStreamSynchronize(stream);
    return err == cudaSuccess;
}

/**
 * @brief 获取设备属性
 */
extern "C" bool getDeviceMemoryInfo(size_t* free, size_t* total) {
    cudaError_t err = cudaMemGetInfo(free, total);
    return err == cudaSuccess;
}

/**
 * @brief 设置设备
 */
extern "C" bool setDevice(int device) {
    cudaError_t err = cudaSetDevice(device);
    return err == cudaSuccess;
}

/**
 * @brief 获取设备数量
 */
extern "C" int getDeviceCount() {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess) ? count : 0;
}

/**
 * @brief 初始化 CUDA
 */
extern "C" bool initializeCuda(int device_id) {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || device_id >= count) {
        return false;
    }
    
    err = cudaSetDevice(device_id);
    return err == cudaSuccess;
}

/**
 * @brief 清理 CUDA
 */
extern "C" void cleanupCuda() {
    cudaDeviceReset();
}

// ============================================================================
// CUDA Kernel 实现
// ============================================================================

/**
 * @brief 复制内核
 */
__global__ void copyKernel(const float* src, float* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

/**
 * @brief 填充内核
 */
__global__ void fillKernel(float* dst, float value, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        dst[i] = value;
    }
}

/**
 * @brief 简单压缩内核（RLE 风格）
 */
__global__ void compressKernel(
    const float* data, 
    unsigned char* compressed,
    size_t n,
    size_t* compressed_size) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx == 0) {
        // 简单实现：只复制数据
        // 实际应该实现真正的压缩
        memcpy(compressed, data, n * sizeof(float));
        *compressed_size = n * sizeof(float);
    }
}

/**
 * @brief 解压缩内核
 */
__global__ void decompressKernel(
    const unsigned char* compressed,
    float* data,
    size_t compressed_size,
    size_t n) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx == 0) {
        // 简单实现：只复制数据
        memcpy(data, compressed, n * sizeof(float));
    }
}

// ============================================================================
// 内核启动包装函数
// ============================================================================

extern "C" void launchCopyKernel(
    const float* src, float* dst, size_t n,
    cudaStream_t stream) {
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    copyKernel<<<numBlocks, blockSize, 0, stream>>>(src, dst, n);
}

extern "C" void launchFillKernel(
    float* dst, float value, size_t n,
    cudaStream_t stream) {
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    fillKernel<<<numBlocks, blockSize, 0, stream>>>(dst, value, n);
}

}  // namespace hetkvcache