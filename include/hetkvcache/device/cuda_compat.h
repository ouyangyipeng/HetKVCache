/**
 * @file cuda_compat.h
 * @brief CUDA API兼容层 - 用于昇腾NPU适配
 * 
 * 将CUDA API映射到昇腾ACL API，使原有代码可以在昇腾上运行
 */

#ifndef HETKVCACHE_CUDA_COMPAT_H
#define HETKVCACHE_CUDA_COMPAT_H

#include <cstdint>
#include <cstddef>

// CUDA兼容头文件
// 根据USE_ASCEND宏决定使用CUDA还是昇腾API

#ifdef USE_ASCEND
// 昇腾NPU模式
#include "acl/acl.h"
#include <atomic>

namespace hetkvcache {
namespace ascend_compat {

// ACL运行时初始化状态
inline std::atomic<bool>& getAclInitialized() {
    static std::atomic<bool> initialized{false};
    return initialized;
}

// 确保ACL运行时已初始化
inline bool ensureAclInitialized() {
    auto& initialized = getAclInitialized();
    if (initialized.load()) {
        return true;
    }
    
    // 初始化ACL运行时
    aclError err = aclInit(nullptr);
    if (err != ACL_SUCCESS && err != ACL_ERROR_REPEAT_INITIALIZE) {
        return false;
    }
    
    // 设置设备
    err = aclrtSetDevice(0);
    if (err != ACL_SUCCESS) {
        return false;
    }
    
    initialized.store(true);
    return true;
}

} // namespace ascend_compat
} // namespace hetkvcache

// 类型定义
using cudaError_t = int;
using cudaStream_t = void*;

// 错误码映射
constexpr int cudaSuccess = 0;
constexpr int cudaErrorMemoryAllocation = 2;

// 内存拷贝方向
constexpr int cudaMemcpyHostToDevice = ACL_MEMCPY_HOST_TO_DEVICE;
constexpr int cudaMemcpyDeviceToHost = ACL_MEMCPY_DEVICE_TO_HOST;
constexpr int cudaMemcpyDeviceToDevice = ACL_MEMCPY_DEVICE_TO_DEVICE;

// CUDA API映射到ACL API
inline cudaError_t cudaMalloc(void** ptr, size_t size) {
    if (!hetkvcache::ascend_compat::ensureAclInitialized()) {
        return cudaErrorMemoryAllocation;
    }
    return aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
}

inline cudaError_t cudaMallocHost(void** ptr, size_t size) {
    if (!hetkvcache::ascend_compat::ensureAclInitialized()) {
        return cudaErrorMemoryAllocation;
    }
    return aclrtMallocHost(ptr, size);
}

inline cudaError_t cudaFree(void* ptr) {
    return aclrtFree(ptr);
}

inline cudaError_t cudaFreeHost(void* ptr) {
    return aclrtFreeHost(ptr);
}

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t size, int kind) {
    if (!hetkvcache::ascend_compat::ensureAclInitialized()) {
        return cudaErrorMemoryAllocation;
    }
    return aclrtMemcpy(dst, size, src, size, static_cast<aclrtMemcpyKind>(kind));
}

inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t size, int kind, void* stream) {
    if (!hetkvcache::ascend_compat::ensureAclInitialized()) {
        return cudaErrorMemoryAllocation;
    }
    return aclrtMemcpyAsync(dst, size, src, size, static_cast<aclrtMemcpyKind>(kind), stream);
}

inline cudaError_t cudaMemset(void* ptr, int value, size_t size) {
    if (!hetkvcache::ascend_compat::ensureAclInitialized()) {
        return cudaErrorMemoryAllocation;
    }
    return aclrtMemset(ptr, size, value, size);
}

inline cudaError_t cudaMemsetAsync(void* ptr, int value, size_t size, void* stream) {
    (void)stream;
    if (!hetkvcache::ascend_compat::ensureAclInitialized()) {
        return cudaErrorMemoryAllocation;
    }
    return aclrtMemset(ptr, size, value, size);
}

inline const char* cudaGetErrorString(int error) {
    static char err_buf[256];
    snprintf(err_buf, sizeof(err_buf), "ACL Error: %d", error);
    return err_buf;
}

// Stream相关
inline int cudaStreamCreate(void** stream) {
    if (!hetkvcache::ascend_compat::ensureAclInitialized()) {
        return cudaErrorMemoryAllocation;
    }
    return aclrtCreateStream(reinterpret_cast<aclrtStream*>(stream));
}

inline int cudaStreamSynchronize(void* stream) {
    if (!hetkvcache::ascend_compat::ensureAclInitialized()) {
        return cudaErrorMemoryAllocation;
    }
    return aclrtSynchronizeStream(stream);
}

inline int cudaStreamDestroy(void* stream) {
    return aclrtDestroyStream(stream);
}

inline int cudaDeviceSynchronize() {
    if (!hetkvcache::ascend_compat::ensureAclInitialized()) {
        return cudaErrorMemoryAllocation;
    }
    return aclrtSynchronizeDevice();
}

// 清理ACL运行时
inline int cudaCleanup() {
    auto& initialized = hetkvcache::ascend_compat::getAclInitialized();
    if (initialized.load()) {
        aclFinalize();
        initialized.store(false);
    }
    return cudaSuccess;
}

#else
// CUDA模式 (或未定义)
#include <cuda_runtime.h>

#endif // USE_ASCEND

#endif // HETKVCACHE_CUDA_COMPAT_H