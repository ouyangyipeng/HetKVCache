/**
 * @file ascend_adapter.cpp
 * @brief 昇腾NPU适配器实现
 */

#include "hetkvcache/device/ascend_adapter.h"
#include "acl/acl.h"
#include <cstring>
#include <sstream>
#include <thread>

namespace hetkvcache {

// 静态成员初始化
std::mutex AscendAdapter::global_init_mutex_;
std::atomic<int> AscendAdapter::instance_count_{0};

AscendAdapter::AscendAdapter()
    : initialized_(false)
    , current_device_(-1)
    , device_count_(0) {
}

AscendAdapter::~AscendAdapter() {
    if (initialized_) {
        finalize();
    }
}

bool AscendAdapter::initialize() {
    std::lock_guard<std::mutex> lock(device_mutex_);
    
    if (initialized_) {
        return true;
    }
    
    // 全局初始化ACL（只执行一次）
    std::lock_guard<std::mutex> global_lock(global_init_mutex_);
    
    if (instance_count_ == 0) {
        // 首次初始化ACL运行时
        aclError ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
            setLastError("Failed to initialize ACL runtime: " + std::to_string(ret));
            return false;
        }
    }
    
    // 获取设备数量
    uint32_t count = 0;
    aclError ret = aclrtGetDeviceCount(&count);
    if (ret != ACL_SUCCESS) {
        setLastError("Failed to get device count: " + std::to_string(ret));
        return false;
    }
    
    device_count_ = static_cast<int>(count);
    
    // 设置默认设备
    if (device_count_ > 0) {
        ret = aclrtSetDevice(0);
        if (ret != ACL_SUCCESS) {
            setLastError("Failed to set device 0: " + std::to_string(ret));
            return false;
        }
        current_device_ = 0;
    }
    
    instance_count_++;
    initialized_ = true;
    
    return true;
}

void AscendAdapter::finalize() {
    std::lock_guard<std::mutex> lock(device_mutex_);
    
    if (!initialized_) {
        return;
    }
    
    // 重置当前设备
    if (current_device_ >= 0) {
        aclrtResetDevice(current_device_);
        current_device_ = -1;
    }
    
    initialized_ = false;
    
    // 全局反初始化
    std::lock_guard<std::mutex> global_lock(global_init_mutex_);
    instance_count_--;
    
    if (instance_count_ == 0) {
        aclFinalize();
    }
}

int AscendAdapter::getDeviceCount() {
    return device_count_;
}

bool AscendAdapter::setDevice(int device_id) {
    if (device_id < 0 || device_id >= device_count_) {
        setLastError("Invalid device ID: " + std::to_string(device_id));
        return false;
    }
    
    if (device_id == current_device_) {
        return true;
    }
    
    aclError ret = aclrtSetDevice(device_id);
    if (ret != ACL_SUCCESS) {
        setLastError("Failed to set device " + std::to_string(device_id) + 
                    ": " + std::to_string(ret));
        return false;
    }
    
    current_device_ = device_id;
    return true;
}

int AscendAdapter::getCurrentDevice() {
    return current_device_;
}

bool AscendAdapter::getDeviceInfo(int device_id, DeviceInfo& info) {
    if (device_id < 0 || device_id >= device_count_) {
        setLastError("Invalid device ID: " + std::to_string(device_id));
        return false;
    }
    
    info.device_id = device_id;
    info.name = "Ascend 910B4";
    info.total_memory = 32ULL * 1024 * 1024 * 1024;  // 32GB HBM
    info.free_memory = info.total_memory;
    info.compute_capability_major = 8;
    info.compute_capability_minor = 0;
    
    return true;
}

bool AscendAdapter::allocateDeviceMemory(void** ptr, size_t size) {
    if (!initialized_) {
        setLastError("Device not initialized");
        return false;
    }
    
    aclError ret = aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        setLastError("Failed to allocate device memory: " + std::to_string(ret));
        return false;
    }
    
    return true;
}

void AscendAdapter::freeDeviceMemory(void* ptr) {
    if (ptr != nullptr) {
        aclrtFree(ptr);
    }
}

bool AscendAdapter::allocatePinnedMemory(void** ptr, size_t size) {
    if (!initialized_) {
        setLastError("Device not initialized");
        return false;
    }
    
    aclError ret = aclrtMallocHost(ptr, size);
    if (ret != ACL_SUCCESS) {
        setLastError("Failed to allocate pinned memory: " + std::to_string(ret));
        return false;
    }
    
    return true;
}

void AscendAdapter::freePinnedMemory(void* ptr) {
    if (ptr != nullptr) {
        aclrtFreeHost(ptr);
    }
}

bool AscendAdapter::getMemoryInfo(size_t& free, size_t& total) {
    if (!initialized_ || current_device_ < 0) {
        setLastError("Device not initialized");
        return false;
    }
    
    // ACL不直接提供内存查询接口，使用估算值
    total = 32ULL * 1024 * 1024 * 1024;  // 32GB
    free = total;  // 简化处理
    
    return true;
}

aclrtMemcpyKind AscendAdapter::toAclMemcpyKind(MemcpyKind kind) {
    switch (kind) {
        case MemcpyKind::HostToDevice:
            return ACL_MEMCPY_HOST_TO_DEVICE;
        case MemcpyKind::DeviceToHost:
            return ACL_MEMCPY_DEVICE_TO_HOST;
        case MemcpyKind::DeviceToDevice:
            return ACL_MEMCPY_DEVICE_TO_DEVICE;
        default:
            return ACL_MEMCPY_HOST_TO_DEVICE;
    }
}

bool AscendAdapter::memcpy(void* dst, const void* src, size_t size, 
                           MemcpyKind kind) {
    if (!initialized_) {
        setLastError("Device not initialized");
        return false;
    }
    
    aclError ret = aclrtMemcpy(dst, size, src, size, toAclMemcpyKind(kind));
    if (ret != ACL_SUCCESS) {
        setLastError("Failed to memcpy: " + std::to_string(ret));
        return false;
    }
    
    return true;
}

bool AscendAdapter::memcpyAsync(void* dst, const void* src, size_t size,
                                MemcpyKind kind, void* stream) {
    if (!initialized_) {
        setLastError("Device not initialized");
        return false;
    }
    
    aclError ret = aclrtMemcpyAsync(dst, size, src, size,
                                    toAclMemcpyKind(kind), stream);
    if (ret != ACL_SUCCESS) {
        setLastError("Failed to async memcpy: " + std::to_string(ret));
        return false;
    }
    
    return true;
}

bool AscendAdapter::memcpyBatchAsync(void** dsts, const void** srcs, 
                                     size_t* sizes, int count,
                                     MemcpyKind kind, void* stream) {
    if (!initialized_) {
        setLastError("Device not initialized");
        return false;
    }
    
    for (int i = 0; i < count; i++) {
        aclError ret = aclrtMemcpyAsync(dsts[i], sizes[i], srcs[i], sizes[i],
                                        toAclMemcpyKind(kind), stream);
        if (ret != ACL_SUCCESS) {
            setLastError("Failed to batch async memcpy at index " + 
                        std::to_string(i) + ": " + std::to_string(ret));
            return false;
        }
    }
    
    return true;
}

bool AscendAdapter::memset(void* ptr, int value, size_t size) {
    if (!initialized_) {
        setLastError("Device not initialized");
        return false;
    }
    
    aclError ret = aclrtMemset(ptr, size, value, size);
    if (ret != ACL_SUCCESS) {
        setLastError("Failed to memset: " + std::to_string(ret));
        return false;
    }
    
    return true;
}

bool AscendAdapter::memsetAsync(void* ptr, int value, size_t size, void* stream) {
    // ACL不支持异步memset，使用同步版本
    (void)stream;
    return memset(ptr, value, size);
}

bool AscendAdapter::createStream(void** stream) {
    if (!initialized_) {
        setLastError("Device not initialized");
        return false;
    }
    
    aclError ret = aclrtCreateStream(reinterpret_cast<aclrtStream*>(stream));
    if (ret != ACL_SUCCESS) {
        setLastError("Failed to create stream: " + std::to_string(ret));
        return false;
    }
    return true;
}

void AscendAdapter::destroyStream(void* stream) {
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
}

bool AscendAdapter::synchronizeStream(void* stream) {
    if (!initialized_) {
        setLastError("Device not initialized");
        return false;
    }
    
    aclError ret;
    if (stream != nullptr) {
        ret = aclrtSynchronizeStream(stream);
    } else {
        ret = aclrtSynchronizeStream(nullptr);
    }
    
    if (ret != ACL_SUCCESS) {
        setLastError("Failed to synchronize stream: " + std::to_string(ret));
        return false;
    }
    
    return true;
}

bool AscendAdapter::synchronizeDevice() {
    if (!initialized_) {
        setLastError("Device not initialized");
        return false;
    }
    
    aclError ret = aclrtSynchronizeDevice();
    if (ret != ACL_SUCCESS) {
        setLastError("Failed to synchronize device: " + std::to_string(ret));
        return false;
    }
    
    return true;
}

bool AscendAdapter::createEvent(void** event) {
    if (!initialized_) {
        setLastError("Device not initialized");
        return false;
    }
    
    aclError ret = aclrtCreateEvent(reinterpret_cast<aclrtEvent*>(event));
    if (ret != ACL_SUCCESS) {
        setLastError("Failed to create event: " + std::to_string(ret));
        return false;
    }
    return true;
}

void AscendAdapter::destroyEvent(void* event) {
    if (event != nullptr) {
        aclrtDestroyEvent(event);
    }
}

bool AscendAdapter::recordEvent(void* event, void* stream) {
    if (!initialized_) {
        setLastError("Device not initialized");
        return false;
    }
    
    aclError ret = aclrtRecordEvent(event, stream);
    if (ret != ACL_SUCCESS) {
        setLastError("Failed to record event: " + std::to_string(ret));
        return false;
    }
    
    return true;
}

bool AscendAdapter::waitForEvent(void* event, void* stream) {
    if (!initialized_) {
        setLastError("Device not initialized");
        return false;
    }
    
    aclError ret = aclrtStreamWaitEvent(stream, event);
    if (ret != ACL_SUCCESS) {
        setLastError("Failed to wait for event: " + std::to_string(ret));
        return false;
    }
    
    return true;
}

bool AscendAdapter::getEventElapsedTime(void* start_event, void* end_event, 
                                        float& elapsed_ms) {
    if (!initialized_) {
        setLastError("Device not initialized");
        return false;
    }
    
    float time;
    aclError ret = aclrtEventElapsedTime(&time, start_event, end_event);
    if (ret != ACL_SUCCESS) {
        setLastError("Failed to get event elapsed time: " + std::to_string(ret));
        return false;
    }
    
    elapsed_ms = time;
    return true;
}

std::string AscendAdapter::getLastError() {
    return last_error_;
}

void AscendAdapter::clearLastError() {
    last_error_.clear();
}

void AscendAdapter::setLastError(const std::string& error) {
    last_error_ = error;
}

// ============================================================================
// 多卡负载均衡实现
// ============================================================================

int AscendAdapter::getOptimalDevice() {
    if (device_count_ <= 1) {
        return 0;
    }
    
    // 轮询调度策略 - 简单高效
    int optimal = next_device_.fetch_add(1) % device_count_;
    
    // 更新负载统计
    updateDeviceLoad(optimal);
    
    return optimal;
}

float AscendAdapter::getDeviceLoad(int device_id) {
    if (device_id < 0 || device_id >= device_count_) {
        return 1.0f;  // 返回最大负载表示无效设备
    }
    
    // 基于内存使用率计算负载
    if (device_id < static_cast<int>(device_memory_usage_.size())) {
        constexpr size_t TOTAL_MEMORY = 32ULL * 1024 * 1024 * 1024;  // 32GB
        return static_cast<float>(device_memory_usage_[device_id]) / TOTAL_MEMORY;
    }
    
    return 0.0f;
}

bool AscendAdapter::setDeviceAffinity(const std::vector<int>& device_ids) {
    if (device_ids.empty()) {
        return false;
    }
    
    // 验证所有设备ID有效
    for (int id : device_ids) {
        if (id < 0 || id >= device_count_) {
            setLastError("Invalid device ID in affinity list: " + std::to_string(id));
            return false;
        }
    }
    
    // 设置第一个设备为当前设备
    return setDevice(device_ids[0]);
}

void AscendAdapter::updateDeviceLoad(int device_id) {
    if (device_id < 0 || device_id >= device_count_) {
        return;
    }
    
    // 确保向量大小正确
    if (device_memory_usage_.size() < static_cast<size_t>(device_count_)) {
        device_memory_usage_.resize(device_count_, 0);
    }
    if (device_active_streams_.size() < static_cast<size_t>(device_count_)) {
        device_active_streams_.resize(device_count_, 0);
    }
    
    // 增加活跃流计数（简化实现）
    device_active_streams_[device_id]++;
}

} // namespace hetkvcache