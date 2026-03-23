/**
 * @file device_factory.cpp
 * @brief 设备工厂实现
 */

#include "hetkvcache/device/device_interface.h"
#include "hetkvcache/device/ascend_adapter.h"
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace hetkvcache {

// CPU Fallback适配器（用于无GPU/NPU环境）
class CPUAdapter : public DeviceInterface {
public:
    CPUAdapter() : initialized_(false), current_device_(0) {}
    
    bool initialize() override {
        initialized_ = true;
        return true;
    }
    
    void finalize() override {
        initialized_ = false;
    }
    
    DeviceType getDeviceType() const override { return DeviceType::CPU; }
    std::string getDeviceTypeName() const override { return "CPU Fallback"; }
    
    int getDeviceCount() override { return 1; }
    bool setDevice(int device_id) override { 
        current_device_ = device_id; 
        return true; 
    }
    int getCurrentDevice() override { return current_device_; }
    bool getDeviceInfo(int device_id, DeviceInfo& info) override {
        info.device_id = device_id;
        info.name = "CPU Fallback";
        info.total_memory = 128ULL * 1024 * 1024 * 1024;  // 128GB
        info.free_memory = info.total_memory;
        info.compute_capability_major = 0;
        info.compute_capability_minor = 0;
        return true;
    }
    
    bool allocateDeviceMemory(void** ptr, size_t size) override {
        *ptr = std::malloc(size);
        return *ptr != nullptr;
    }
    
    void freeDeviceMemory(void* ptr) override {
        std::free(ptr);
    }
    
    bool allocatePinnedMemory(void** ptr, size_t size) override {
        *ptr = std::malloc(size);
        return *ptr != nullptr;
    }
    
    void freePinnedMemory(void* ptr) override {
        std::free(ptr);
    }
    
    bool getMemoryInfo(size_t& free, size_t& total) override {
        total = 128ULL * 1024 * 1024 * 1024;
        free = total;
        return true;
    }
    
    bool memcpy(void* dst, const void* src, size_t size, 
               MemcpyKind kind) override {
        (void)kind;
        std::memcpy(dst, src, size);
        return true;
    }
    
    bool memcpyAsync(void* dst, const void* src, size_t size,
                    MemcpyKind kind, void* stream) override {
        (void)stream;
        return memcpy(dst, src, size, kind);
    }
    
    bool memcpyBatchAsync(void** dsts, const void** srcs, 
                          size_t* sizes, int count,
                          MemcpyKind kind, void* stream) override {
        for (int i = 0; i < count; i++) {
            if (!memcpy(dsts[i], srcs[i], sizes[i], kind)) {
                return false;
            }
        }
        (void)stream;
        return true;
    }
    
    bool memset(void* ptr, int value, size_t size) override {
        std::memset(ptr, value, size);
        return true;
    }
    
    bool memsetAsync(void* ptr, int value, size_t size, void* stream) override {
        (void)stream;
        return memset(ptr, value, size);
    }
    
    bool createStream(void** stream) override {
        *stream = reinterpret_cast<void*>(1);  // Dummy stream
        return true;
    }
    
    void destroyStream(void* stream) override {
        (void)stream;
    }
    
    bool synchronizeStream(void* stream) override {
        (void)stream;
        return true;
    }
    
    bool synchronizeDevice() override { return true; }
    
    bool createEvent(void** event) override {
        *event = reinterpret_cast<void*>(1);  // Dummy event
        return true;
    }
    
    void destroyEvent(void* event) override {
        (void)event;
    }
    
    bool recordEvent(void* event, void* stream) override {
        (void)event;
        (void)stream;
        return true;
    }
    
    bool waitForEvent(void* event, void* stream) override {
        (void)event;
        (void)stream;
        return true;
    }
    
    bool getEventElapsedTime(void* start_event, void* end_event, 
                            float& elapsed_ms) override {
        (void)start_event;
        (void)end_event;
        elapsed_ms = 0.0f;
        return true;
    }
    
    std::string getLastError() override { return last_error_; }
    void clearLastError() override { last_error_.clear(); }

private:
    bool initialized_;
    int current_device_;
    std::string last_error_;
};

// ============================================================================
// 设备工厂实现
// ============================================================================

std::unique_ptr<DeviceInterface> DeviceFactory::create(DeviceType type) {
    switch (type) {
        case DeviceType::ASCEND:
            return std::make_unique<AscendAdapter>();
        case DeviceType::CPU:
            return std::make_unique<CPUAdapter>();
        default:
            return std::make_unique<CPUAdapter>();
    }
}

std::unique_ptr<DeviceInterface> DeviceFactory::createAuto() {
    DeviceType type = detectDeviceType();
    return create(type);
}

DeviceType DeviceFactory::detectDeviceType() {
    // 优先检测昇腾NPU
    if (isDeviceAvailable(DeviceType::ASCEND)) {
        return DeviceType::ASCEND;
    }
    
    // 默认使用CPU fallback
    return DeviceType::CPU;
}

bool DeviceFactory::isDeviceAvailable(DeviceType type) {
    switch (type) {
        case DeviceType::ASCEND: {
            // 检查昇腾设备
            // 方法1: 检查npu-smi是否存在且可执行
            if (std::system("which npu-smi > /dev/null 2>&1") == 0) {
                // 方法2: 检查CANN库是否存在
                const char* ascend_path = std::getenv("ASCEND_HOME_PATH");
                if (ascend_path != nullptr) {
                    return true;
                }
                // 检查默认路径
                std::ifstream acl_header("/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl.h");
                if (acl_header.good()) {
                    return true;
                }
            }
            return false;
        }
        case DeviceType::CPU:
            return true;
        default:
            return false;
    }
}

} // namespace hetkvcache