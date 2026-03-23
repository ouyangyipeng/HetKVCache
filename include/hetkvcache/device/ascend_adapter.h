/**
 * @file ascend_adapter.h
 * @brief 昇腾NPU适配器头文件
 */

#ifndef HETKVCACHE_ASCEND_ADAPTER_H
#define HETKVCACHE_ASCEND_ADAPTER_H

#include "device_interface.h"
#include <mutex>
#include <atomic>

// 包含ACL头文件以使用aclrtMemcpyKind等类型
#include "acl/acl.h"

namespace hetkvcache {

/**
 * @brief 昇腾NPU适配器
 * 
 * 实现DeviceInterface接口，封装华为昇腾ACL API。
 */
class AscendAdapter : public DeviceInterface {
public:
    AscendAdapter();
    ~AscendAdapter() override;
    
    // 禁止拷贝
    AscendAdapter(const AscendAdapter&) = delete;
    AscendAdapter& operator=(const AscendAdapter&) = delete;
    
    // ========================================================================
    // 初始化与清理
    // ========================================================================
    
    bool initialize() override;
    void finalize() override;
    DeviceType getDeviceType() const override { return DeviceType::ASCEND; }
    std::string getDeviceTypeName() const override { return "Ascend NPU"; }
    
    // ========================================================================
    // 设备管理
    // ========================================================================
    
    int getDeviceCount() override;
    bool setDevice(int device_id) override;
    int getCurrentDevice() override;
    bool getDeviceInfo(int device_id, DeviceInfo& info) override;
    
    // ========================================================================
    // 内存管理
    // ========================================================================
    
    bool allocateDeviceMemory(void** ptr, size_t size) override;
    void freeDeviceMemory(void* ptr) override;
    bool allocatePinnedMemory(void** ptr, size_t size) override;
    void freePinnedMemory(void* ptr) override;
    bool getMemoryInfo(size_t& free, size_t& total) override;
    
    // ========================================================================
    // 内存传输
    // ========================================================================
    
    bool memcpy(void* dst, const void* src, size_t size, 
               MemcpyKind kind) override;
    bool memcpyAsync(void* dst, const void* src, size_t size,
                    MemcpyKind kind, void* stream) override;
    bool memcpyBatchAsync(void** dsts, const void** srcs, 
                          size_t* sizes, int count,
                          MemcpyKind kind, void* stream) override;
    bool memset(void* ptr, int value, size_t size) override;
    bool memsetAsync(void* ptr, int value, size_t size, void* stream) override;
    
    // ========================================================================
    // 流管理
    // ========================================================================
    
    bool createStream(void** stream) override;
    void destroyStream(void* stream) override;
    bool synchronizeStream(void* stream) override;
    bool synchronizeDevice() override;
    
    // ========================================================================
    // 事件管理
    // ========================================================================
    
    bool createEvent(void** event) override;
    void destroyEvent(void* event) override;
    bool recordEvent(void* event, void* stream) override;
    bool waitForEvent(void* event, void* stream) override;
    bool getEventElapsedTime(void* start_event, void* end_event, 
                            float& elapsed_ms) override;
    
    // ========================================================================
    // 错误处理
    // ========================================================================
    
    std::string getLastError() override;
    void clearLastError() override;
    
private:
    std::atomic<bool> initialized_;
    std::atomic<int> current_device_;
    std::mutex device_mutex_;
    std::string last_error_;
    int device_count_;
    
    // ACL运行时配置
    static std::mutex global_init_mutex_;
    static std::atomic<int> instance_count_;
    
    /**
     * @brief 转换MemcpyKind到ACL类型
     */
    static aclrtMemcpyKind toAclMemcpyKind(MemcpyKind kind);
    
    /**
     * @brief 设置错误信息
     */
    void setLastError(const std::string& error);
};

} // namespace hetkvcache

#endif // HETKVCACHE_ASCEND_ADAPTER_H