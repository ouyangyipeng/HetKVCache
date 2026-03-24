/**
 * @file device_interface.h
 * @brief 设备抽象接口 - 支持CUDA和昇腾NPU后端切换
 */

#ifndef HETKVCACHE_DEVICE_INTERFACE_H
#define HETKVCACHE_DEVICE_INTERFACE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <memory>
#include <vector>

namespace hetkvcache {

/**
 * @brief 内存拷贝类型
 */
enum class MemcpyKind {
    HostToDevice,    ///< 主机到设备
    DeviceToHost,    ///< 设备到主机
    DeviceToDevice   ///< 设备到设备
};

/**
 * @brief 设备类型
 */
enum class DeviceType {
    CUDA,    ///< NVIDIA CUDA
    ASCEND,  ///< 华为昇腾
    CPU      ///< CPU fallback
};

/**
 * @brief 设备信息
 */
struct DeviceInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
};

/**
 * @brief 设备抽象接口
 * 
 * 提供统一的设备操作接口，支持CUDA和昇腾NPU后端切换。
 */
class DeviceInterface {
public:
    virtual ~DeviceInterface() = default;
    
    // ========================================================================
    // 初始化与清理
    // ========================================================================
    
    /**
     * @brief 初始化设备
     * @return 成功返回true
     */
    virtual bool initialize() = 0;
    
    /**
     * @brief 关闭设备
     */
    virtual void finalize() = 0;
    
    /**
     * @brief 获取设备类型
     */
    virtual DeviceType getDeviceType() const = 0;
    
    /**
     * @brief 获取设备类型名称
     */
    virtual std::string getDeviceTypeName() const = 0;
    
    // ========================================================================
    // 设备管理
    // ========================================================================
    
    /**
     * @brief 获取设备数量
     */
    virtual int getDeviceCount() = 0;
    
    /**
     * @brief 设置当前设备
     */
    virtual bool setDevice(int device_id) = 0;
    
    /**
     * @brief 获取当前设备ID
     */
    virtual int getCurrentDevice() = 0;
    
    /**
     * @brief 获取设备信息
     */
    virtual bool getDeviceInfo(int device_id, DeviceInfo& info) = 0;
    
    // ========================================================================
    // 内存管理
    // ========================================================================
    
    /**
     * @brief 分配设备内存
     */
    virtual bool allocateDeviceMemory(void** ptr, size_t size) = 0;
    
    /**
     * @brief 释放设备内存
     */
    virtual void freeDeviceMemory(void* ptr) = 0;
    
    /**
     * @brief 分配锁页内存（主机端）
     */
    virtual bool allocatePinnedMemory(void** ptr, size_t size) = 0;
    
    /**
     * @brief 释放锁页内存
     */
    virtual void freePinnedMemory(void* ptr) = 0;
    
    /**
     * @brief 获取设备内存使用情况
     */
    virtual bool getMemoryInfo(size_t& free, size_t& total) = 0;
    
    // ========================================================================
    // 内存传输
    // ========================================================================
    
    /**
     * @brief 同步内存拷贝
     */
    virtual bool memcpy(void* dst, const void* src, size_t size, 
                       MemcpyKind kind) = 0;
    
    /**
     * @brief 异步内存拷贝
     */
    virtual bool memcpyAsync(void* dst, const void* src, size_t size,
                            MemcpyKind kind, void* stream) = 0;
    
    /**
     * @brief 批量异步内存拷贝
     */
    virtual bool memcpyBatchAsync(void** dsts, const void** srcs, 
                                  size_t* sizes, int count,
                                  MemcpyKind kind, void* stream) = 0;
    
    /**
     * @brief 设置设备内存
     */
    virtual bool memset(void* ptr, int value, size_t size) = 0;
    
    /**
     * @brief 异步设置设备内存
     */
    virtual bool memsetAsync(void* ptr, int value, size_t size, void* stream) = 0;
    
    // ========================================================================
    // 流管理
    // ========================================================================
    
    /**
     * @brief 创建流
     */
    virtual bool createStream(void** stream) = 0;
    
    /**
     * @brief 销毁流
     */
    virtual void destroyStream(void* stream) = 0;
    
    /**
     * @brief 同步流
     */
    virtual bool synchronizeStream(void* stream) = 0;
    
    /**
     * @brief 同步设备
     */
    virtual bool synchronizeDevice() = 0;
    
    // ========================================================================
    // 事件管理（可选）
    // ========================================================================
    
    /**
     * @brief 创建事件
     */
    virtual bool createEvent(void** event) = 0;
    
    /**
     * @brief 销毁事件
     */
    virtual void destroyEvent(void* event) = 0;
    
    /**
     * @brief 记录事件
     */
    virtual bool recordEvent(void* event, void* stream) = 0;
    
    /**
     * @brief 等待事件
     */
    virtual bool waitForEvent(void* event, void* stream) = 0;
    
    /**
     * @brief 计算事件经过时间（毫秒）
     */
    virtual bool getEventElapsedTime(void* start_event, void* end_event, 
                                    float& elapsed_ms) = 0;
    
    // ========================================================================
    // 错误处理
    // ========================================================================
    
    /**
     * @brief 获取最后的错误信息
     */
    virtual std::string getLastError() = 0;
    
    /**
     * @brief 清除错误状态
     */
    virtual void clearLastError() = 0;
    
    // ========================================================================
    // 多卡负载均衡（可选实现）
    // ========================================================================
    
    /**
     * @brief 获取最优设备ID（负载均衡）
     * @return 最优设备ID，失败返回0
     */
    virtual int getOptimalDevice() { return 0; }
    
    /**
     * @brief 获取设备负载
     * @param device_id 设备ID
     * @return 设备负载（0.0-1.0），失败返回0.0
     */
    virtual float getDeviceLoad(int device_id) {
        (void)device_id;
        return 0.0f;
    }
    
    /**
     * @brief 设置设备亲和性
     * @param device_ids 设备ID列表
     * @return 成功返回true
     */
    virtual bool setDeviceAffinity(const std::vector<int>& device_ids) {
        (void)device_ids;
        return true;
    }
};

/**
 * @brief 设备工厂类
 */
class DeviceFactory {
public:
    /**
     * @brief 创建设备实例
     * @param type 设备类型
     * @return 设备接口指针
     */
    static std::unique_ptr<DeviceInterface> create(DeviceType type);
    
    /**
     * @brief 自动检测并创建最佳设备
     * @return 设备接口指针
     */
    static std::unique_ptr<DeviceInterface> createAuto();
    
    /**
     * @brief 检测可用设备类型
     */
    static DeviceType detectDeviceType();
    
    /**
     * @brief 检查设备类型是否可用
     */
    static bool isDeviceAvailable(DeviceType type);
    
    /**
     * @brief 获取设备类型名称
     */
    static const char* getDeviceTypeName(DeviceType type);
};

} // namespace hetkvcache

#endif // HETKVCACHE_DEVICE_INTERFACE_H