/**
 * @file stream_manager.cu
 * @brief CUDA Stream 管理器实现
 */

#include <cuda_runtime.h>
#include <vector>
#include <queue>
#include <mutex>
#include <atomic>
#include <memory>

namespace hetkvcache {

/**
 * @brief CUDA Stream 管理器类
 */
class StreamManager {
public:
    StreamManager(size_t pool_size = 8)
        : pool_size_(pool_size)
        , initialized_(false) {}
    
    ~StreamManager() {
        cleanup();
    }
    
    // 禁止拷贝
    StreamManager(const StreamManager&) = delete;
    StreamManager& operator=(const StreamManager&) = delete;
    
    bool initialize() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (initialized_) {
            return true;
        }
        
        for (size_t i = 0; i < pool_size_; i++) {
            cudaStream_t stream;
            cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            if (err != cudaSuccess) {
                cleanup();
                return false;
            }
            streams_.push_back(stream);
            available_.push(stream);
        }
        
        initialized_ = true;
        return true;
    }
    
    void cleanup() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (auto stream : streams_) {
            cudaStreamDestroy(stream);
        }
        streams_.clear();
        
        while (!available_.empty()) {
            available_.pop();
        }
        
        initialized_ = false;
    }
    
    cudaStream_t acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (available_.empty()) {
            // 创建临时 stream
            cudaStream_t stream;
            cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            if (err != cudaSuccess) {
                return nullptr;
            }
            return stream;
        }
        
        cudaStream_t stream = available_.front();
        available_.pop();
        return stream;
    }
    
    void release(cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // 检查是否属于池
        for (auto s : streams_) {
            if (s == stream) {
                available_.push(stream);
                return;
            }
        }
        
        // 不属于池，销毁
        cudaStreamDestroy(stream);
    }
    
    bool synchronize(cudaStream_t stream) {
        return cudaStreamSynchronize(stream) == cudaSuccess;
    }
    
    bool synchronizeAll() {
        return cudaDeviceSynchronize() == cudaSuccess;
    }
    
    size_t getPoolSize() const {
        return pool_size_;
    }
    
    size_t getAvailableCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return available_.size();
    }

private:
    size_t pool_size_;
    std::vector<cudaStream_t> streams_;
    std::queue<cudaStream_t> available_;
    mutable std::mutex mutex_;
    bool initialized_;
};

// 全局 Stream 管理器实例
static std::unique_ptr<StreamManager> g_stream_manager;
static std::atomic<bool> g_initialized{false};

// ============================================================================
// C 接口
// ============================================================================

extern "C" {

bool stream_manager_initialize(size_t pool_size) {
    if (g_initialized.load()) {
        return true;
    }
    
    g_stream_manager = std::make_unique<StreamManager>(pool_size);
    if (g_stream_manager->initialize()) {
        g_initialized = true;
        return true;
    }
    g_stream_manager.reset();
    return false;
}

void stream_manager_cleanup() {
    if (g_stream_manager) {
        g_stream_manager->cleanup();
    }
    g_stream_manager.reset();
    g_initialized = false;
}

cudaStream_t stream_acquire() {
    if (!g_initialized.load() || !g_stream_manager) {
        return nullptr;
    }
    return g_stream_manager->acquire();
}

void stream_release(cudaStream_t stream) {
    if (g_stream_manager) {
        g_stream_manager->release(stream);
    }
}

bool stream_synchronize(cudaStream_t stream) {
    if (!g_stream_manager) {
        return false;
    }
    return g_stream_manager->synchronize(stream);
}

bool stream_synchronize_all() {
    if (!g_stream_manager) {
        return false;
    }
    return g_stream_manager->synchronizeAll();
}

size_t stream_get_pool_size() {
    if (!g_stream_manager) {
        return 0;
    }
    return g_stream_manager->getPoolSize();
}

size_t stream_get_available_count() {
    if (!g_stream_manager) {
        return 0;
    }
    return g_stream_manager->getAvailableCount();
}

}  // extern "C"

}  // namespace hetkvcache