/**
 * @file timer.h
 * @brief 计时器工具
 */

#ifndef HETKVCACHE_TIMER_H
#define HETKVCACHE_TIMER_H

#include <chrono>
#include <string>

namespace hetkvcache {

/**
 * @brief 高精度计时器
 */
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    /**
     * @brief 重置计时器
     */
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    /**
     * @brief 获取经过的纳秒数
     */
    int64_t elapsed_ns() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_).count();
    }
    
    /**
     * @brief 获取经过的微秒数
     */
    double elapsed_us() const {
        return static_cast<double>(elapsed_ns()) / 1000.0;
    }
    
    /**
     * @brief 获取经过的毫秒数
     */
    double elapsed_ms() const {
        return elapsed_us() / 1000.0;
    }
    
    /**
     * @brief 获取经过的秒数
     */
    double elapsed_s() const {
        return elapsed_ms() / 1000.0;
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_;
};

/**
 * @brief RAII 风格的计时器
 */
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name, double* result = nullptr)
        : name_(name), result_(result) {}
    
    ~ScopedTimer() {
        double ms = timer_.elapsed_ms();
        if (result_) {
            *result_ = ms;
        }
    }
    
private:
    std::string name_;
    double* result_;
    Timer timer_;
};

}  // namespace hetkvcache

#endif  // HETKVCACHE_TIMER_H