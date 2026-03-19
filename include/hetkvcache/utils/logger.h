/**
 * @file logger.h
 * @brief 日志工具头文件
 */

#ifndef HETKVCACHE_LOGGER_H
#define HETKVCACHE_LOGGER_H

#include <string>
#include <fstream>
#include <mutex>
#include <cstdarg>

namespace hetkvcache {

/**
 * @brief 日志级别
 */
enum class LogLevel {
    OFF = 0,
    ERROR = 1,
    WARN = 2,
    INFO = 3,
    DEBUG = 4
};

/**
 * @brief 日志工具类（单例）
 */
class Logger {
public:
    static Logger& getInstance();
    
    void setLogLevel(int level);
    void setLogFile(const std::string& filepath);
    
    void log(int level, const char* file, int line, const char* format, ...);
    
    // 便捷宏
    #define LOG_ERROR(...) Logger::getInstance().log(1, __FILE__, __LINE__, __VA_ARGS__)
    #define LOG_WARN(...) Logger::getInstance().log(2, __FILE__, __LINE__, __VA_ARGS__)
    #define LOG_INFO(...) Logger::getInstance().log(3, __FILE__, __LINE__, __VA_ARGS__)
    #define LOG_DEBUG(...) Logger::getInstance().log(4, __FILE__, __LINE__, __VA_ARGS__)

private:
    Logger();
    ~Logger();
    
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    int level_;
    std::ofstream file_stream_;
    std::mutex mutex_;
};

}  // namespace hetkvcache

#endif  // HETKVCACHE_LOGGER_H