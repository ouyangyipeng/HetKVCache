/**
 * @file logger.cpp
 * @brief 日志工具实现
 */

#include "hetkvcache/utils/logger.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>

namespace hetkvcache {

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

void Logger::setLogLevel(int level) {
    level_ = level;
}

void Logger::setLogFile(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (file_stream_.is_open()) {
        file_stream_.close();
    }
    
    if (!filepath.empty()) {
        file_stream_.open(filepath, std::ios::app);
    }
}

void Logger::log(int level, const char* file, int line, const char* format, ...) {
    if (level > level_) {
        return;
    }
    
    // 获取时间戳
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::tm tm_buf;
    localtime_r(&time_t, &tm_buf);
    
    // 格式化日志级别
    const char* level_str = "";
    switch (level) {
        case 1: level_str = "ERROR"; break;
        case 2: level_str = "WARN"; break;
        case 3: level_str = "INFO"; break;
        case 4: level_str = "DEBUG"; break;
        default: level_str = "UNKNOWN"; break;
    }
    
    // 格式化消息
    char buffer[4096];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    
    // 构建完整日志行
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count();
    oss << " [" << level_str << "] ";
    oss << "[" << file << ":" << line << "] ";
    oss << buffer << "\n";
    
    std::string log_line = oss.str();
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 输出到控制台
    if (level <= 2) {
        std::cerr << log_line;
    } else {
        std::cout << log_line;
    }
    
    // 输出到文件
    if (file_stream_.is_open()) {
        file_stream_ << log_line;
        file_stream_.flush();
    }
}

Logger::Logger()
    : level_(2)  // 默认 WARN 级别
{
}

Logger::~Logger() {
    if (file_stream_.is_open()) {
        file_stream_.close();
    }
}

}  // namespace hetkvcache