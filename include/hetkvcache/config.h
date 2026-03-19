/**
 * @file config.h
 * @brief HetKVCache 配置管理
 */

#ifndef HETKVCACHE_CONFIG_H
#define HETKVCACHE_CONFIG_H

#include "hetkvcache/types.h"
#include <string>
#include <fstream>
#include <sstream>
#include <map>

namespace hetkvcache {

/**
 * @brief 配置管理器类
 * 
 * 支持从文件、环境变量和命令行参数加载配置。
 */
class ConfigManager {
public:
    static ConfigManager& getInstance() {
        static ConfigManager instance;
        return instance;
    }
    
    /**
     * @brief 从文件加载配置
     */
    bool loadFromFile(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            // 跳过注释和空行
            if (line.empty() || line[0] == '#') continue;
            
            // 解析 key=value
            size_t pos = line.find('=');
            if (pos == std::string::npos) continue;
            
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            
            // 去除空格
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            config_values_[key] = value;
        }
        
        applyToConfig();
        return true;
    }
    
    /**
     * @brief 从环境变量加载配置
     */
    void loadFromEnv() {
        // VRAM 预算
        const char* vram_budget = std::getenv("HETKVCACHE_VRAM_BUDGET_MB");
        if (vram_budget) {
            config_.vram_budget_mb = std::stoull(vram_budget);
        }
        
        // RAM 预算
        const char* ram_budget = std::getenv("HETKVCACHE_RAM_BUDGET_MB");
        if (ram_budget) {
            config_.ram_budget_mb = std::stoull(ram_budget);
        }
        
        // SSD 交换区大小
        const char* ssd_swap = std::getenv("HETKVCACHE_SSD_SWAP_GB");
        if (ssd_swap) {
            config_.ssd_swap_gb = std::stoull(ssd_swap);
        }
        
        // SSD 交换路径
        const char* ssd_path = std::getenv("HETKVCACHE_SSD_SWAP_PATH");
        if (ssd_path) {
            config_.ssd_swap_path = ssd_path;
        }
        
        // 热数据阈值
        const char* hot_thresh = std::getenv("HETKVCACHE_HOT_THRESHOLD");
        if (hot_thresh) {
            config_.hot_threshold = std::stof(hot_thresh);
        }
        
        // 温数据阈值
        const char* warm_thresh = std::getenv("HETKVCACHE_WARM_THRESHOLD");
        if (warm_thresh) {
            config_.warm_threshold = std::stof(warm_thresh);
        }
        
        // 启用预取
        const char* prefetch = std::getenv("HETKVCACHE_ENABLE_PREFETCH");
        if (prefetch) {
            config_.enable_prefetch = (std::string(prefetch) == "1" || 
                                       std::string(prefetch) == "true");
        }
        
        // 启用压缩
        const char* compression = std::getenv("HETKVCACHE_ENABLE_COMPRESSION");
        if (compression) {
            config_.enable_compression = (std::string(compression) == "1" || 
                                          std::string(compression) == "true");
        }
        
        // 块大小
        const char* block_size = std::getenv("HETKVCACHE_BLOCK_SIZE");
        if (block_size) {
            config_.block_size = std::stoull(block_size);
        }
        
        // 日志级别
        const char* log_level = std::getenv("HETKVCACHE_LOG_LEVEL");
        if (log_level) {
            config_.log_level = std::stoi(log_level);
        }
    }
    
    /**
     * @brief 获取当前配置
     */
    const HetKVCacheConfig& getConfig() const {
        return config_;
    }
    
    /**
     * @brief 设置配置
     */
    void setConfig(const HetKVCacheConfig& config) {
        config_ = config;
    }
    
    /**
     * @brief 获取默认配置
     */
    static HetKVCacheConfig getDefaultConfig() {
        return HetKVCacheConfig();
    }
    
    /**
     * @brief 验证配置有效性
     */
    bool validateConfig(const HetKVCacheConfig& config) const {
        if (config.vram_budget_mb == 0) {
            return false;
        }
        if (config.hot_threshold <= config.warm_threshold) {
            return false;
        }
        if (config.hot_threshold > 1.0f || config.warm_threshold < 0.0f) {
            return false;
        }
        if (config.alpha + config.beta + config.gamma > 1.01f ||
            config.alpha + config.beta + config.gamma < 0.99f) {
            return false;
        }
        if (config.block_size == 0 || config.block_size % 1024 != 0) {
            return false;
        }
        return true;
    }
    
    /**
     * @brief 生成配置文件
     */
    void saveToFile(const std::string& filepath) const {
        std::ofstream file(filepath);
        if (!file.is_open()) return;
        
        file << "# HetKVCache Configuration File\n";
        file << "# Generated automatically\n\n";
        
        file << "# Memory Configuration\n";
        file << "vram_budget_mb=" << config_.vram_budget_mb << "\n";
        file << "ram_budget_mb=" << config_.ram_budget_mb << "\n";
        file << "ssd_swap_gb=" << config_.ssd_swap_gb << "\n";
        file << "ssd_swap_path=" << config_.ssd_swap_path << "\n\n";
        
        file << "# Heat Evaluation Configuration\n";
        file << "hot_threshold=" << config_.hot_threshold << "\n";
        file << "warm_threshold=" << config_.warm_threshold << "\n";
        file << "alpha=" << config_.alpha << "\n";
        file << "beta=" << config_.beta << "\n";
        file << "gamma=" << config_.gamma << "\n";
        file << "lambda=" << config_.lambda << "\n\n";
        
        file << "# Migration Configuration\n";
        file << "enable_prefetch=" << (config_.enable_prefetch ? "true" : "false") << "\n";
        file << "enable_compression=" << (config_.enable_compression ? "true" : "false") << "\n";
        file << "max_concurrent_migrations=" << config_.max_concurrent_migrations << "\n";
        file << "migration_batch_size=" << config_.migration_batch_size << "\n\n";
        
        file << "# Allocator Configuration\n";
        file << "block_size=" << config_.block_size << "\n";
        file << "use_pinned_memory=" << (config_.use_pinned_memory ? "true" : "false") << "\n\n";
        
        file << "# Logging Configuration\n";
        file << "log_level=" << config_.log_level << "\n";
        file << "log_file=" << config_.log_file << "\n";
    }

private:
    ConfigManager() : config_() {}
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;
    
    void applyToConfig() {
        if (config_values_.count("vram_budget_mb")) {
            config_.vram_budget_mb = std::stoull(config_values_["vram_budget_mb"]);
        }
        if (config_values_.count("ram_budget_mb")) {
            config_.ram_budget_mb = std::stoull(config_values_["ram_budget_mb"]);
        }
        if (config_values_.count("ssd_swap_gb")) {
            config_.ssd_swap_gb = std::stoull(config_values_["ssd_swap_gb"]);
        }
        if (config_values_.count("ssd_swap_path")) {
            config_.ssd_swap_path = config_values_["ssd_swap_path"];
        }
        if (config_values_.count("hot_threshold")) {
            config_.hot_threshold = std::stof(config_values_["hot_threshold"]);
        }
        if (config_values_.count("warm_threshold")) {
            config_.warm_threshold = std::stof(config_values_["warm_threshold"]);
        }
        if (config_values_.count("alpha")) {
            config_.alpha = std::stof(config_values_["alpha"]);
        }
        if (config_values_.count("beta")) {
            config_.beta = std::stof(config_values_["beta"]);
        }
        if (config_values_.count("gamma")) {
            config_.gamma = std::stof(config_values_["gamma"]);
        }
        if (config_values_.count("lambda")) {
            config_.lambda = std::stof(config_values_["lambda"]);
        }
        if (config_values_.count("enable_prefetch")) {
            config_.enable_prefetch = (config_values_["enable_prefetch"] == "true");
        }
        if (config_values_.count("enable_compression")) {
            config_.enable_compression = (config_values_["enable_compression"] == "true");
        }
        if (config_values_.count("max_concurrent_migrations")) {
            config_.max_concurrent_migrations = std::stoul(config_values_["max_concurrent_migrations"]);
        }
        if (config_values_.count("migration_batch_size")) {
            config_.migration_batch_size = std::stoul(config_values_["migration_batch_size"]);
        }
        if (config_values_.count("block_size")) {
            config_.block_size = std::stoull(config_values_["block_size"]);
        }
        if (config_values_.count("use_pinned_memory")) {
            config_.use_pinned_memory = (config_values_["use_pinned_memory"] == "true");
        }
        if (config_values_.count("log_level")) {
            config_.log_level = std::stoi(config_values_["log_level"]);
        }
        if (config_values_.count("log_file")) {
            config_.log_file = config_values_["log_file"];
        }
    }
    
    HetKVCacheConfig config_;
    std::map<std::string, std::string> config_values_;
};

}  // namespace hetkvcache

#endif  // HETKVCACHE_CONFIG_H