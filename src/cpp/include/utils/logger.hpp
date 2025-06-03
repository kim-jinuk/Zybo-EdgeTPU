#pragma once
#include <iostream>
#include <mutex>
#include <chrono>
#include <iomanip>

class Logger {
public:
    enum Level { INFO, WARN, ERROR };
    static Logger& get() {
        static Logger instance;
        return instance;
    }
    void log(const std::string& msg, Level level = INFO) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto now = std::chrono::system_clock::now();
        auto t_c = std::chrono::system_clock::to_time_t(now);
        std::tm tm;
        localtime_r(&t_c, &tm);
        std::cout << "[" << std::put_time(&tm, "%F %T") << "] ";
        switch(level) {
            case INFO:  std::cout << "[INFO] "; break;
            case WARN:  std::cout << "[WARN] "; break;
            case ERROR: std::cout << "[ERROR] "; break;
        }
        std::cout << msg << std::endl;
    }
private:
    Logger() {}
    std::mutex mtx_;
};
#define LOG(msg)      Logger::get().log(msg, Logger::INFO)
#define LOG_WARN(msg) Logger::get().log(msg, Logger::WARN)
#define LOG_ERR(msg)  Logger::get().log(msg, Logger::ERROR)
