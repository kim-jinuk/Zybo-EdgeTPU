#pragma once
#include <iostream>
#include <chrono>
#include <iomanip>

inline void log(const std::string& tag, const std::string& msg) {
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm = *std::localtime(&now);
    std::cout << "[" << std::put_time(&tm, "%H:%M:%S") << "] [" << tag << "] " << msg << std::endl;
}