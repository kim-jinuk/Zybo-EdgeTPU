#pragma once
#include <chrono>

class FpsCounter {
public:
    FpsCounter() : last_(std::chrono::steady_clock::now()), fps_(0) {}
    float update() {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_).count();
        last_ = now;
        float fps_now = (dt > 0) ? 1.0f / dt : 0;
        fps_ = 0.9f * fps_ + 0.1f * fps_now; // 더 부드러운 EMA
        return fps_;
    }
private:
    std::chrono::steady_clock::time_point last_;
    float fps_;
};
