#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

template<typename T>
class SafeQueue {
public:
    void push(const T& v) {
        std::unique_lock<std::mutex> lk(m_);
        q_.push(v);
        cv_.notify_one();
    }
    T pop() {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&]{ return !q_.empty(); });
        T v = q_.front(); q_.pop();
        return v;
    }
private:
    std::queue<T> q_;
    std::mutex m_;
    std::condition_variable cv_;
};