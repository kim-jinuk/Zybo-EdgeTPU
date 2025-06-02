// ================= util/threadsafe_queue.hpp ====================================
#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

namespace zybo::util {

template <typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t max_size = 3) : max_size_(max_size) {}

    void push(T item) {
        std::unique_lock lock(mtx_);
        cond_not_full_.wait(lock, [&]{ return q_.size() < max_size_; });
        q_.push(std::move(item));
        lock.unlock();
        cond_not_empty_.notify_one();
    }

    std::optional<T> pop() {
        std::unique_lock lock(mtx_);
        cond_not_empty_.wait(lock, [&]{ return !q_.empty() || terminated_; });
        if (q_.empty()) return std::nullopt;
        T value = std::move(q_.front());
        q_.pop();
        lock.unlock();
        cond_not_full_.notify_one();
        return value;
    }

    void terminate() {
        std::scoped_lock lock(mtx_);
        terminated_ = true;
        cond_not_empty_.notify_all();
    }
private:
    std::queue<T> q_;
    mutable std::mutex mtx_;
    std::condition_variable cond_not_empty_, cond_not_full_;
    size_t max_size_;
    bool terminated_ = false;
};

} // namespace zybo::util