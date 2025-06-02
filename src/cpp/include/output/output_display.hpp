// ================= output/output_display.hpp ===================================
#pragma once
#include "util/threadsafe_queue.hpp"
#include "capture/frame.hpp"
#include "tracking/sort_tracker.hpp"
#include <thread>
#include <atomic>
namespace zybo::output {
struct DrawPack { zybo::capture::Frame frame; std::vector<zybo::tracking::Track> tracks; };
class OutputDisplay {
public:
    using Queue = zybo::util::ThreadSafeQueue<DrawPack>;
    explicit OutputDisplay(Queue& in_q):q_(in_q){}
    ~OutputDisplay(){stop();}
    void start(){running_=true;thr_=std::thread(&OutputDisplay::loop,this);} void stop(){running_=false;q_.terminate();if(thr_.joinable())thr_.join();}
private:
    void loop(); Queue& q_; std::thread thr_; std::atomic<bool> running_{false}; };
} // namespace