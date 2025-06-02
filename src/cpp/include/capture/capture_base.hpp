// ================= capture/capture_base.hpp ====================================
#pragma once
#include "util/threadsafe_queue.hpp"
#include "capture/frame.hpp"
#include <thread>
#include <atomic>
namespace zybo::capture {
class CaptureBase {
public:
    using Queue = zybo::util::ThreadSafeQueue<Frame>;
    explicit CaptureBase(Queue& q):q_(q){}
    virtual ~CaptureBase(){stop();}
    void start(){running_=true;thr_=std::thread(&CaptureBase::loop,this);}    
    void stop(){running_=false; if(thr_.joinable())thr_.join();}
protected: virtual bool grab(Frame&)=0; Queue& q_;
private: void loop(){while(running_){Frame f;if(grab(f))q_.push(std::move(f));}}
    std::atomic<bool> running_{false}; std::thread thr_;
}; } // namespace