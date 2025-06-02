#pragma once
#include "utils/safe_queue.hpp"
#include "pipeline/pipeline_thread.hpp"
class OutputThread {
public:
    OutputThread(SafeQueue<FramePack>& in):in_(in){}
    void operator()();
private: SafeQueue<FramePack>& in_; };