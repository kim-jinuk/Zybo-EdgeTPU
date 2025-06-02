#include "pipeline/pipeline_thread.hpp"
#include <chrono>
PipelineThread::PipelineThread(SafeQueue<cv::Mat>& i,SafeQueue<FramePack>& o,
                               EdgeTpuDetector& d, SortTracker& t):in_(i),out_(o),det_(d),trk_(t){}
void PipelineThread::operator()(){
    auto t0=std::chrono::high_resolution_clock::now();
    while(true){
        cv::Mat f=in_.pop();
        cv::Mat p=preprocess::run(f);
        auto dets=det_.infer(p);
        auto tracks=trk_.update(dets);
        auto t1=std::chrono::high_resolution_clock::now();
        double fps=1e9/std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
        t0=t1;
        out_.push({f,tracks,fps});
    }
}