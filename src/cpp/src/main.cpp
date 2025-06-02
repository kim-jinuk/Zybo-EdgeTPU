#include "capture/camera_source.hpp"
#include "pipeline/pipeline_thread.hpp"
#include "pipeline/output_thread.hpp"
#include "detection/edgetpu_detector.hpp"
#include "tracking/sort_tracker.hpp"
#include <thread>
int main(int argc,char** argv){
    if(argc<2){ std::cerr<<"usage: ./app model_edgetpu.tflite [cam]"<<std::endl; return 0; }
    EdgeTpuDetector det(argv[1]);
    SortTracker tracker;
    SafeQueue<cv::Mat>   q_cap;      // raw frames
    SafeQueue<FramePack> q_pipe;     // processed frames+tracks

    CameraSource    cap_thread(argc>2?std::stoi(argv[2]):0,q_cap);
    PipelineThread  pipe_thread(q_cap,q_pipe,det,tracker);
    OutputThread    out_thread(q_pipe);

    std::thread t1(cap_thread), t2(pipe_thread), t3(out_thread);
    t1.join(); t2.join(); t3.join();
    return 0;
}