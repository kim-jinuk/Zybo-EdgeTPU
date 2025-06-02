// ================= pipeline/pipeline.hpp =======================================
#pragma once
#include "util/threadsafe_queue.hpp"
#include "capture/frame.hpp"
#include "capture/camera_capture.hpp"
#include "processing/enhancer.hpp"
#include "detection/tpu_detector.hpp"
#include "tracking/sort_tracker.hpp"
#include "output/output_display.hpp"
#include <thread>
#include <atomic>
namespace zybo::pipeline {
class Pipeline {
public:
    Pipeline(int cam=0,int w=640,int h=480,int fps=30,const std::string& model="models/model.tflite",float thr=0.4f);
    ~Pipeline(); void start(); void stop();
private:
    using FrameQueue = zybo::util::ThreadSafeQueue<zybo::capture::Frame>;
    using DrawQueue  = zybo::util::ThreadSafeQueue<zybo::output::DrawPack>;
    void loop();
    FrameQueue cap_q_{3}; DrawQueue draw_q_{3};
    zybo::capture::CameraCapture cam_;
    zybo::processing::Compose proc_;
    zybo::detection::TPUDetector det_;
    zybo::tracking::SortTracker tracker_;
    zybo::output::OutputDisplay out_{draw_q_};
    std::thread thr_; std::atomic<bool> running_{false}; };
} // namespace