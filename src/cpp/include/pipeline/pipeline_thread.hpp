#pragma once
#include "utils/safe_queue.hpp"
#include "processing/preprocess.hpp"
#include "detection/edgetpu_detector.hpp"
#include "tracking/sort_tracker.hpp"
struct FramePack { cv::Mat frame; std::vector<Track> tracks; double fps; };
class PipelineThread {
public:
    PipelineThread(SafeQueue<cv::Mat>& in, SafeQueue<FramePack>& out,
                   EdgeTpuDetector& det, SortTracker& trk);
    void operator()();
private:
    SafeQueue<cv::Mat>& in_; SafeQueue<FramePack>& out_;
    EdgeTpuDetector& det_; SortTracker& trk_;
};