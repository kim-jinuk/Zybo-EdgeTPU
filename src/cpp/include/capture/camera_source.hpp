#pragma once
#include <opencv2/opencv.hpp>
#include "utils/safe_queue.hpp"

class CameraSource {
public:
    CameraSource(int cam_id, SafeQueue<cv::Mat>& out, int width=640, int height=480)
        : cap_(cam_id), q_(out) {
        cap_.set(cv::CAP_PROP_FRAME_WIDTH,  width);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    }
    void operator()() {
        while(true) { cv::Mat f; cap_ >> f; if(f.empty()) continue; q_.push(f); }
    }
private:
    cv::VideoCapture cap_;
    SafeQueue<cv::Mat>& q_;
};