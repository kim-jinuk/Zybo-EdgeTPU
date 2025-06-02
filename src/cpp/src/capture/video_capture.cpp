// ================= capture/video_capture.cpp ===================================
#include "capture/video_capture.hpp"
#include <chrono>

using namespace zybo::capture;

VideoCapture::VideoCapture(const std::string& path, Queue& out_q)
    : CaptureBase(out_q) {
    cap_.open(path);
    if (!cap_.isOpened()) throw std::runtime_error("Video open failed");
    double fps = cap_.get(cv::CAP_PROP_FPS);
    frame_period_ms_ = fps > 1 ? 1000.0 / fps : 33.3;
}

bool VideoCapture::grabFrame(Frame& out) {
    cv::Mat frame;
    if (!cap_.read(frame)) {
        // End of file – terminate outer loop
        return false;
    }
    out.timestamp = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
    out.img = std::move(frame);
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(frame_period_ms_)));
    return true;
}