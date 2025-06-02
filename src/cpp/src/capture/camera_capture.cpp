// ================= capture/camera_capture.cpp ==================================
#include "capture/camera_capture.hpp"
#include <chrono>

using namespace zybo::capture;

CameraCapture::CameraCapture(int cam_id, int width, int height, int fps, Queue& out_q)
    : CaptureBase(out_q) {
    cap_.open(cam_id, cv::CAP_V4L2);
    if (!cap_.isOpened()) throw std::runtime_error("Camera open failed");
    cap_.set(cv::CAP_PROP_FRAME_WIDTH,  width);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    cap_.set(cv::CAP_PROP_FPS,          fps);
    frame_period_ms_ = 1000.0 / fps;
}

bool CameraCapture::grabFrame(Frame& out) {
    cv::Mat frame;
    if (!cap_.read(frame)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(frame_period_ms_)));
        return false;
    }
    out.timestamp = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
    out.img = std::move(frame);
    return true;
}