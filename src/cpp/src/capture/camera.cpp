// --- src/cpp/src/capture/camera.cpp ---------------------------------
#include "capture/camera.hpp"
#include <chrono>

using namespace zybo::capture;

CameraCapture::CameraCapture(int cam_id, int width, int height, int fps, std::queue<Frame>& out_q)
    : cam_id_{cam_id}, width_{width}, height_{height}, fps_{fps}, q_{out_q} {
    cap_.open(cam_id_);
    if (!cap_.isOpened()) {
        throw std::runtime_error("Camera open failed");
    }
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, width_);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
    cap_.set(cv::CAP_PROP_FPS, fps_);
}

CameraCapture::~CameraCapture() {
    stop();
}

void CameraCapture::start() {
    running_ = true;
    th_ = std::thread(&CameraCapture::run, this);
}

void CameraCapture::stop() {
    running_ = false;
    if (th_.joinable()) th_.join();
}

void CameraCapture::run() {
    const auto frame_period = std::chrono::milliseconds(static_cast<int>(1000 / fps_));
    while (running_) {
        cv::Mat frame;
        if (!cap_.read(frame)) {
            std::this_thread::sleep_for(frame_period);
            continue;
        }
        Frame f{static_cast<double>(cv::getTickCount()) / cv::getTickFrequency(), frame};
        if (q_.size() >= 2) q_.pop();
        q_.push(std::move(f));
        std::this_thread::sleep_for(frame_period);
    }
}