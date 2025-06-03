#include "capture/camera_source.hpp"
#include "utils/logger.hpp"

CameraSource::CameraSource(int device, int w, int h)
    : running_(true) {
    cap_.open(device);
    if (w > 0 && h > 0) {
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, w);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, h);
    }
    if (!cap_.isOpened())
        throw std::runtime_error("Failed to open camera!");
    th_ = std::thread(&CameraSource::capture_loop, this);
}
CameraSource::~CameraSource() { stop(); }
void CameraSource::stop() {
    running_ = false;
    if (th_.joinable()) th_.join();
}
void CameraSource::capture_loop() {
    while (running_) {
        cv::Mat frame;
        cap_ >> frame;
        if (!frame.empty()) {
            std::lock_guard<std::mutex> lock(mtx_);
            if (frame_q_.size() < 2)
                frame_q_.push(frame.clone());
            cv_.notify_one();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
bool CameraSource::get(cv::Mat& out, int timeout_ms) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (frame_q_.empty()) {
        if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] { return !frame_q_.empty(); }))
            return false;
    }
    if (!frame_q_.empty()) {
        out = frame_q_.front();
        frame_q_.pop();
        return true;
    }
    return false;
}
