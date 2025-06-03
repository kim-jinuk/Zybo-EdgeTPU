#include "capture/camera_source.hpp"
#include "utils/logger.hpp"

CameraSource::CameraSource(int cam_id, SafeQueue<cv::Mat>& out, int width, int height)
    : cap_(cam_id), q_(out) {
    if (!cap_.isOpened()) {
        log("Camera", "Failed to open camera: " + std::to_string(cam_id));
        std::exit(EXIT_FAILURE);
    }
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    log("Camera", "Camera initialized");
}

void CameraSource::operator()() {
    log("Camera", "Capture thread started");
    while (true) {
        cv::Mat frame;
        cap_ >> frame;
        if (frame.empty()) {
            log("Camera", "Empty frame captured, skipping");
            continue;
        }
        q_.push(frame);
    }
}
