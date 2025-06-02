// ================= capture/video_capture.hpp ===================================
#pragma once
#include "capture/capture_base.hpp"
#include <opencv2/opencv.hpp>

namespace zybo::capture {
class VideoCapture : public CaptureBase {
public:
    VideoCapture(const std::string& path, Queue& out_q);
    ~VideoCapture() override = default;
protected:
    bool grab(Frame& out) override;
private:
    cv::VideoCapture cap_;
    double frame_period_ms_;
};
} // namespace zybo::capture