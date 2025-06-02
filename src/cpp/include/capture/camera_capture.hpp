// ================= capture/camera_capture.hpp ==================================
#pragma once
#include "capture/capture_base.hpp"
#include <opencv2/opencv.hpp>

namespace zybo::capture {
class CameraCapture : public CaptureBase {
public:
    CameraCapture(int cam_id, int width, int height, int fps, Queue& out_q);
    ~CameraCapture() override = default;
protected:
    bool grabFrame(Frame& out) override;
private:
    cv::VideoCapture cap_;
    double frame_period_ms_;
};
} // namespace zybo::capture