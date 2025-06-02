// ================= capture/frame.hpp ===========================================
#pragma once
#include <opencv2/opencv.hpp>
namespace zybo::capture {
struct Frame { double timestamp; cv::Mat img; };
} // namespace