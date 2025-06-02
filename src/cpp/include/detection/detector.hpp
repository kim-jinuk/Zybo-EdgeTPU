// // ================= detection/detector.hpp =====================================
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

namespace zybo::detection {
struct Detection { float x1, y1, x2, y2, score; };
class Detector {
public:
    virtual ~Detector() = default;
    virtual std::vector<Detection> detect(const cv::Mat& img) = 0;
};
} // namespace zybo::detection