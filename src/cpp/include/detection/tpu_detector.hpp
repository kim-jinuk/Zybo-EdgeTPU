// ================= detection/tpu_detector.hpp ==================================
#pragma once
#include "detection/detector.hpp"
#include <string>
#include <memory>

namespace zybo::detection {
class TPUDetector : public Detector {
public:
    TPUDetector(const std::string& model_path, float threshold = 0.4f);
    ~TPUDetector() override;
    std::vector<Detection> detect(const cv::Mat& img) override;
private:
    float threshold_;
    cv::Size input_size_;
    struct Impl;                    // Pimpl → hides PyCoral / TFLite details
    std::unique_ptr<Impl> impl_;
};
} // namespace zybo::detection
