// ================= detection/factory.hpp ======================================
#pragma once
#include "detection/tpu_detector.hpp"
#include <memory>

namespace zybo::detection {
inline std::unique_ptr<Detector> createTPU(const std::string& model, float thr) {
    return std::make_unique<TPUDetector>(model, thr);
}
} // namespace