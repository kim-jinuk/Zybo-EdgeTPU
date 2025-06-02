#pragma once
#include <opencv2/opencv.hpp>
namespace preprocess {
cv::Mat run(const cv::Mat& src);   // 내부: sharpen → blur → CLAHE (CV_8UC3)
}