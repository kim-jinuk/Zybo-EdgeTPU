#pragma once
#include <opencv2/opencv.hpp>

class Preprocessor {
public:
    struct Options {
        bool use_clahe = true;
        bool use_sharpen = true;
        bool use_denoise = false;
        bool use_unsharp = true;
    };
    static cv::Mat enhance(const cv::Mat& src, const Options& opt = Options());
};
