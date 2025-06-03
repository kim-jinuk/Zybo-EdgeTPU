#include "processing/preprocess.hpp"

cv::Mat Preprocessor::enhance(const cv::Mat& src, const Options& opt) {
    cv::Mat dst = src.clone();
    if (opt.use_clahe) {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        cv::Mat lab; cv::cvtColor(dst, lab, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> lab_planes(3);
        cv::split(lab, lab_planes);
        clahe->apply(lab_planes[0], lab_planes[0]);
        cv::merge(lab_planes, lab);
        cv::cvtColor(lab, dst, cv::COLOR_Lab2BGR);
    }
    if (opt.use_sharpen) {
        cv::Mat sharp;
        cv::GaussianBlur(dst, sharp, cv::Size(0, 0), 3);
        cv::addWeighted(dst, 1.5, sharp, -0.5, 0, dst);
    }
    if (opt.use_denoise) {
        cv::fastNlMeansDenoisingColored(dst, dst, 10, 10, 7, 21);
    }
    if (opt.use_unsharp) {
        cv::Mat blurred;
        cv::GaussianBlur(dst, blurred, cv::Size(0, 0), 1.0);
        cv::addWeighted(dst, 1.3, blurred, -0.3, 0, dst);
    }
    return dst;
}
