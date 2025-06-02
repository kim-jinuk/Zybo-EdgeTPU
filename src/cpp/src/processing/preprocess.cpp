#include "processing/preprocess.hpp"
namespace preprocess {
cv::Mat run(const cv::Mat& src) {
    static cv::Mat k = (cv::Mat_<float>(3,3)<<0,-1,0,-1,5,-1,0,-1,0);
    cv::Mat x; cv::filter2D(src, x, src.depth(), k);
    cv::GaussianBlur(x, x, {7,7}, 0);
    cv::Mat yuv; cv::cvtColor(x, yuv, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> ch; cv::split(yuv, ch);
    cv::Ptr<cv::CLAHE> c = cv::createCLAHE(2.0, {8,8});
    c->apply(ch[0], ch[0]); cv::merge(ch, yuv);
    cv::cvtColor(yuv, x, cv::COLOR_YCrCb2BGR);
    return x;
}}