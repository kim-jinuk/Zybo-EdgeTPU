#pragma once
#include <opencv2/opencv.hpp>
#include <edgetpu.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <coral/vision/detection.h>
#include <memory>
struct Detection { cv::Rect box; float score; int cls; };
class EdgeTpuDetector {
public:
    EdgeTpuDetector(const std::string& model, float thr=0.4);
    std::vector<Detection> infer(const cv::Mat& bgr);
    std::pair<int,int> input_size() const { return {w_,h_}; }
private:
    std::unique_ptr<tflite::Interpreter> itp_;
    int w_, h_; float thr_;
};