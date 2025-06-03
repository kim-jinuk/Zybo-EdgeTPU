#pragma once
#include <opencv2/opencv.hpp>
#include <edgetpu.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>
#include <memory>
#include <vector>

struct Detection {
    cv::Rect box;
    float score;
    int cls;
};

class EdgeTpuDetector {
public:
    EdgeTpuDetector(const std::string& model_path, float thr = 0.4);
    std::vector<Detection> infer(const cv::Mat& bgr);
    std::pair<int, int> input_size() const { return {w_, h_}; }
private:
    std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_ctx_;
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
    int w_, h_;
    float thr_;
};
