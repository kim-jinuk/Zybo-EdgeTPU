#include "detection/edgetpu_detector.hpp"
#include "utils/logger.hpp"

EdgeTpuDetector::EdgeTpuDetector(const std::string& model_path, float thr)
    : thr_(thr)
{
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    tflite::ops::builtin::BuiltinOpResolver resolver;
    edgetpu_ctx_ = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
    tflite::InterpreterBuilder(*model_, resolver)(&interpreter_);
    if (!interpreter_)
        throw std::runtime_error("Failed to build TFLite interpreter");
    interpreter_->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_ctx_.get());
    interpreter_->AllocateTensors();

    const TfLiteTensor* input_tensor = interpreter_->input_tensor(0);
    h_ = input_tensor->dims->data[1];
    w_ = input_tensor->dims->data[2];
}

std::vector<Detection> EdgeTpuDetector::infer(const cv::Mat& bgr)
{
    if (bgr.empty()) return {};
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, rgb, {w_, h_});
    std::memcpy(interpreter_->typed_input_tensor<uint8_t>(0), rgb.data, rgb.total() * 3);
    interpreter_->Invoke();

    // SSD Detection output
    const float* boxes = interpreter_->typed_output_tensor<float>(0);
    const float* classes = interpreter_->typed_output_tensor<float>(1);
    const float* scores = interpreter_->typed_output_tensor<float>(2);
    const float* num = interpreter_->typed_output_tensor<float>(3);

    int num_detections = static_cast<int>(num[0]);
    std::vector<Detection> out;
    for (int i = 0; i < num_detections; ++i) {
        if (scores[i] < thr_) continue;
        float ymin = boxes[i * 4 + 0], xmin = boxes[i * 4 + 1];
        float ymax = boxes[i * 4 + 2], xmax = boxes[i * 4 + 3];
        int x = std::max(0, int(xmin * bgr.cols));
        int y = std::max(0, int(ymin * bgr.rows));
        int w = std::min(bgr.cols - x, int((xmax - xmin) * bgr.cols));
        int h = std::min(bgr.rows - y, int((ymax - ymin) * bgr.rows));
        int cls = static_cast<int>(classes[i]);
        out.push_back({cv::Rect(x, y, w, h), scores[i], cls});
    }
    return out;
}
