#include "detection/edgetpu_detector.hpp"
#include <coral/tflite_utils.h>
EdgeTpuDetector::EdgeTpuDetector(const std::string& model, float thr):thr_(thr){
    auto mdl = tflite::FlatBufferModel::BuildFromFile(model.c_str());
    auto ctx = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
    itp_ = coral::MakeEdgeTpuInterpreterOrDie(*mdl, ctx.get());
    itp_->AllocateTensors();
    auto* in = itp_->input_tensor(0); h_=in->dims->data[1]; w_=in->dims->data[2];
}
std::vector<Detection> EdgeTpuDetector::infer(const cv::Mat& bgr){
    cv::Mat rgb; cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb,rgb,{w_,h_});
    memcpy(itp_->typed_input_tensor<uint8_t>(0), rgb.data, rgb.total()*3);
    itp_->Invoke();
    auto res = coral::GetDetectionResults(*itp_, thr_);
    std::vector<Detection> out;
    for(auto &r:res){ out.push_back({
        cv::Rect(r.bbox.xmin*bgr.cols, r.bbox.ymin*bgr.rows,
                 (r.bbox.xmax-r.bbox.xmin)*bgr.cols,
                 (r.bbox.ymax-r.bbox.ymin)*bgr.rows), r.score, r.id});}
    return out;
}