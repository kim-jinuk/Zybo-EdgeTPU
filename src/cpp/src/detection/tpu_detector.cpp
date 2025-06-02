// ================= detection/tpu_detector.cpp ==================================
#include "detection/tpu_detector.hpp"
#include <pybind11/embed.h>
#include <opencv2/imgproc.hpp>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace zybo::detection;

struct TPUDetector::Impl {
    py::scoped_interpreter guard{};   // start Python interpreter once
    py::object interpreter;
    py::tuple   input_size;

    Impl(const std::string& model) {
        py::module coral_utils = py::module::import("pycoral.utils.edgetpu");
        py::module coral_common = py::module::import("pycoral.adapters.common");
        interpreter = coral_utils.attr("make_interpreter")(model);
        interpreter.attr("allocate_tensors")();
        input_size = coral_common.attr("input_size")(interpreter);
    }
    
    std::vector<Detection> run(const cv::Mat& img, float thresh) {
        // Resize
        int w = input_size[0].cast<int>();
        int h = input_size[1].cast<int>();
        cv::Mat resized; cv::resize(img, resized, {w, h});

        // Copy to Python
        auto np = py::module::import("numpy");
        py::array_t<uint8_t> input = py::cast(resized);
        py::module coral_common = py::module::import("pycoral.adapters.common");
        coral_common.attr("set_input")(interpreter, input);
        interpreter.attr("invoke")();

        py::module detect_mod = py::module::import("pycoral.adapters.detect");
        py::list objs = detect_mod.attr("get_objects")(interpreter, thresh);

        std::vector<Detection> dets;
        for (auto& o : objs) {
            auto bbox = o.attr("bbox");
            Detection d;
            d.x1 = bbox.attr("xmin").cast<int>();
            d.y1 = bbox.attr("ymin").cast<int>();
            d.x2 = bbox.attr("xmax").cast<int>();
            d.y2 = bbox.attr("ymax").cast<int>();
            d.score = o.attr("score").cast<float>();
            // scale back (handled in caller for now)
            dets.push_back(d);
        }
        return dets;
    }
};

TPUDetector::TPUDetector(const std::string& model_path, float threshold)
    : threshold_{threshold}, impl_{std::make_unique<Impl>(model_path)} {
    input_size_ = {impl_->input_size[0].cast<int>(), impl_->input_size[1].cast<int>()};
}

std::vector<Detection> TPUDetector::detect(const cv::Mat& img) {
    auto dets = impl_->run(img, threshold_);
    // scale coords back
    float sx = static_cast<float>(img.cols) / input_size_.width;
    float sy = static_cast<float>(img.rows) / input_size_.height;
    for (auto& d : dets) {
        d.x1 *= sx; d.x2 *= sx; d.y1 *= sy; d.y2 *= sy;
    }
    return dets;
}
TPUDetector::~TPUDetector() = default;