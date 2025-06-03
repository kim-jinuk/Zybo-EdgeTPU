#include <iostream>
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>
#include "capture/camera_source.hpp"
#include "processing/preprocess.hpp"
#include "detection/edgetpu_detector.hpp"
#include "tracking/sort_tracker.hpp"
#include "utils/fps_counter.hpp"
#include "utils/logger.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        LOG_ERR("Usage: ./app model_edgetpu.tflite");
        return 1;
    }
    std::string model_path = argv[1];
    LOG("App started");

    CameraSource cam(0, 640, 480);
    EdgeTpuDetector detector(model_path, 0.4);
    SortTracker tracker(15, 2, 0.4);
    FpsCounter fps;
    Preprocessor::Options p_opt;
    p_opt.use_clahe = true; p_opt.use_sharpen = true; p_opt.use_unsharp = true; p_opt.use_denoise = false;

    while (true) {
        cv::Mat frame;
        if (!cam.get(frame)) continue;
        auto enhanced = Preprocessor::enhance(frame, p_opt);

        auto dets = detector.infer(enhanced);
        std::vector<cv::Rect> boxes;
        for (auto& d : dets) boxes.push_back(d.box);

        auto tracks = tracker.update(boxes);

        // Draw results
        for (const auto& t : tracks)
            cv::rectangle(enhanced, t.box, {0,255,0}, 2);
        float f = fps.update();
        cv::putText(enhanced, "FPS: " + std::to_string(int(f)), {10,30}, cv::FONT_HERSHEY_SIMPLEX, 1, {255,255,0}, 2);

        cv::imshow("result", enhanced);
        int key = cv::waitKey(1);
        if (key == 27) break;
    }
    cam.stop();
    LOG("App exit");
}
