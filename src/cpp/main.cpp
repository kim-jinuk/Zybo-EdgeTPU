#include <iostream>
#include <exception>
#include "pipeline/pipeline.hpp"

int main(int argc, char* argv[]) {
    int cam_id = 0;
    try {
        zybo::pipeline::Pipeline pipe(cam_id, 640, 480, 30,
                                      "../models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite", 0.4f);
        pipe.start();
        std::cerr << "Pipeline started\n";
        std::this_thread::sleep_for(std::chrono::seconds(10));
        pipe.stop();
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << std::endl;
    }
}
