#include "pipeline/pipeline.hpp"
#include <thread>
#include <chrono>
int main() {
    zybo::pipeline::Pipeline pipe(0, 640, 480, 30,
                                  "models/model.tflite", 0.4f);
    pipe.start();
    //  ⌘ 30초 돌리고 종료
    std::this_thread::sleep_for(std::chrono::seconds(30));
    pipe.stop();
}
