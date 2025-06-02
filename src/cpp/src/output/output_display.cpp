#include "output/output_display.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace zybo::output;

void OutputDisplay::loop()
{
    cv::namedWindow("ZyboEdge", cv::WINDOW_AUTOSIZE);
    std::cout << "[Display] thread started" << std::endl;

    while (running_) {
        auto opt = q_.pop();
        if (!opt) break; // queue terminated
        std::cout << "[Display] got frame\n";
        auto pack = std::move(*opt);
        auto img = pack.frame.img;

        // draw tracks
        for (auto &t : pack.tracks) {
            cv::rectangle(img, {(int)t.x1, (int)t.y1}, {(int)t.x2, (int)t.y2}, {0, 255, 0}, 2);
            cv::putText(img, std::to_string(t.id), {(int)t.x1, (int)t.y1 - 5},
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 0, 0}, 1);
        }

        cv::imshow("ZyboEdge", img);
        int key = cv::waitKey(1);
        if (key == 27) { // ESC
            running_ = false;
            break;
        }
    }

    cv::destroyAllWindows();
    std::cout << "[Display] thread finished" << std::endl;
}