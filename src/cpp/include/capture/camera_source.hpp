#pragma once
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>

class CameraSource {
public:
    CameraSource(int device = 0, int w = 640, int h = 480);
    ~CameraSource();
    bool get(cv::Mat& out, int timeout_ms = 50);
    void stop();

private:
    void capture_loop();
    cv::VideoCapture cap_;
    std::thread th_;
    std::atomic<bool> running_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::queue<cv::Mat> frame_q_;
};
