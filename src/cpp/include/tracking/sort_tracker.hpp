// include/tracking/sort_tracker.hpp
#pragma once
#include "detection/edgetpu_detector.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
struct Track { int id; cv::Rect box; int age; int hits; int no_hit; };
class SortTracker {
public:
    explicit SortTracker(float iou_thr=0.3f,int max_age=5,int min_hits=3);
    std::vector<Track> update(const std::vector<Detection>& dets);
private:
    struct KF{ cv::KalmanFilter kf; cv::Mat state;};
    struct TrackInt{ int id; Track trk; KF kf; };
    float iou_thr_; int max_age_; int min_hits_; int next_id_=0;
    std::vector<TrackInt> tracks_;
    // helpers
    static float iou(const cv::Rect& a,const cv::Rect& b);
    static std::vector<std::pair<int,int>> hungarian(const std::vector<std::vector<float>>& cost);
    cv::Rect predict_box(KF&);
    KF create_kalman(const cv::Rect&);
};