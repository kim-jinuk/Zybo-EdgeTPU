// ================= tracking/sort_tracker.hpp ===================================
#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <memory>

namespace zybo::tracking {

struct Track {
    float x1, y1, x2, y2; int id; int age; int hit_streak;
};

class SortTracker {
public:
    SortTracker(int max_age = 30, int min_hits = 3, float iou_thr = 0.3f);
    std::vector<Track> update(const std::vector<cv::Rect2f>& detections);
private:
    struct KalmanBox {                     // ★ 헤더에 전체 정의
        cv::Rect2f bbox;
        int id;
        int age = 0, hit = 0, time_since_update = 0;
        KalmanBox(const cv::Rect2f& r, int i) : bbox(r), id(i) {}
        void predict() { age++; time_since_update++; }
        void update(const cv::Rect2f& r) { bbox = r; time_since_update = 0; hit++; }
    };                       // fwd
    float iou(const cv::Rect2f&, const cv::Rect2f&) const;
    int max_age_, min_hits_; float iou_thr_;
    int next_id_ = 1;
    std::vector<std::unique_ptr<KalmanBox>> trackers_;
};

} // namespace zybo::tracking