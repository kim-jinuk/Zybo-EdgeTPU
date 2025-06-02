// ================= tracking/sort_tracker.cpp ===================================
#include "tracking/sort_tracker.hpp"
#include <algorithm>
#include <numeric>

using namespace zybo::tracking;

// ---- simplistic 4‑state linear Kalman stub ----
struct SortTracker::KalmanBox {
    cv::Rect2f bbox; int id; int age=0; int hit=0; int time_since_update=0;
    KalmanBox(const cv::Rect2f& r,int i):bbox(r),id(i){}
    void predict(){age++; time_since_update++; /* no motion model */}
    void update(const cv::Rect2f& r){bbox=r; time_since_update=0; hit++;}
};

SortTracker::SortTracker(int max_age,int min_hits,float thr)
    :max_age_(max_age),min_hits_(min_hits),iou_thr_(thr){}

float SortTracker::iou(const cv::Rect2f& a,const cv::Rect2f& b) const {
    float inter=(a&b).area(); if(inter<=0) return 0.f; return inter/(a.area()+b.area()-inter);
}

std::vector<Track> SortTracker::update(const std::vector<cv::Rect2f>& dets){
    // 1) predict existing
    for(auto& t:trackers_) t->predict();

    // 2) IoU matrix
    size_t D=dets.size(),T=trackers_.size();
    std::vector<std::vector<float>> iouM(D, std::vector<float>(T,0));
    for(size_t d=0;d<D;++d) for(size_t t=0;t<T;++t) iouM[d][t]=iou(dets[d],trackers_[t]->bbox);

    std::vector<int> det_assigned(D,-1),trk_assigned(T,-1);
    // greedy matching
    while(true){float best= iou_thr_; int bi=-1,bj=-1;
        for(size_t d=0;d<D;++d) if(det_assigned[d]==-1)
            for(size_t t=0;t<T;++t) if(trk_assigned[t]==-1 && iouM[d][t]>best){best=iouM[d][t];bi=d;bj=t;}
        if(bi==-1) break;
        det_assigned[bi]=bj; trk_assigned[bj]=bi;
    }

    // 3) update matched trackers
    for(size_t t=0;t<T;++t){ if(trk_assigned[t]!=-1)
        trackers_[t]->update(dets[trk_assigned[t]]);
    }

    // 4) create new trackers for unmatched dets
    for(size_t d=0;d<D;++d) if(det_assigned[d]==-1){
        trackers_.push_back(std::make_unique<KalmanBox>(dets[d], next_id_++));
    }

    // 5) collect results & prune old trackers
    std::vector<Track> outputs;
    trackers_.erase(std::remove_if(trackers_.begin(),trackers_.end(),[&](auto& t){
        bool expired = t->time_since_update>max_age_;
        bool valid   = t->hit>=min_hits_ || next_id_<=min_hits_;
        if(!expired && valid){ outputs.push_back({t->bbox.x, t->bbox.y, t->bbox.x+t->bbox.width,
                                                 t->bbox.y+t->bbox.height, t->id, t->age, t->hit}); }
        return expired; }), trackers_.end());
    return outputs;
}