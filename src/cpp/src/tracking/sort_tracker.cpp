// src/tracking/sort_tracker.cpp
#include "tracking/sort_tracker.hpp"
#include <algorithm>
#include <cmath>

float SortTracker::iou(const cv::Rect& a,const cv::Rect& b){
    int x1=std::max(a.x,b.x),y1=std::max(a.y,b.y);
    int x2=std::min(a.x+a.width,b.x+b.width);
    int y2=std::min(a.y+a.height,b.y+b.height);
    int inter=std::max(0,x2-x1)*std::max(0,y2-y1);
    int uni=a.area()+b.area()-inter; return uni? (float)inter/uni:0.f; }

std::vector<std::pair<int,int>> SortTracker::hungarian(const std::vector<std::vector<float>>& cost){
    int n=cost.size(),m=cost[0].size();
    std::vector<std::pair<int,int>> out; std::vector<char> used(m,0);
    for(int i=0;i<n;++i){ float best=1e9; int bj=-1; for(int j=0;j<m;++j) if(!used[j]&&cost[i][j]<best){ best=cost[i][j]; bj=j; }
        if(bj>=0){ used[bj]=1; out.push_back({i,bj}); }} return out; }

SortTracker::KF SortTracker::create_kalman(const cv::Rect& b){
    KF k; k.kf.init(7,4,0); float cx=b.x+b.width/2.f,cy=b.y+b.height/2.f; float s=b.area(),r=b.width/(float)b.height;
    k.state=(cv::Mat_<float>(7,1)<<cx,cy,s,r,0,0,0);
    k.kf.statePre=k.state.clone(); setIdentity(k.kf.measurementMatrix);
    setIdentity(k.kf.transitionMatrix); k.kf.transitionMatrix.at<float>(0,4)=1; k.kf.transitionMatrix.at<float>(1,5)=1; k.kf.transitionMatrix.at<float>(2,6)=1;
    setIdentity(k.kf.processNoiseCov,cv::Scalar::all(1e-2)); setIdentity(k.kf.measurementNoiseCov,cv::Scalar::all(1e-1)); return k; }

cv::Rect SortTracker::predict_box(KF& kf){
    cv::Mat p=kf.kf.predict(); float cx=p.at<float>(0),cy=p.at<float>(1),s=p.at<float>(2),r=p.at<float>(3);
    float w=sqrtf(s*r),h=s/w; return {int(cx-w/2),int(cy-h/2),int(w),int(h)}; }

SortTracker::SortTracker(float iou_thr,int max_age,int min_hits):iou_thr_(iou_thr),max_age_(max_age),min_hits_(min_hits){}

std::vector<Track> SortTracker::update(const std::vector<Detection>& dets){
    // 1 predict
    std::vector<cv::Rect> preds; preds.reserve(tracks_.size()); for(auto &t:tracks_) preds.push_back(predict_box(t.kf));
    // 2 cost
    std::vector<std::vector<float>> cost(tracks_.size(),std::vector<float>(dets.size(),1.f));
    for(size_t i=0;i<tracks_.size();++i) for(size_t j=0;j<dets.size();++j) cost[i][j]=1.f-iou(preds[i],dets[j].box);
    // 3 assign
    auto matches=hungarian(cost); std::vector<int> det_asg(dets.size(),-1),trk_asg(tracks_.size(),-1);
    for(auto&m:matches){ if(1.f-cost[m.first][m.second]<iou_thr_) continue; trk_asg[m.first]=m.second; det_asg[m.second]=m.first; }
    // 4 update
    for(size_t i=0;i<tracks_.size();++i){ auto &t=tracks_[i]; if(trk_asg[i]!=-1){ int j=trk_asg[i]; cv::Mat z=(cv::Mat_<float>(4,1)<<dets[j].box.x+dets[j].box.width/2.f,dets[j].box.y+dets[j].box.height/2.f,(float)dets[j].box.area(),dets[j].box.width/(float)dets[j].box.height);
            t.kf.kf.correct(z); t.trk.box=predict_box(t.kf); t.trk.hits++; t.trk.no_hit=0; } else { t.trk.box=predict_box(t.kf); t.trk.no_hit++; } t.trk.age++; }
    // 5 new
    for(size_t j=0;j<dets.size();++j) if(det_asg[j]==-1){ TrackInt nt; nt.id=next_id_++; nt.kf=create_kalman(dets[j].box); nt.trk={nt.id,dets[j].box,1,1,0}; tracks_.push_back(nt);}    
    // 6 prune
    tracks_.erase(std::remove_if(tracks_.begin(),tracks_.end(),[&](const TrackInt&t){return t.trk.no_hit>max_age_;}),tracks_.end());
    // 7 output
    std::vector<Track> out; for(auto &ti:tracks_) if(ti.trk.hits>=min_hits_||ti.trk.age<=min_hits_) out.push_back(ti.trk); return out; }