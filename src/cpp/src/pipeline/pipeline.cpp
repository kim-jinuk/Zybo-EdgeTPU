// ================= pipeline/pipeline.cpp =======================================
#include "pipeline/pipeline.hpp"
using namespace zybo::pipeline; using namespace zybo;
Pipeline::Pipeline(int cam,int w,int h,int fps,const std::string& model,float thr)
    :cam_(cam,w,h,fps,cap_q_),
     proc_({std::make_shared<processing::GammaContrast>(1.5f), std::make_shared<processing::UnsharpMask>(1.0f)}),
     det_(model,thr), tracker_{}
{}
Pipeline::~Pipeline(){stop();}
void Pipeline::start(){cam_.start(); out_.start(); running_=true; thr_=std::thread(&Pipeline::loop,this);} 
void Pipeline::stop(){running_=false; cam_.stop(); cap_q_.terminate(); out_.stop(); if(thr_.joinable()) thr_.join();}

void Pipeline::loop(){ while(running_){ auto opt=cap_q_.pop(); if(!opt) break; auto f=std::move(*opt); auto img=proc_(f.img); auto dets=det_.detect(img); std::vector<cv::Rect2f> b; b.reserve(dets.size()); for(auto& d:dets) b.emplace_back(cv::Rect2f(d.x1,d.y1,d.x2-d.x1,d.y2-d.y1)); auto trk=tracker_.update(b); zybo::output::DrawPack pack{ {f.timestamp,img}, trk }; draw_q_.push(std::move(pack)); } }
