// ================= output/output_display.cpp ===================================
#include "output/output_display.hpp"
#include <opencv2/opencv.hpp>
using namespace zybo::output;
void OutputDisplay::loop(){ while(running_){ auto opt=q_.pop(); if(!opt) break; auto pack=std::move(*opt); auto img=pack.frame.img; for(auto& t:pack.tracks){ cv::rectangle(img,{(int)t.x1,(int)t.y1},{(int)t.x2,(int)t.y2},{0,255,0},2); cv::putText(img,std::to_string(t.id),{(int)t.x1,(int)t.y1-5},cv::FONT_HERSHEY_SIMPLEX,0.5,{255,0,0},1);} cv::imshow("ZyboEdge",img); if(cv::waitKey(1)==27){running_=false;break;} } cv::destroyAllWindows(); }
