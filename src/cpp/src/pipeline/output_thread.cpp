#include "pipeline/output_thread.hpp"
#include <opencv2/opencv.hpp>
void OutputThread::operator()(){
    while(true){
        FramePack p=in_.pop();
        for(auto &tr: p.tracks){
            cv::rectangle(p.frame,tr.box,{0,255,0},2);
            cv::putText(p.frame,std::to_string(tr.id),{tr.box.x,tr.box.y-4},
                        cv::FONT_HERSHEY_SIMPLEX,0.5,{0,255,0},1);
        }
        cv::putText(p.frame,cv::format("FPS %.1f",p.fps),{10,20},
                    cv::FONT_HERSHEY_SIMPLEX,0.6,{255,0,0},2);
        cv::imshow("TPU + SORT",p.frame);
        if(cv::waitKey(1)==27) break;
    }
}