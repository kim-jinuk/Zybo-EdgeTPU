// ================= processing/enhancer.cpp =====================================
#include "processing/enhancer.hpp"
#include <cmath>
using namespace zybo::processing;
GammaContrast::GammaContrast(float g){ lut_=cv::Mat(1,256,CV_8U); for(int i=0;i<256;++i) lut_.at<uchar>(i)=cv::saturate_cast<uchar>(pow(i/255.f,g)*255.f);} cv::Mat GammaContrast::operator()(const cv::Mat& in) const{cv::Mat o;cv::LUT(in,lut_,o);return o;}
UnsharpMask::UnsharpMask(float s,int k):s_(s),k_(k){} cv::Mat UnsharpMask::operator()(const cv::Mat& in) const{cv::Mat b;cv::GaussianBlur(in,b,{k_,k_},0);cv::Mat o;cv::addWeighted(in,1+s_,b,-s_,0,o);return o;}
CLAHE::CLAHE(double c,cv::Size g){clahe_=cv::createCLAHE(c,g);} cv::Mat CLAHE::operator()(const cv::Mat& in) const{cv::Mat lab,l,a,b;cv::cvtColor(in,lab,cv::COLOR_BGR2Lab);cv::split(lab,std::vector<cv::Mat>{l,a,b});clahe_->apply(l,l);cv::merge(std::vector<cv::Mat>{l,a,b},lab);cv::Mat o;cv::cvtColor(lab,o,cv::COLOR_Lab2BGR);return o;}
Denoise::Denoise(float h):h_(h){} cv::Mat Denoise::operator()(const cv::Mat& in) const{cv::Mat o;cv::fastNlMeansDenoisingColored(in,o,h_,h_,7,21);return o;}
Compose::Compose(std::vector<std::shared_ptr<Preprocessor>> v):v_(std::move(v)){} cv::Mat Compose::operator()(const cv::Mat& in) const{cv::Mat img=in.clone();for(auto& s:v_) img=(*s)(img);return img;}
