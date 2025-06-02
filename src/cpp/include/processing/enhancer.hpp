// ================= processing/enhancer.hpp =====================================
#pragma once
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
namespace zybo::processing {
class Preprocessor { public: virtual ~Preprocessor() = default; virtual cv::Mat operator()(const cv::Mat&) const = 0; };
class GammaContrast : public Preprocessor { public: explicit GammaContrast(float g=1.5f); cv::Mat operator()(const cv::Mat&) const override; private: cv::Mat lut_; };
class UnsharpMask  : public Preprocessor { public: explicit UnsharpMask(float s=1.0f,int k=5); cv::Mat operator()(const cv::Mat&) const override; private: float s_; int k_; };
class CLAHE       : public Preprocessor { public: explicit CLAHE(double clip=4.0, cv::Size grid={8,8}); cv::Mat operator()(const cv::Mat&) const override; private: cv::Ptr<cv::CLAHE> clahe_; };
class Denoise     : public Preprocessor { public: explicit Denoise(float h=10.0f); cv::Mat operator()(const cv::Mat&) const override; private: float h_; };
class Compose     : public Preprocessor { public: explicit Compose(std::vector<std::shared_ptr<Preprocessor>> v); cv::Mat operator()(const cv::Mat&) const override; private: std::vector<std::shared_ptr<Preprocessor>> v_; };
} // namespace