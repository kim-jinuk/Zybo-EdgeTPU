// ================= pipeline/pipeline.cpp =========================
#include <opencv2/core.hpp>
#include "pipeline/pipeline.hpp"

namespace zybo::pipeline {          // ← 블록 시작 -------------------

Pipeline::Pipeline(int cam, int w, int h, int fps,
                   const std::string& model, float thr)
    : cam_(cam, w, h, fps, cap_q_),
      proc_({ std::make_shared<processing::GammaContrast>(1.5f),
              std::make_shared<processing::UnsharpMask>(1.0f)}),
      det_(model, thr),
      tracker_ {},
      out_{draw_q_}
{}

Pipeline::~Pipeline() { stop(); }

void Pipeline::start() {
    cam_.start();
    out_.start();
    running_ = true;
    thr_ = std::thread(&Pipeline::loop, this);
}
void Pipeline::stop() {
    running_ = false;
    cam_.stop();
    cap_q_.terminate();
    out_.stop();
    if (thr_.joinable()) thr_.join();
}

void Pipeline::loop()
{
    while (running_) {
        auto opt = cap_q_.pop();
        if (!opt) break;
        auto in = std::move(*opt);

        // --- preprocessing
        auto pre = proc_(in.img);

        // --- detection
        std::vector<detection::Detection> dets;
        try { dets = det_.detect(pre); }
        catch (const std::exception& e) {
            std::cerr << "[Detect] " << e.what() << '\n';
            continue;
        }

        // --- tracking
        std::vector<cv::Rect2f> det_rects;
        for (auto& d : dets)
            det_rects.emplace_back(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);

        std::vector<tracking::Track> tracks;
        try { tracks = tracker_.update(det_rects); }
        catch (const std::exception& e) {
            std::cerr << "[Track] " << e.what() << '\n';
            continue;
        }

        // --- push to display
        zybo::output::DrawPack pack{ std::move(in), std::move(tracks) };
        draw_q_.push(std::move(pack));
    }
}

} // namespace zybo::pipeline       // ← 블록 끝 --------------------
