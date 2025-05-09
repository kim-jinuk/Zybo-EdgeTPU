// vision_core/draw.cpp (요약)
void draw_box(cv::Mat& frame,
    const cv::Rect2f& box,
    const cv::Scalar& color,
    const std::string& label)
{
cv::rectangle(frame, box, color, 2);
if (!label.empty()) {
int  baseline = 0;
auto size = cv::getTextSize(label,
                          cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
cv::rectangle(frame,
            cv::Point(box.x, box.y - size.height - 4),
            cv::Point(box.x + size.width, box.y), color, cv::FILLED);
cv::putText(frame, label,
          cv::Point(box.x, box.y - 2),
          cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0,0,0), 1);
}
}
