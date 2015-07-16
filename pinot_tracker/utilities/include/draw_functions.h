#ifndef DRAW_FUNCTIONS_H
#define DRAW_FUNCTIONS_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


namespace pinot_tracker
{

void drawBoundingBox(const std::vector<cv::Point2f>& box, cv::Scalar color,
                     int line_width, cv::Mat& out);

void drawBoundingCube(const cv::Point3f& scdC,
                      const std::vector<cv::Point3f>& scdFrontBox,
                      const std::vector<cv::Point3f>& scdBackBox,
                      const float focal, const cv::Point2f& imgCenter,
                      cv::Mat& out);

void applyColorMap(const cv::Mat& in, cv::Mat& out);


}// end namespace

#endif // DRAW_FUNCTIONS_H

