#ifndef DRAW_FUNCTIONS_H
#define DRAW_FUNCTIONS_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


namespace pinot_tracker
{

void drawBoundingBox(const std::vector<cv::Point2f>& box, cv::Scalar color,
                     int line_width, cv::Mat& out);


}// end namespace

#endif // DRAW_FUNCTIONS_H

