#ifndef DRAW_FUNCTIONS_H
#define DRAW_FUNCTIONS_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "constants.h"

namespace pinot_tracker {

void drawBoundingBox(const std::vector<cv::Point2f>& box, cv::Scalar color,
                     int line_width, cv::Mat& out);

void drawBoundingCube(const cv::Point3f& center,
                      const std::vector<cv::Point3f>& front_box,
                      const std::vector<cv::Point3f>& back_box,
                      const float focal, const cv::Point2f& imgCenter,
                      cv::Mat& out);

void applyColorMap(const cv::Mat& in, cv::Mat& out);

void drawObjectLocation(const std::vector<cv::Point3f>& back_box,
                        const std::vector<cv::Point3f>& front_box,
                        const cv::Point3f& center,
                        const std::vector<bool>& visibleFaces,
                        const float focal, const cv::Point2f& imgCenter,
                        cv::Mat& out);

void drawCentroidVotes(const std::vector<cv::Point3f*>& points,
                       const std::vector<cv::Point3f*>& votes,
                       const cv::Point2f& center,
                       bool drawLines,
                       const float focal,
                       cv::Mat& out);

}  // end namespace

#endif  // DRAW_FUNCTIONS_H
