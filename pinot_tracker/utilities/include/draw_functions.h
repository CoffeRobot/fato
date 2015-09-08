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

void drawBoundingCube(const std::vector<cv::Point3f>& front_box,
                      const std::vector<cv::Point3f>& back_box,
                      const float focal, const cv::Point2f& imgCenter,
                      cv::Mat& out);

void drawBoundingCube(const std::vector<cv::Point3f>& front_box,
                      const std::vector<cv::Point3f>& back_box,
                      const float focal, const cv::Point2f& imgCenter,
                      const cv::Scalar& color, int line_width, cv::Mat& out);

void applyColorMap(const cv::Mat& in, cv::Mat& out);

void drawObjectLocation(const std::vector<cv::Point3f>& back_box,
                        const std::vector<cv::Point3f>& front_box,
                        const cv::Point3f& center,
                        const std::vector<bool>& visibleFaces,
                        const float focal, const cv::Point2f& imgCenter,
                        cv::Mat& out);

void drawCentroidVotes(const std::vector<cv::Point3f*>& points,
                       const std::vector<cv::Point3f*>& votes,
                       const cv::Point2f& center, bool drawLines,
                       const float focal, cv::Mat& out);

void drawObjectPose(const cv::Point3f& centroid, const float focal,
                    const cv::Point2f& img_center, const cv::Mat& rotation,
                    cv::Mat& out);

void arrowedLine(cv::Mat& img, cv::Point2f pt1, cv::Point2f pt2,
                 const cv::Scalar& color, int thickness = 1, int line_type = 8,
                 int shift = 0, double tipLength = 0.1);

}  // end namespace

#endif  // DRAW_FUNCTIONS_H
