#ifndef POSE_ESTIMATION_H
#define POSE_ESTIMATION_H

#include <opencv2/core/core.hpp>
#include <vector>

namespace pinot_tracker {

void getPose2D(const std::vector<cv::Point3f>& model_points,
               const std::vector<cv::Point2f>& tracked_points,
               const cv::Mat& camera_model, int iterations, float distance,
               std::vector<int>& inliers, cv::Mat& rotation, cv::Mat& translation);

void getPose2D(const std::vector<cv::Point2f*>& model_points,
               const std::vector<cv::Point2f*>& tracked_points,
               float& scale,
               float& angle);

}  // end namespace

#endif  // POSE_ESTIMATION_H
