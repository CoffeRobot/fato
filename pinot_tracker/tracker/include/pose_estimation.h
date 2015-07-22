#ifndef POSE_ESTIMATION_H
#define POSE_ESTIMATION_H

#include <opencv2/core/core.hpp>
#include <vector>

#include "../../utilities/include/constants.h"

namespace pinot_tracker {

void getPose2D(const std::vector<cv::Point3f>& model_points,
               const std::vector<cv::Point2f>& tracked_points,
               const cv::Mat& camera_model, int iterations, float distance,
               std::vector<int>& inliers, cv::Mat& rotation,
               cv::Mat& translation);

void getPose2D(const std::vector<cv::Point2f*>& model_points,
               const std::vector<cv::Point2f*>& tracked_points, float& scale,
               float& angle);

//cv::Mat getPose3D(const std::vector<cv::Point3f*>& model_points,
//                  const std::vector<cv::Point3f*>& tracked_points,
//                  const std::vector<Status*>& points_status);

cv::Mat getRigidTransform(cv::Mat& a, cv::Mat& b);

cv::Mat getRigidTransform(cv::Mat& a, cv::Mat& b, std::vector<float>& cA,
                          std::vector<float>& cB);

void rotateBBox(const std::vector<cv::Point3f>& bBox, const cv::Mat& rotation,
                std::vector<cv::Point3f>& updatedBBox);

void rotatePoint(const cv::Point3f& point, const cv::Mat& rotation,
                 cv::Point3f& updatedPoint);

void rotatePoint(const cv::Vec3f& point, const cv::Mat& rotation,
                 cv::Vec3f& updatedPoint);

void rotatePoint(const cv::Vec3f& point, const cv::Mat& rotation,
                 cv::Point3f& updatedPoint);

}  // end namespace

#endif  // POSE_ESTIMATION_H
