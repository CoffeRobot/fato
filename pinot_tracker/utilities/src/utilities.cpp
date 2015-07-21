#include "../include/utilities.h"
#include <iostream>

namespace pinot_tracker {

cv::Mat1b getMask(int rows, int cols, const cv::Point2d& begin,
                  const cv::Point2d& end) {
  cv::Mat1b mask(rows, cols, static_cast<uchar>(0));
  rectangle(mask, begin, end, static_cast<uchar>(255), -1);
  return mask;
}

void opencvToEigen(const cv::Mat& rot, Eigen::Matrix3d& rotation) {
  rotation = Eigen::Matrix3d(3, 3);

  rotation(0, 0) = static_cast<double>(rot.at<float>(0, 0));
  rotation(0, 1) = static_cast<double>(rot.at<float>(0, 1));
  rotation(0, 2) = static_cast<double>(rot.at<float>(0, 2));
  rotation(1, 0) = static_cast<double>(rot.at<float>(1, 0));
  rotation(1, 1) = static_cast<double>(rot.at<float>(1, 1));
  rotation(1, 2) = static_cast<double>(rot.at<float>(1, 2));
  rotation(2, 0) = static_cast<double>(rot.at<float>(2, 0));
  rotation(2, 1) = static_cast<double>(rot.at<float>(2, 1));
  rotation(2, 2) = static_cast<double>(rot.at<float>(2, 2));
}

void eigenToOpencv(const Eigen::Matrix3d& src, cv::Mat& dst) {
  dst = cv::Mat(3, 3, CV_32FC1, 0.0f);

  dst.at<float>(0, 0) = static_cast<float>(src(0, 0));
  dst.at<float>(0, 1) = static_cast<float>(src(0, 1));
  dst.at<float>(0, 2) = static_cast<float>(src(0, 2));

  dst.at<float>(1, 0) = static_cast<float>(src(1, 0));
  dst.at<float>(1, 1) = static_cast<float>(src(1, 1));
  dst.at<float>(1, 2) = static_cast<float>(src(1, 2));

  dst.at<float>(2, 0) = static_cast<float>(src(2, 0));
  dst.at<float>(2, 1) = static_cast<float>(src(2, 1));
  dst.at<float>(2, 2) = static_cast<float>(src(2, 1));
}

cv::Point2f projectPoint(const float focal, const cv::Point2f& center,
                         const cv::Point3f& src) {
  cv::Point2f dst;

  dst.x = (focal * src.x / src.z) + center.x;
  dst.y = (center.y - (focal * src.y / src.z));

  return dst;
}

cv::Point2f projectPoint(const float focal, const cv::Point2f& center,
                         const cv::Point3f* src) {
  cv::Point2f dst;

  dst.x = (focal * src->x / src->z) + center.x;
  dst.y = (center.y - (focal * src->y / src->z));

  return dst;
}

bool projectPoint(const float focal, const cv::Point2f& center,
                  const cv::Point3f& src, cv::Point2f& dst) {
  if (src.z == 0) return false;

  dst.x = (focal * src.x / src.z) + center.x;
  dst.y = (center.y - (focal * src.y / src.z));

  if (dst.x < 0 || dst.x > center.x * 2) return false;
  if (dst.y < 0 || dst.y > center.y * 2) return false;

  if (isnan(dst.x) || isnan(dst.y)) return false;

  return true;
}

bool projectPoint(const float focal, const cv::Point2f& center,
                  const cv::Point3f* src, cv::Point2f& dst)
{
    if (src->z == 0) return false;

    dst.x = (focal * src->x / src->z) + center.x;
    dst.y = (center.y - (focal * src->y / src->z));

    if (dst.x < 0 || dst.x > center.x * 2) return false;
    if (dst.y < 0 || dst.y > center.y * 2) return false;

    if (isnan(dst.x) || isnan(dst.y)) return false;

    return true;
}

void depthTo3d(const cv::Mat& disparity, float cx, float cy,
                      float fx, float fy, cv::Mat3f& depth) {
  int cols = disparity.cols;
  int rows = disparity.rows;

  assert(depth.cols != 0 && depth.rows != 0);
  assert(fx != 0 && fy != 0);

  for (size_t y = 0; y < rows; y++) {
    for (size_t x = 0; x < cols; x++) {


      uint16_t val = disparity.at<uint16_t>(y, x);
      float d = static_cast<float>(val);

      if(!is_valid(val))
          continue;
      if(val == 0)
          continue;

      if(d == 0)
          continue;

      float xp = x - cx;
      float yp = -(y - cy);

      float Z = DepthTraits<uint16_t>::toMeters(d);;
      float X = xp * Z / fx;
      float Y = yp * Z / fy;

      depth.at<cv::Vec3f>(y, x) = cv::Vec3f(X, Y, Z);
    }
  }
}



}  // end namespace
