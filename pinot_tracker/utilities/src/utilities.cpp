#include "../include/utilities.h"

namespace pinot_tracker{

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

} // end namespace

