#ifndef TOSTRING_H
#define TOSTRING_H

#include <string>
#include <opencv2/core/core.hpp>
#include <sstream>
#ifdef __unix__
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#else
#include <Eigen/Dense>
#include <Eigen/Geometry>
#endif


#include "Constants.h"

namespace pinot_tracker{

std::string toString(const cv::Point2f& p);

std::string toString(const Status& s);

std::string toString(const cv::Point3f& point);

std::string toString(const cv::Vec3f& point);

std::string toString(const Eigen::Matrix3d& rotation);

std::string toString(const Eigen::Quaterniond& quaternion);

std::string toPythonString(const cv::Mat& rotation);

std::string toPythonArray(const cv::Mat& rotation);

std::string toPythonString(const std::vector<cv::Point3f>& cloud);

template <typename T>
std::string toString(const cv::Mat& rotation, int precision);

} // end namespace

#endif
