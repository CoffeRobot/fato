#ifndef UTILITIES_H
#define UTILITIES_H

#include<cmath>
#include <opencv2/core/core.hpp>
#include <image_geometry/pinhole_camera_model.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <limits>
#ifdef __WIN64
#include <Eigen/Dense>
#endif
#ifdef __unix__
#include <eigen3/Eigen/Dense>
#endif
//#include <pcl/PCLPointCloud2.h>


#include "../include/traits.h"

namespace pinot_tracker{

inline float roundUpToNearest(float src_number, float round_to)
{
  if(round_to == 0)
    return src_number;
  else if(src_number > 0)
    return std::ceil(src_number / round_to) * round_to;
  else
    return std::floor(src_number / round_to) * round_to;
}

inline float roundDownToNearest(float src_number, float round_to)
{
  if(round_to == 0)
    return src_number;
  else if(src_number > 0)
    return std::floor(src_number / round_to) * round_to;
  else
    return std::ceil(src_number / round_to) * round_to;
}

inline float getDistance(const cv::Point2f& a, const cv::Point2f& b)
{
  return sqrt( (a.x*a.x - b.x*b.x) + (a.y*a.y - b.y*b.y) );
//  return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

inline float getDistance(const cv::Point2f* a, const cv::Point2f* b)
{
    return sqrt(pow(a->x - b->x, 2) + pow(a->y - b->y, 2));
}

inline cv::Point2f mult(const cv::Mat2f& rot, const cv::Point2f& p) {
  return cv::Point2f(rot.at<float>(0, 0) * p.x + rot.at<float>(1, 0) * p.y,
                 rot.at<float>(0, 1) * p.x + rot.at<float>(1, 1) * p.y);
}

cv::Mat1b getMask(int rows, int cols, const cv::Point2d& begin,
                         const cv::Point2d& end);

void opencvToEigen(const cv::Mat& rot, Eigen::Matrix3d& rotation);

void eigenToOpencv(const Eigen::Matrix3d& src, cv::Mat& dst);

cv::Point2f projectPoint(const float focal, const cv::Point2f& center,
                           const cv::Point3f& src);

bool projectPoint(const float focal, const cv::Point2f& center,
                    const cv::Point3f& src, cv::Point2f& dst);

bool projectPoint(const float focal, const cv::Point2f& center,
                  const cv::Point3f* src, cv::Point2f& dst);

void depthTo3d(const cv::Mat& disparity, float cx, float cy,
                      float fx, float fy, cv::Mat3f &depth);

template<typename T>
bool is_infinite( const T &value )
{
    T max_value = std::numeric_limits<T>::max();
    T min_value = - max_value;

    return ! ( min_value <= value && value <= max_value );
}

template<typename T>
bool is_nan( const T &value )
{
    // True if NAN
    return value != value;
}

template<typename T>
bool is_valid( const T &value )
{
    return ! is_infinite(value) && ! is_nan(value);
}



} // end namespace



#endif // UTILITIES_H
