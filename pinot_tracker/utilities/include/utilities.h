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
  return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
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

template <typename T>
void disparityToDepth(const sensor_msgs::ImageConstPtr& depth_msg,
                      const image_geometry::PinholeCameraModel& model,
                      cv::Mat3f& cloud)
{
    double range_max = 0.0;
    // Use correct principal point from calibration
    float center_x = model.cx();
    float center_y = model.cy();

    assert(cloud.cols != 0 && cloud.rows != 0);

    // Combine unit conversion (if necessary) with scaling by focal length for
    // computing (X,Y)
    double unit_scaling = DepthTraits<T>::toMeters(T(1));
    float constant_x = unit_scaling / model.fx();
    float constant_y = unit_scaling / model.fy();
    float bad_point = std::numeric_limits<float>::quiet_NaN();

    const T* depth_row = reinterpret_cast<const T*>(&depth_msg->data[0]);
    int row_step = depth_msg->step / sizeof(T);

    for (int v = 0; v < (int)depth_msg->height; ++v, depth_row += row_step) {
      for (int u = 0; u < (int)depth_msg->width; ++u) {
        T depth = depth_row[u];
        //int data_id = (u * 4) + (v * row_offset);

        // Missing points denoted by NaNs
        if (!DepthTraits<T>::valid(depth)) {
          if (range_max != 0.0) {
            depth = DepthTraits<T>::fromMeters(range_max);
          } else {
            depth = bad_point;
            continue;
          }
        }

        float x = (u - center_x) * depth * constant_x;
        float y = (v - center_y) * depth * constant_y;
        float z = DepthTraits<T>::toMeters(depth);

        cloud.at<cv::Vec3f>(u,v)[0] = x;
        cloud.at<cv::Vec3f>(u,v)[1] = y;
        cloud.at<cv::Vec3f>(u,v)[2] = z;

      }
    }
}


void disparityToDepth(const cv::Mat& disparity, float cx, float cy,
                      float fx, float fy, cv::Mat3f &depth);



template<typename T>
bool is_infinite( const T &value );
template<typename T>
bool is_nan( const T &value );
template<typename T>
bool is_valid( const T &value );


} // end namespace



#endif // UTILITIES_H
