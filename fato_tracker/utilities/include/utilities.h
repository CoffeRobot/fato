/*****************************************************************************/
/*  Copyright (c) 2016, Alessandro Pieropan                                  */
/*  All rights reserved.                                                     */
/*                                                                           */
/*  Redistribution and use in source and binary forms, with or without       */
/*  modification, are permitted provided that the following conditions       */
/*  are met:                                                                 */
/*                                                                           */
/*  1. Redistributions of source code must retain the above copyright        */
/*  notice, this list of conditions and the following disclaimer.            */
/*                                                                           */
/*  2. Redistributions in binary form must reproduce the above copyright     */
/*  notice, this list of conditions and the following disclaimer in the      */
/*  documentation and/or other materials provided with the distribution.     */
/*                                                                           */
/*  3. Neither the name of the copyright holder nor the names of its         */
/*  contributors may be used to endorse or promote products derived from     */
/*  this software without specific prior written permission.                 */
/*                                                                           */
/*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      */
/*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        */
/*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    */
/*  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     */
/*  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   */
/*  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         */
/*  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    */
/*  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    */
/*  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      */
/*  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    */
/*  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     */
/*****************************************************************************/

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
//#include <pcl/point_types.h>
//#include <pcl/point_cloud.h>
#include <cuda_runtime.h>


#include "../include/traits.h"

namespace fato{

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
  return sqrt( (a.x - b.x) * (a.x - b.x) +
               (a.y - b.y) * (a.y - b.y));
//  return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

inline float getDistance(const cv::Point2f* a, const cv::Point2f* b)
{
    return sqrt(pow(a->x - b->x, 2) + pow(a->y - b->y, 2));
}

inline float getDistance(const cv::Point3f& a, const cv::Point3f& b)
{
    return sqrt( (a.x - b.x) * (a.x- b.x) +
                 (a.y - b.y) * (a.y- b.y) +
                 (a.z - b.z) * (a.z- b.z));
}

inline float getDistance(const cv::Vec3f& a, const cv::Vec3f& b)
{
    return sqrt( (a[0] - b[0]) * (a[0] - b[0]) +
                 (a[1] - b[1]) * (a[1] - b[1]) +
                 (a[2] - b[2]) * (a[2] - b[2]));
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

//void cvToPcl(const cv::Mat3f& points, pcl::PointCloud<pcl::PointXYZ>& cloud);

//void cvToPcl(const cv::Mat3f& points, const cv::Mat1b& mask,
//             pcl::PointCloud<pcl::PointXYZ>& cloud);

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

void initializeCUDARuntime(int device = 0);

class TimerGPU {

public:
  TimerGPU(cudaStream_t stream_ = 0);
  ~TimerGPU();
  float read();
  void reset();

private:
  const cudaStream_t stream_;
  cudaEvent_t start_, stop_;
};


} // end namespace



#endif // UTILITIES_H
