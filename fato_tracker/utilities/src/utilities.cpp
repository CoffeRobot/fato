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

#include "../include/utilities.h"
#include "../include/device_1d.h"
#include <iostream>

namespace fato {

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
                  const cv::Point3f* src, cv::Point2f& dst) {
  if (src->z == 0) return false;

  dst.x = (focal * src->x / src->z) + center.x;
  dst.y = (center.y - (focal * src->y / src->z));

  if (dst.x < 0 || dst.x > center.x * 2) return false;
  if (dst.y < 0 || dst.y > center.y * 2) return false;

  if (isnan(dst.x) || isnan(dst.y)) return false;

  return true;
}

void depthTo3d(const cv::Mat& disparity, float cx, float cy, float fx, float fy,
               cv::Mat3f& depth) {
  int cols = disparity.cols;
  int rows = disparity.rows;

  assert(depth.cols != 0 && depth.rows != 0);
  assert(fx != 0 && fy != 0);

  float inv_fx = 1.0 / fx;
  float inv_fy = 1.0 / fy;

  for (size_t y = 0; y < rows; y++) {
    for (size_t x = 0; x < cols; x++) {
      uint16_t val = disparity.at<uint16_t>(y, x);
      float d = static_cast<float>(val);

      if (!is_valid(val)) continue;
      if (val == 0) continue;

      if (d == 0) continue;

      float xp = x - cx;
      float yp = -(y - cy);

      float Z = DepthTraits<uint16_t>::toMeters(d);

      float X = xp * Z * inv_fx;
      float Y = yp * Z * inv_fy;

      depth.at<cv::Vec3f>(y, x) = cv::Vec3f(X, Y, Z);
    }
  }
}

void initializeCUDARuntime(int device) {
  cudaSetDevice(device);
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  // dummy memcpy to init cuda runtime
  util::Device1D<float> d_dummy(1);
  std::vector<float> h_dummy(1);
  d_dummy.copyFrom(h_dummy);

  if (cudaGetLastError() != cudaSuccess)
    throw std::runtime_error(
        std::string("initializeCUDARuntime: CUDA initialization problem\n"));
}

TimerGPU::TimerGPU(cudaStream_t stream) : stream_(stream) {
  cudaEventCreate(&start_);
  cudaEventCreate(&stop_);
  cudaEventRecord(start_, stream);
}

TimerGPU::~TimerGPU() {
  cudaEventDestroy(start_);
  cudaEventDestroy(stop_);
}

float TimerGPU::read() {
  cudaEventRecord(stop_, stream_);
  cudaEventSynchronize(stop_);
  float time;
  cudaEventElapsedTime(&time, start_, stop_);
  return time;
}

void TimerGPU::reset() { cudaEventRecord(start_, stream_); }

//void cvToPcl(const cv::Mat3f& points, pcl::PointCloud<pcl::PointXYZ>& cloud) {
//  int width = points.cols, height = points.rows;

//  cloud.points.resize(width * height);
//  cloud.width = width;
//  cloud.height = height;

//  for (int v = 0; v < height; ++v) {
//    for (int u = 0; u < width; ++u) {
//      auto& point = points.at<cv::Vec3f>(v, u);
//      pcl::PointXYZ& p = cloud(u, v);
//      p.x = point[0];
//      p.y = point[1];
//      p.z = point[2];
//    }
//  }
//}

//void cvToPcl(const cv::Mat3f& points, const cv::Mat1b& mask,
//             pcl::PointCloud<pcl::PointXYZ>& cloud) {
//  int width = points.cols, height = points.rows;

//  cloud.points.resize(width * height);
//  cloud.width = width;
//  cloud.height = height;

//  for (int v = 0; v < height; ++v) {
//    for (int u = 0; u < width; ++u) {
//      auto& point = points.at<cv::Vec3f>(v, u);
//      pcl::PointXYZ& p = cloud(u, v);
//      if (mask.at<uchar>(u, v) == 255) {
//        p.x = point[0];
//        p.y = point[1];
//        p.z = point[2];
//      } else {
//        p.x = p.y = p.z = 0;
//      }
//    }
//  }
//}
}  // end namespace
