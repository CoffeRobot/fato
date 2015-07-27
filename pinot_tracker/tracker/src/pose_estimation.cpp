/*****************************************************************************/
/*  Copyright (c) 2015, Alessandro Pieropan                                  */
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

#include "../include/pose_estimation.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <boost/math/constants/constants.hpp>
#include "../../utilities/include/utilities.h"
#include <eigen3/Eigen/Geometry>

using namespace std;
using namespace cv;
using namespace Eigen;

namespace pinot_tracker {

void getPoseRansac(const std::vector<cv::Point3f>& model_points,
                   const std::vector<cv::Point2f>& tracked_points, int method,
                   const cv::Mat& camera_model, int iterations, float distance,
                   vector<int>& inliers, Mat& rotation, Mat& translation) {
  if (model_points.size() > 4) {
    Mat rotation_vect;
    solvePnPRansac(model_points, tracked_points, camera_model,
                   Mat::zeros(1, 8, CV_32F), rotation_vect, translation, false,
                   iterations, distance, model_points.size(), inliers,
                   CV_ITERATIVE);
    Rodrigues(rotation_vect, rotation);
    rotation.convertTo(rotation, CV_32FC1);
  }
  else
  {
     rotation = Mat(3,3,CV_32FC1,0.0f);
     translation = Mat(1,3,CV_32FC1,0.0f);
  }
}

void getPose2D(const std::vector<cv::Point2f*>& model_points,
               const std::vector<cv::Point2f*>& tracked_points, float& scale,
               float& angle) {
  vector<double> angles;
  vector<float> scales;

  angles.reserve(model_points.size() * 2);
  scales.reserve(model_points.size() * 2);

  const double pi = boost::math::constants::pi<double>();

  for (size_t i = 0; i < model_points.size(); ++i) {
    for (size_t j = 0; j < model_points.size(); j++) {
      // computing angle
      Point2f a = *tracked_points.at(i) - *tracked_points.at(j);
      Point2f b = *model_points.at(i) - *model_points.at(j);
      double val = atan2(a.y, a.x) - atan2(b.y, b.x);

      if (abs(val) > pi) {
        int sign = (val < 0) ? -1 : 1;
        val = val - sign * 2 * pi;
      }
      angles.push_back(val);

      if (i == j) continue;

      // computing scale
      auto tracked_dis =
          getDistance(tracked_points.at(i), tracked_points.at(j));
      auto model_dis = getDistance(model_points.at(i), model_points.at(j));
      if (model_dis != 0) scales.push_back(tracked_dis / model_dis);
    }
  }

  sort(angles.begin(), angles.end());
  sort(scales.begin(), scales.end());

  auto angles_size = angles.size();
  auto scale_size = scales.size();

  if (angles_size == 0)
    angle = 0;

  else if (angles_size % 2 == 0)
    angle = (angles[angles_size / 2 - 1] + angles[angles_size / 2]) / 2;
  else
    angle = angles[angles_size / 2];

  if (scale_size == 0)
    scale = 1;
  else if (scale_size % 2 == 0)
    scale = (scales[scale_size / 2 - 1] + scales[scale_size / 2]) / 2;
  else
    scale = scales[scale_size / 2];
}

Mat getRigidTransform(Mat& a, Mat& b) {
  int numRows = a.rows;

  vector<float> srcM = {0, 0, 0};
  vector<float> dstM = {0, 0, 0};

  // compute centroid of the two sets of points
  for (int i = 0; i < numRows; i++) {
    for (int j = 0; j < 3; j++) {
      srcM[j] += a.at<float>(i, j);
      dstM[j] += b.at<float>(i, j);
    }
  }

  for (int i = 0; i < 3; i++) {
    srcM[i] = srcM[i] / static_cast<float>(numRows);
    dstM[i] = dstM[i] / static_cast<float>(numRows);
  }

  Mat AA, BB, AA_T, BB_T;
  a.copyTo(AA);
  b.copyTo(BB);
  // subtracting centroid from all the points
  for (int i = 0; i < numRows; i++) {
    for (int j = 0; j < 3; j++) {
      AA.at<float>(i, j) -= srcM[j];
      BB.at<float>(i, j) -= dstM[j];
    }
  }

  transpose(AA, AA_T);

  Mat3f H;
  H = AA_T * BB;

  // cout << "H\n" << H << endl;

  Mat w, u, vt, vt_T, u_T;
  SVD::compute(H, w, u, vt);

  transpose(vt, vt_T);
  transpose(u, u_T);

  Mat3f R;
  R = vt_T * u_T;

  // reflection case
  if (determinant(R) < 0) {
    for (size_t i = 0; i < 3; i++) {
      vt.at<float>(2, i) *= -1;
    }
    transpose(vt, vt_T);
    R = vt_T * u_T;
  }

  return R;
}

Mat getRigidTransform(Mat& a, Mat& b, vector<float>& cA, vector<float>& cB) {
  int numRows = a.rows;

  vector<float>& srcM = cA;
  vector<float>& dstM = cB;

  Mat AA, BB, AA_T, BB_T;
  a.copyTo(AA);
  b.copyTo(BB);
  // subtracting centroid from all the points
  for (int i = 0; i < numRows; i++) {
    for (int j = 0; j < 3; j++) {
      AA.at<float>(i, j) -= srcM[j];
      BB.at<float>(i, j) -= dstM[j];
    }
  }

  transpose(AA, AA_T);

  Mat3f H;
  H = AA_T * BB;

  Mat w, u, vt, vt_T, u_T;
  SVD::compute(H, w, u, vt);

  transpose(vt, vt_T);
  transpose(u, u_T);

  Mat3f R;
  R = vt_T * u_T;

  // reflection case
  if (determinant(R) < 0) {
    for (size_t i = 0; i < 3; i++) {
      vt.at<float>(2, i) *= -1;
    }
    transpose(vt, vt_T);
    R = vt_T * u_T;
  }

  return R;
}

// inline Mat getRigidTransform(Mat& a, Mat& b, ofstream& file)
//{

//    file << "Debug Rigid Transform:\n\n";

//    int numRows = a.rows;

//    vector<float> srcM = { 0, 0, 0 };
//    vector<float> dstM = { 0, 0, 0 };

//    // compute centroid of the two sets of points
//    for (int i = 0; i < numRows; i++)
//    {
//        for (int j = 0; j < 3; j++)
//        {
//            srcM[j] += a.at<float>(i, j);
//            dstM[j] += b.at<float>(i, j);
//        }
//    }

//    for (int i = 0; i < 3; i++)
//    {
//        srcM[i] = srcM[i] / static_cast<float>(numRows);
//        dstM[i] = dstM[i] / static_cast<float>(numRows);
//    }

//    Mat AA, BB, AA_T, BB_T;
//    a.copyTo(AA);
//    b.copyTo(BB);
//    // subtracting centroid from all the points
//    for (int i = 0; i < numRows; i++)
//    {
//        for (int j = 0; j < 3; j++)
//        {
//            AA.at<float>(i, j) -= srcM[j];
//            BB.at<float>(i, j) -= dstM[j];
//        }
//    }

//    file << "AA\n" << AA << "\n";
//    file << "BB\n" << BB << "\n";

//    transpose(AA, AA_T);

//    Mat3f H;
//    H = AA_T * BB;

//    file << "H\n" << H << "\n";

//    Mat w, u, vt, vt_T, u_T;
//    SVD::compute(H, w, u, vt);

//    file << "U\n" << u << "\n";
//    file << "VT\n" << vt << "\n";
//    ///cout << "w\n" << w << endl;

//    transpose(vt, vt_T);
//    transpose(u, u_T);

//    Mat3f R;
//    R = vt_T * u_T;

//    // reflection case
//    if (determinant(R) < 0)
//    {
//        for (size_t i = 0; i < 3; i++)
//        {
//            vt.at<float>(2, i) *= -1;
//        }
//        transpose(vt, vt_T);
//        R = vt_T * u_T;
//    }

//    return R;
//}

// inline float getRotationError(Mat& a, Mat&b, Mat& rotation)
//{
//    Mat b_T, proj, error, proj_T;
//    transpose(b, b_T);

//    proj = rotation * b_T;
//    transpose(proj, proj_T);

//    error = a - proj_T;

//    float errorVal = 0;

//    for (int i = 0; i < error.rows; ++i)
//    {
//        errorVal += pow(error.at<float>(i, 0), 2) +
//            pow(error.at<float>(i, 1), 2) +
//            pow(error.at<float>(i, 2), 2);
//    }

//    return sqrtf(errorVal / static_cast<float>(error.rows));
//}

void rotateBBox(const vector<Point3f>& bBox, const Mat& rotation,
                vector<Point3f>& updatedBBox) {
  Mat a(4, 3, CV_32FC1);
  Mat b, b_T, a_T;

  for (size_t i = 0; i < 4; i++) {
    a.at<float>(i, 0) = bBox.at(i).x;
    a.at<float>(i, 1) = bBox.at(i).y;
    a.at<float>(i, 2) = bBox.at(i).z;
  }

  transpose(a, a_T);

  b_T = rotation * a_T;
  transpose(b_T, b);

  for (size_t i = 0; i < 4; i++) {
    updatedBBox.push_back(
        Point3f(b.at<float>(i, 0), b.at<float>(i, 1), b.at<float>(i, 2)));
  }
}

void rotatePoint(const Point3f& point, const Mat& rotation,
                 Point3f& updatedPoint) {
  Mat a(1, 3, CV_32FC1);
  a.at<float>(0) = point.x;
  a.at<float>(1) = point.y;
  a.at<float>(2) = point.z;
  Mat b, b_T, a_T;

  transpose(a, a_T);
  b_T = rotation * a_T;
  transpose(b_T, b);

  updatedPoint.x = b.at<float>(0);
  updatedPoint.y = b.at<float>(1);
  updatedPoint.z = b.at<float>(2);
}

void rotatePoint(const Vec3f& point, const Mat& rotation, Vec3f& updatedPoint) {
  Mat a(1, 3, CV_32FC1);
  a.at<float>(0) = point[0];
  a.at<float>(1) = point[1];
  a.at<float>(2) = point[2];
  Mat b, b_T, a_T;

  transpose(a, a_T);
  b_T = rotation * a_T;
  transpose(b_T, b);

  updatedPoint[0] = b.at<float>(0);
  updatedPoint[1] = b.at<float>(1);
  updatedPoint[2] = b.at<float>(2);
}

void rotatePoint(const Vec3f& point, const Mat& rotation,
                 Point3f& updatedPoint) {
  Mat a(1, 3, CV_32FC1);
  a.at<float>(0) = point[0];
  a.at<float>(1) = point[1];
  a.at<float>(2) = point[2];
  Mat b, b_T, a_T;

  transpose(a, a_T);
  b_T = rotation * a_T;
  transpose(b_T, b);

  updatedPoint.x = b.at<float>(0);
  updatedPoint.y = b.at<float>(1);
  updatedPoint.z = b.at<float>(2);
}

void rotationVecToMat(const Mat& vec, Mat& mat) {}

}  // end namespace
