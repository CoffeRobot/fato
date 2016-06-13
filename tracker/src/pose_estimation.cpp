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
#include "../include/utilities.h"
#include <eigen3/Eigen/Geometry>

using namespace std;
using namespace cv;
using namespace Eigen;

namespace fato {

void getPoseRansac(const std::vector<cv::Point3f>& model_points,
                   const std::vector<cv::Point2f>& tracked_points,
                   const cv::Mat& camera_model, int iterations, float distance,
                   vector<int>& inliers, Mat& rotation, Mat& translation) {
  if (model_points.size() > 4) {
    Mat rotation_vect;
    solvePnPRansac(model_points, tracked_points, camera_model,
                   Mat::zeros(1, 8, CV_32F), rotation_vect, translation, false,
                   iterations, distance, model_points.size(), inliers, CV_P3P);

    try {
      Rodrigues(rotation_vect, rotation);
      rotation.convertTo(rotation, CV_32FC1);
    } catch (cv::Exception& e) {
      cout << "Error estimating ransac rotation: " << e.what() << endl;
    }

  } else {
    rotation = Mat(3, 3, CV_32FC1, 0.0f);
    translation = Mat(1, 3, CV_32FC1, 0.0f);
  }
}


void getPoseFromFlow(const std::vector<cv::Point2f>& prev_pts,
                     const std::vector<float>& prev_depth,
                     const std::vector<cv::Point2f>& next_pts, float nodal_x,
                     float nodal_y, float focal_x, float focal_y,
                     vector<float>& translation, vector<float>& rotation) {
  int valid_points = prev_pts.size();

  Eigen::MatrixXf A(valid_points * 2, 6);
  Eigen::VectorXf b(valid_points * 2);

  float focal = (focal_x + focal_y) / 2.0;

  for (auto i = 0; i < prev_pts.size(); ++i) {
    float mz = prev_depth.at(i);
    float x = next_pts.at(i).x - nodal_x;
    float y = next_pts.at(i).y - nodal_y;
    float xy = x * y / focal;

    int id = 2 * i;
    int id2 = id + 1;
    // first equation for X, u flow
    A(id, 0) = focal / mz;
    A(id, 1) = 0;
    A(id, 2) = -x / mz;
    A(id, 3) = -xy;
    A(id, 4) = focal + (x * x) / focal;
    A(id, 5) = -y;
    // second equation for X, v flow
    A(id2, 0) = 0;
    A(id2, 1) = focal / mz;
    A(id2, 2) = -y / mz;
    A(id2, 3) = -(focal + (x * x) / focal);
    A(id2, 4) = xy;
    A(id2, 5) = x;
    // setting u,v flow in b vector
    b[id] = next_pts.at(i).x - prev_pts.at(i).x;
    b[id2] = next_pts.at(i).y - prev_pts.at(i).y;
  }

  translation.resize(3,0);
  rotation.resize(3,0);
  // solving least square to find the 6 unknown parameters: tx,ty, tz, wx, wy, wz
  Eigen::VectorXf Y = (A.transpose() * A).ldlt().solve(A.transpose() * b);

  cout << "LQ: " << fixed << setprecision(3) << Y[0] << " " << Y[1] << " " << Y[2] << " "
                 << Y[3] << " " << Y[4] << " " << Y[5] << endl;

  translation[0] = Y[0];
  translation[1] = Y[1];
  translation[2] = Y[2];

  rotation[0] = Y[3];
  rotation[1] = Y[4];
  rotation[2] = Y[5];
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

// TODO: implement ransac approach for better SVD estimation
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
