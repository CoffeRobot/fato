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
#include <algorithm>
#include <stdexcept>
#include <iostream>

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

void getMatrices(const std::vector<cv::Point2f>& prev_pts,
                 const std::vector<float>& prev_depth,
                 const std::vector<cv::Point2f>& next_pts, float nodal_x,
                 float nodal_y, float focal_x, float focal_y,
                 Eigen::MatrixXf& A, Eigen::VectorXf& b) {
  int valid_points = prev_pts.size();

  A = Eigen::MatrixXf(valid_points * 2, 6);
  b = Eigen::VectorXf(valid_points * 2);

  float focal = (focal_x + focal_y) / 2.0;

  for (auto i = 0; i < valid_points; ++i) {
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

  translation.resize(3, 0);
  rotation.resize(3, 0);
  // solving least square to find the 6 unknown parameters: tx,ty, tz, wx, wy,
  // wz
  Eigen::VectorXf Y = (A.transpose() * A).ldlt().solve(A.transpose() * b);

  //  cout << "LQ: " << fixed << setprecision(3) << Y[0] << " " << Y[1] << " "
  //  << Y[2] << " "
  //                 << Y[3] << " " << Y[4] << " " << Y[5] << endl;

  translation[0] = Y[0];
  translation[1] = Y[1];
  translation[2] = Y[2];

  rotation[0] = Y[3];
  rotation[1] = Y[4];
  rotation[2] = Y[5];
}

Eigen::VectorXf getPoseFromFlowRobust(const std::vector<cv::Point2f>& prev_pts,
                           const std::vector<float>& prev_depth,
                           const std::vector<cv::Point2f>& next_pts,
                           float nodal_x, float nodal_y, float focal_x,
                           float focal_y, int num_iters,
                           std::vector<float>& translation,
                           std::vector<float>& rotation,
                           std::vector<int>& outliers) {
  Eigen::MatrixXf X;
  Eigen::VectorXf y;

  getMatrices(prev_pts, prev_depth, next_pts, nodal_x, nodal_y, focal_x,
              focal_y, X, y);

  // during tracking possible to initialize with the previous estimation???
  Eigen::VectorXf beta = (X.transpose() * X).ldlt().solve(X.transpose() * y);

  Eigen::VectorXf W;
  Eigen::MatrixXf XW(X.rows(), X.cols());

  for (auto i = 0; i < num_iters; ++i) {
    // cout << "iter " << i;// << endl;

    Eigen::VectorXf residuals = ((X * beta) - y);
    residuals = residuals.array().abs();

    // cout << residuals << endl;

    std::vector<float> res_vector(residuals.rows(), 0);

    for (int j = 0; j < residuals.rows(); ++j) {
      res_vector.at(j) = residuals(j);
    }

    sort(res_vector.begin(), res_vector.end());

    float residual_scale;
    if (res_vector.size() % 2 != 0)
      residual_scale = res_vector.at(res_vector.size() / 2);
    else
      residual_scale = res_vector.at(res_vector.size() / 2) +
                       res_vector.at(res_vector.size() / 2 + 1) / 2.0f;

    // avoid division by 0
    residual_scale = max(residual_scale, 0.0001f);

    residual_scale *=
        6.9460;  // constant defined in the IRLS and bisquare distance

    W = residuals / residual_scale;

    for (auto j = 0; j < W.rows(); ++j) {
      if (W(j) > 1) {
        W(j) = 0;
      } else {
        auto tmp = 1 - W(j) * W(j);
        W(j) = tmp * tmp;
      }
    }

    for (auto j = 0; j < XW.rows(); ++j) {
      for (auto k = 0; k < XW.cols(); ++k) {
        XW(j, k) = W(j) * X(j, k);
      }
    }

    beta = (XW.transpose() * X).ldlt().solve(XW.transpose() * y);
  }

  // cout << "\n here" << endl;

  int num_points = W.rows() / 2;

  for (auto j = 0; j < num_points; ++j) {
    int id = 2 * j;
    if (W(id) == 0 || W(id + 1) == 0) {
      outliers.push_back(j);
    }
  }

  translation.resize(3, 0);
  rotation.resize(3, 0);

  translation[0] = beta[0];
  translation[1] = beta[1];
  translation[2] = beta[2];

  rotation[0] = beta[3];
  rotation[1] = beta[4];
  rotation[2] = beta[5];

  return beta;
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

/**************************************************************/
/*           POSE CLASS                                       */
/**************************************************************/

Pose::Pose() { pose_ = Matrix4d(4, 4); }

Pose::Pose(Mat& r_mat, Mat& t_vect) {
  pose_ = Matrix4d(4, 4);

  if (t_vect.cols != 3 && t_vect.rows != 3) {
    cout << t_vect.cols << " " << t_vect.rows << endl;
    throw std::runtime_error(
        "Pose: bad translation vector format, 3x1 accepted");
  }

  Mat tmp_rot, tmp_t;
  r_mat.convertTo(tmp_rot, CV_64FC1);
  t_vect.convertTo(tmp_t, CV_64FC1);

  if (tmp_rot.cols == 3 && tmp_rot.rows == 3) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        pose_(i, j) = tmp_rot.at<double>(i, j);
      }
      pose_(i, 3) = tmp_t.at<double>(i);
      pose_(3, i) = 0;
    }
    pose_(3, 3) = 1;
  } else if (r_mat.cols == 3 && r_mat.cols == 1) {
    Eigen::Matrix3d rot_view;
    rot_view =
        Eigen::AngleAxisd(tmp_rot.at<double>(2), Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(tmp_rot.at<double>(1), Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(tmp_rot.at<double>(0), Eigen::Vector3d::UnitX());

    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        pose_(i, j) = rot_view(i, j);
      }
      pose_(i, 3) = tmp_t.at<double>(i);
      pose_(3, i) = 0;
    }
    pose_(3, 3) = 1;
  } else {
    throw std::runtime_error(
        "Pose: bad rotation matrix format, 3x3 or 1x3 accepted");
  }

  init_rot = r_mat;
  init_tr = t_vect;
}

Pose::Pose(Matrix4d& pose) {
  pose_ = Matrix4d(4, 4);

  for (auto i = 0; i < 4; ++i) {
    for (auto j = 0; j < 4; ++j) {
      pose_(i, j) = pose(i, j);
    }
  }
}

Pose::Pose(std::vector<double>& beta) {
  if (beta.size() != 6) {
    throw std::runtime_error("Pose: bad parameters size, should be 6");
  }

  pose_ = Matrix4d(4, 4);

  Eigen::Matrix3d rot_view;
  rot_view = Eigen::AngleAxisd(beta.at(5), Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(beta.at(4), Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(beta.at(3), Eigen::Vector3d::UnitX());

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      pose_(i, j) = rot_view(i, j);
    }
    pose_(3, i) = 0;
  }
  pose_(3, 3) = 1;

  pose_(0, 3) = beta.at(0);
  pose_(1, 3) = beta.at(1);
  pose_(2, 3) = beta.at(2);

  init_beta_ = beta;
}

Pose::Pose(VectorXf &beta)
{
    if (beta.rows() != 6) {
      throw std::runtime_error("Pose: bad parameters size, should be 6");
    }

    init_beta_.resize(6,0);

    for(auto i = 0; i < 6; ++i)
        init_beta_[i] = static_cast<double>(beta[i]);

    pose_ = Matrix4d(4, 4);

    Eigen::Matrix3d rot_view;
    rot_view = Eigen::AngleAxisd(init_beta_.at(5), Eigen::Vector3d::UnitZ()) *
               Eigen::AngleAxisd(init_beta_.at(4), Eigen::Vector3d::UnitY()) *
               Eigen::AngleAxisd(init_beta_.at(3), Eigen::Vector3d::UnitX());

    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        pose_(i, j) = rot_view(i, j);
      }
      pose_(3, i) = 0;
    }
    pose_(3, 3) = 1;

    pose_(0, 3) = init_beta_.at(0);
    pose_(1, 3) = init_beta_.at(1);
    pose_(2, 3) = init_beta_.at(2);
}

std::pair<cv::Mat, cv::Mat> Pose::toCV() const {
  cv::Mat rotation = cv::Mat(3, 3, CV_64FC1, 0.0f);
  cv::Mat translation = cv::Mat(1, 3, CV_64FC1, 0.0f);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      rotation.at<double>(i, j) = pose_(i, j);
    }
    translation.at<double>(i) = pose_(i, 3);
  }

  return pair<cv::Mat, cv::Mat>(rotation, translation);
}

std::pair<Eigen::Matrix3d, Eigen::Vector3d> Pose::toEigen() const
{
    Eigen::Matrix3d rotation_temp;
    Eigen::Vector3d translation_vect;
    for(auto i = 0; i < 3; ++i)
    {
        for(auto j = 0; j < 3; ++j)
        {
          rotation_temp(i,j) = pose_(i,j);
        }
        translation_vect(i) = pose_(i,3);
    }

    return pair<Eigen::Matrix3d, Eigen::Vector3d>(rotation_temp, translation_vect);
}

void Pose::transform(Eigen::Matrix4d& transform) { pose_ = transform * pose_; }


void Pose::transform(std::vector<double> &beta)
{

    Eigen::Matrix3d rot_view;
    rot_view = Eigen::AngleAxisd(beta.at(5), Eigen::Vector3d::UnitZ()) *
               Eigen::AngleAxisd(beta.at(4), Eigen::Vector3d::UnitY()) *
               Eigen::AngleAxisd(beta.at(3), Eigen::Vector3d::UnitX());

    Eigen::Matrix4d transform;

    for(auto i = 0; i < 3; ++i)
    {
        for(auto j = 0; j < 3; ++j)
        {
            transform(i,j) = rot_view(i,j);
        }
        transform(i,3) = beta[i];
        transform(3,i) = 0;
    }
    transform(3,3) = 1;

    pose_ = transform * pose_;
}

vector<double> Pose::getBeta() {
  Eigen::Matrix3d tmp_mat(3, 3);

  for (auto i = 0; i < 3; ++i) {
    for (auto j = 0; j < 3; ++j) tmp_mat(i, j) = pose_(i, j);
  }

  Eigen::Vector3d angles = tmp_mat.eulerAngles(2, 1, 0);

  vector<double> tmp = {pose_(0, 3), pose_(1, 3), pose_(2, 3),
                        angles(2),   angles(1),   angles(0)};

  return tmp;
}

vector<double> Pose::translation() const
{
    return vector<double>{pose_(0, 3), pose_(1, 3), pose_(2, 3)};
}

Eigen::Quaternionf Pose::rotation() const
{
    Eigen::Matrix3f tmp_mat(3, 3);

    for (auto i = 0; i < 3; ++i) {
      for (auto j = 0; j < 3; ++j) tmp_mat(i, j) = (float)pose_(i, j);
    }

    return Quaternionf(tmp_mat);
}

string Pose::str() const {
  stringstream ss;
  ss << fixed << setprecision(3);

  for (auto i = 0; i < 4; ++i) {
    for (auto j = 0; j < 4; ++j) {
      ss << pose_(i, j) << " ";
    }
    ss << "\n";
  }

  return ss.str();
}

}  // end namespace
