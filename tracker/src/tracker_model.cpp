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

#include "../include/tracker_model.h"

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#define _USE_MATH_DEFINES
#include <math.h>
#include <limits>
#include <algorithm>
#include <utility>
#include <iomanip>

#include <hdf5_file.h>

#include "../include/pose_estimation.h"
#include "../include/profiler.h"
#include "../include/DBScanClustering.h"
#include "../../utilities/include/utilities.h"
#include "../include/epnp.h"
#include "../include/utilities.h"

using namespace std;
using namespace cv;

namespace fato {

TrackerMB::TrackerMB() : matcher_confidence_(0.75), matcher_ratio_(0.75) {}

TrackerMB::TrackerMB(Config& params, int descriptor_type,
                     unique_ptr<FeatureMatcher> matcher)
    : matcher_confidence_(0.75), matcher_ratio_(0.75) {
  feature_detector_ = std::move(matcher);
  stop_matcher = false;

  debug_file.open("/home/alessandro/debug/object_pose.txt");
}

TrackerMB::~TrackerMB() { taskFinished(); }

void TrackerMB::addModel(const Mat& descriptors,
                         const std::vector<Point3f>& points) {
  m_initDescriptors = descriptors.clone();

  target_object_.descriptors_ = descriptors.clone();
  target_object_.model_points_ = points;
}

void TrackerMB::addModel(const string& h5_file) {
  util::HDF5File out_file(h5_file);
  std::vector<uchar> descriptors;
  std::vector<int> dscs_size, point_size;
  std::vector<float> points;
  // int test_feature_size;
  out_file.readArray<uchar>("descriptors", descriptors, dscs_size);
  out_file.readArray<float>("positions", points, point_size);

  vectorToMat(descriptors, dscs_size, m_initDescriptors);

  vector<Point3f> pts(point_size[0], cv::Point3f(0, 0, 0));

  for (int i = 0; i < point_size[0]; ++i) {
    pts.at(i) =
        cv::Point3f(points[3 * i], points[3 * i + 1], points[3 * i + 2]);
  }

  target_object_.init(pts, m_initDescriptors);

  // initFilter();

  feature_detector_->setTarget(m_initDescriptors);

  std::cout << "Model loaded with " << target_object_.model_points_.size()
            << " key points and " << m_initDescriptors.rows << " descriptors "
            << std::endl;
}

void TrackerMB::setCameraMatrix(Mat& camera_matrix) {
  camera_matrix_ = camera_matrix.clone();
}

void TrackerMB::learnBackground(const Mat& rgb) {
  Mat gray;
  cvtColor(rgb, gray, CV_BGR2GRAY);
  feature_detector_->extractTarget(gray);

  Mat bg_descriptors = feature_detector_->getTargetDescriptors();

  cout << "Learning the background, keypoints: " << bg_descriptors.rows << endl;
  cout << " - prev descriptors " << m_initDescriptors.rows << endl;

  cv::vconcat(m_initDescriptors, bg_descriptors, m_initDescriptors);
  cout << " - curr descriptors " << m_initDescriptors.rows << endl;

  feature_detector_->setTarget(m_initDescriptors);
}

void TrackerMB::resetTarget() { target_object_.resetPose(); }

void TrackerMB::setMatcerParameters(float confidence, float second_ratio) {
  matcher_confidence_ = confidence;
  matcher_ratio_ = second_ratio;
}

void TrackerMB::getOpticalFlow(const Mat& prev, const Mat& next,
                               Target& target) {
  if (target_object_.active_points.size() <= 0) return;

  vector<Point2f> next_points, prev_points;
  vector<uchar> next_status, prev_status;
  vector<float> next_errors, prev_errors;

  calcOpticalFlowPyrLK(prev, next, target.active_points, next_points,
                       next_status, next_errors);
  calcOpticalFlowPyrLK(next, prev, next_points, prev_points, prev_status,
                       prev_errors);

  vector<int> to_remove;

  // TODO: first implementation need to optimize
  target.prev_points_ = target.active_points;

  for (int i = 0; i < next_points.size(); ++i) {
    float error = fato::getDistance(target.active_points.at(i), prev_points[i]);

    const int& id = target.active_to_model_.at(i);
    KpStatus& s = target.point_status_.at(id);

    if (prev_status[i] == 1 && error < 5) {
      // const int& id = ids[i];

      if (s == KpStatus::MATCH) s = KpStatus::PNP;

      // saving previous location of the point, needed to compute pose from
      // motion
      target.active_points.at(i) = next_points[i];

    } else {
      to_remove.push_back(i);
    }
  }

  if (!target_object_.isConsistent())
    cout << "OF1: object not consistent" << endl;
  target_object_.removeInvalidPoints(to_remove);
  if (!target_object_.isConsistent())
    cout << "OF2: object not consistent" << endl;

  // cout << "OF 4" << endl;
}

void TrackerMB::computeNextSequential(Mat& rgb) {
  trackSequential(rgb);
  detectSequential(rgb);
}

void TrackerMB::taskFinished() {
  m_isRunning = false;
  m_trackerCondition.notify_one();
  m_detectorCondition.notify_one();
}

void TrackerMB::trackSequential(Mat& next) {
  // cout << "Track target " << endl;
  /*************************************************************************************/
  /*                       LOADING IMAGES */
  /*************************************************************************************/
  Mat next_gray;
  cvtColor(next, next_gray, CV_BGR2GRAY);
  /*************************************************************************************/
  /*                       TRACKING */
  /*************************************************************************************/
  // TOFIX: crushing here because of empty points vector
  getOpticalFlow(prev_gray_, next_gray, target_object_);
  if (!target_object_.isConsistent())
    cout << "OF: Error status of object is incosistent!!" << endl;
  /*************************************************************************************/
  /*                             POSE ESTIMATION /
  /*************************************************************************************/
  // finding all the points in the model that are active
  vector<Point3f> model_pts;
  for (int i = 0; i < target_object_.active_points.size(); ++i) {
    const int& id = target_object_.active_to_model_.at(i);
    model_pts.push_back(target_object_.model_points_.at(id));
  }

  if (model_pts.size() > 4) {
    vector<int> inliers;
    poseFromPnP(model_pts, inliers);
    target_object_.pnp_pose =
        Pose(target_object_.rotation, target_object_.translation);
    auto inliers_count = inliers.size();
    removeOutliers(inliers);
    validateInliers(inliers);

    if (!target_object_.target_found_) {
      // removing outliers using pnp ransac

      // cout << "object found!" << endl;
      Eigen::MatrixXd projection_matrix = getProjectionMatrix(
          target_object_.rotation, target_object_.translation);

      model_pts.clear();
      for (int i = 0; i < target_object_.active_points.size(); ++i) {
        const int& id = target_object_.active_to_model_.at(i);
        model_pts.push_back(target_object_.model_points_.at(id));
      }
      vector<float> projected_depth;
      projectPointsDepth(model_pts, projection_matrix, projected_depth);

      for (auto i = 0; i < target_object_.active_points.size(); ++i) {
        const int& id = target_object_.active_to_model_.at(i);
        target_object_.projected_depth_.at(id) = projected_depth.at(i);
        // cout << setprecision(3) << fixed << projected_depth.at(i) << endl;
      }
      if (inliers_count > 4) {
        target_object_.target_found_ = true;
        target_object_.pose_ = projection_matrix * target_object_.pose_;
        target_object_.kalman_pose_ =
            projection_matrix * target_object_.kalman_pose_;
      }

      target_object_.flow_pose =
          Pose(target_object_.rotation, target_object_.translation);

      initFilter(kalman_pose_pnp_, projection_matrix);
      initFilter(kalman_pose_flow_, projection_matrix);

      target_object_.rotation_kalman = target_object_.rotation.clone();
      target_object_.translation_kalman = target_object_.translation.clone();

    } else {
      predictPose();
      poseFromFlow();
    }
  } else {
    // cout << "object lost" << endl;
    target_object_.rotation = Mat(3, 3, CV_64FC1, 0.0f);
    setIdentity(target_object_.rotation);
    target_object_.translation = Mat(1, 3, CV_64FC1, 0.0f);
    target_object_.rotation_vec = Mat(1, 3, CV_64FC1, 0.0f);
    target_object_.target_found_ = false;
    target_object_.resetPose();
  }
  /*************************************************************************************/
  /*                       SWAP MEMORY */
  /*************************************************************************************/
  next_gray.copyTo(prev_gray_);
}

void TrackerMB::detectSequential(Mat& next) {
  if (stop_matcher) return;

  vector<KeyPoint> keypoints;
  Mat gray, descriptors;
  cvtColor(next, gray, CV_BGR2GRAY);
  /*************************************************************************************/
  /*                       FEATURE MATCHING */
  /*************************************************************************************/
  vector<vector<DMatch>> matches;
  // cout << "Before match" << endl;
  feature_detector_->match(gray, keypoints, descriptors, matches);
  // std::cout << "Number of matches " << matches.size() << std::endl;
  // cout << "After match" << endl;
  // m_customMatcher.matchHamming(descriptors, m_initDescriptors, 2, matches);
  /*************************************************************************************/
  /*                       ADD FEATURES TO TRACKER */
  /*************************************************************************************/
  if (matches.size() < 2) return;
  if (matches.at(0).size() != 2) return;

  vector<cv::Point2f>& active_points = target_object_.active_points;
  vector<KpStatus>& point_status = target_object_.point_status_;

  int bg_count = 0;

  for (size_t i = 0; i < matches.size(); i++) {
    const DMatch& fst_match = matches.at(i).at(0);
    const DMatch& scd_match = matches.at(i).at(1);

    const int& model_idx = fst_match.trainIdx;
    const int& match_idx = fst_match.queryIdx;

    if (match_idx < 0 || match_idx >= keypoints.size()) continue;
    // the descriptor mat includes the objects and the background stacked
    // vertically, therefore
    // the index shoud be less than the target object points
    if (model_idx < 0 ||
        model_idx >=
            m_initDescriptors.rows)  // target_object_.model_points_.size())
      continue;

    if (model_idx >= target_object_.model_points_.size()) {
      bg_count++;
      continue;
    }

    float confidence = 1 - (matches[i][0].distance / 512.0);
    auto ratio = (fst_match.distance / scd_match.distance);

    KpStatus& s = point_status.at(model_idx);

    if (confidence < 0.80f) continue;

    if (ratio > 0.8f) continue;

    if (s != KpStatus::LOST) continue;

    // new interface using target class and keeping a list of pointers for speed
    if (point_status.at(model_idx) == KpStatus::LOST) {
      active_points.push_back(keypoints.at(match_idx).pt);
      target_object_.active_to_model_.push_back(model_idx);
      point_status.at(model_idx) = KpStatus::MATCH;
    }
  }
}

void TrackerMB::projectPointsToModel(const Point2f& model_centroid,
                                     const Point2f& upd_centroid,
                                     const float angle, const float scale,
                                     const std::vector<Point2f>& pts,
                                     std::vector<Point2f>& proj_pts) {
  Mat2f rotMat(2, 2);
  rotMat.at<float>(0, 0) = cosf(angle);
  rotMat.at<float>(0, 1) = -sinf(angle);
  rotMat.at<float>(1, 0) = sinf(angle);
  rotMat.at<float>(1, 1) = cosf(angle);

  for (auto i = 0; i < pts.size(); i++) {
    Point2f rm = pts[i] - upd_centroid;
    if (scale != 0) {
      rm.x = rm.x / scale;
      rm.y = rm.y / scale;
    }
    rm = mult(rotMat, rm);

    proj_pts.push_back(model_centroid + rm);
  }
}

void TrackerMB::removeOutliers(vector<int>& inliers) {
  vector<int> to_remove;
  auto check_point = [&](int p) {

    int id = target_object_.active_to_model_.at(p);
    if (target_object_.point_status_[id] == KpStatus::PNP) {
      to_remove.push_back(p);
    }
  };

  if (inliers.size() > 0) {
    for (int i = 0; i < inliers.size(); ++i) {
      int start;
      int end;
      if (i == 0) {
        start = 0;
        end = inliers.at(i);
      } else {
        start = inliers.at(i - 1) + 1;
        end = inliers.at(i);
      }

      for (int j = start; j < end; ++j) check_point(j);
    }

    for (int j = inliers.back() + 1; j < target_object_.active_points.size();
         ++j) {
      check_point(j);
    }
  } else {
    for (int j = 0; j < target_object_.active_points.size(); ++j) {
      check_point(j);
    }
  }
  target_object_.removeInvalidPoints(to_remove);
}

void TrackerMB::validateInliers(std::vector<int>& inliers) {
  for (auto i = 0; i < target_object_.active_points.size(); ++i) {
    int id = target_object_.active_to_model_.at(i);
    KpStatus& s = target_object_.point_status_.at(id);
    if (s == KpStatus::PNP) s = KpStatus::TRACK;
  }
}

void TrackerMB::poseFromPnP(vector<Point3f>& model_pts, vector<int>& inliers) {
  float ransac_error = 2.0f;
  int num_iters = 50;

  solvePnPRansac(model_pts, target_object_.active_points, camera_matrix_,
                 Mat::zeros(1, 8, CV_64F), target_object_.rotation_vec,
                 target_object_.translation, true, num_iters, ransac_error,
                 model_pts.size(), inliers, CV_EPNP);

  try {
    Rodrigues(target_object_.rotation_vec, target_object_.rotation);
  } catch (cv::Exception& e) {
    cout << "Error estimating ransac rotation: " << e.what() << endl;
  }
}

void TrackerMB::poseFromFlow() {
  vector<Point2f> prev_points;
  vector<Point2f> curr_points;
  vector<float> depths;

  prev_points.reserve(target_object_.active_points.size());
  curr_points.reserve(target_object_.active_points.size());
  depths.reserve(target_object_.active_points.size());

  // cout << "BEFORE POSE" << endl;

  for (auto i = 0; i < target_object_.active_points.size(); ++i) {
    int id = target_object_.active_to_model_.at(i);
    if (!is_nan(target_object_.projected_depth_.at(id))) {
      prev_points.push_back(target_object_.prev_points_.at(i));
      curr_points.push_back(target_object_.active_points.at(i));
      depths.push_back(target_object_.projected_depth_.at(id));
      // cout << setprecision(3) << fixed <<
      // target_object_.projected_depth_.at(id) << " " << endl;
    }
  }

  float nodal_x = camera_matrix_.at<double>(0, 2);
  float nodal_y = camera_matrix_.at<double>(1, 2);
  float focal_x = camera_matrix_.at<double>(0, 0);
  float focal_y = camera_matrix_.at<double>(1, 1);

  vector<float> t, r;
  //  getPoseFromFlow(prev_points, depths, curr_points, nodal_x, nodal_y,
  //  focal_x,
  //                  focal_y, t, r);

  vector<int> outliers;
  getPoseFromFlowRobust(prev_points, depths, curr_points, nodal_x, nodal_y,
                        focal_x, focal_y, 10, t, r, outliers);
  //  cout << "outliers size " << outliers.size() << endl;

  //  target_object_.removeInvalidPoints(outliers);

  Eigen::Matrix3d rot_view;
  rot_view = Eigen::AngleAxisd(r.at(2), Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(r.at(1), Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(r.at(0), Eigen::Vector3d::UnitX());

  Eigen::Matrix4d proj_mat(4, 4);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      proj_mat(i, j) = rot_view(i, j);
    }
    proj_mat(i, 3) = t[i];
    proj_mat(3, i) = 0;
  }
  proj_mat(3, 3) = 1;

  // cout << "z check " << target_object_.pose_(2,3) << " " <<
  // target_object_.kalman_pose_(2,3) << endl;

  predictPoseFlow(t, r);

  target_object_.pose_ = proj_mat * target_object_.pose_;
  // cout << "z check2 " << target_object_.pose_(2,3) << " " <<
  // target_object_.kalman_pose_(2,3) << endl;

  vector<Point3f> model_pts;
  for (int i = 0; i < target_object_.active_points.size(); ++i) {
    const int& id = target_object_.active_to_model_.at(i);
    model_pts.push_back(target_object_.model_points_.at(id));
  }
  vector<float> projected_depth;
  projectPointsDepth(model_pts, target_object_.pose_, projected_depth);

  // cout << "FLOW POSE" << endl;
  for (auto i = 0; i < target_object_.active_points.size(); ++i) {
    const int& id = target_object_.active_to_model_.at(i);
    target_object_.projected_depth_.at(id) = projected_depth.at(i);
    // cout << setprecision(3) << fixed << projected_depth.at(i) << " " << endl;
  }
}

void TrackerMB::projectPointsDepth(std::vector<Point3f>& points,
                                   Eigen::MatrixXd& projection,
                                   std::vector<float>& projected_depth) {
  Eigen::MatrixXd eig_points(4, points.size());

  for (auto i = 0; i < points.size(); ++i) {
    eig_points(0, i) = points.at(i).x;
    eig_points(1, i) = points.at(i).y;
    eig_points(2, i) = points.at(i).z;
    eig_points(3, i) = 1;
  }

  Eigen::MatrixXd eig_proj = projection * eig_points;

  projected_depth.resize(points.size(), 0);

  for (auto i = 0; i < points.size(); ++i) {
    projected_depth.at(i) = eig_proj(2, i);
  }
}

void TrackerMB::initFilter(KalmanFilter& filter, Eigen::MatrixXd& projection) {
  filter = KalmanFilter(18, 6);

  Mat A = Mat::eye(18, 18, CV_32FC1);

  double dt = 1 / 30.0;

  cout << "initializing kalman filter" << endl;

  for (auto i = 0; i < 9; ++i) {
    auto id_vel = i + 3;
    auto id_acc = i + 6;
    auto id_vel2 = i + 12;
    auto id_acc2 = i + 15;

    if (id_vel < 9) A.at<float>(i, id_vel) = dt;
    if (id_acc < 9) A.at<float>(i, id_acc) = 0.5 * dt * dt;
    if (id_vel2 < 18) A.at<float>(i + 9, id_vel2) = dt;
    if (id_acc2 < 18) A.at<float>(i + 9, id_acc2) = 0.5 * dt * dt;
  }

  filter.transitionMatrix = A.clone();

  filter.measurementMatrix.at<float>(0, 0) = 1;
  filter.measurementMatrix.at<float>(1, 1) = 1;
  filter.measurementMatrix.at<float>(2, 2) = 1;
  filter.measurementMatrix.at<float>(3, 9) = 1;
  filter.measurementMatrix.at<float>(4, 10) = 1;
  filter.measurementMatrix.at<float>(5, 11) = 1;

  setIdentity(filter.processNoiseCov, Scalar::all(1e-4));
  setIdentity(filter.measurementNoiseCov, Scalar::all(1e-2));
  setIdentity(filter.errorCovPost, Scalar::all(.1));

  filter.measurementNoiseCov.at<float>(0, 0) = 0.01;
  filter.measurementNoiseCov.at<float>(1, 1) = 0.01;
  filter.measurementNoiseCov.at<float>(2, 2) = 0.01;
  filter.measurementNoiseCov.at<float>(3, 3) = 0.01;
  filter.measurementNoiseCov.at<float>(4, 4) = 0.01;
  filter.measurementNoiseCov.at<float>(5, 5) = 0.01;

  //  ofstream file("/home/alessandro/debug/object_pose.txt",
  //                ofstream::out | ofstream::app);

  Pose& pnp_pose = target_object_.pnp_pose;

  vector<double> beta = pnp_pose.getBeta();

  filter.statePre.at<float>(0) = beta.at(0);  // projection(0, 3);
  filter.statePre.at<float>(1) = beta.at(1);  // projection(1, 3);
  filter.statePre.at<float>(2) = beta.at(2);  // projection(2, 3);
  filter.statePre.at<float>(3) = 0;
  filter.statePre.at<float>(4) = 0;
  filter.statePre.at<float>(5) = 0;
  filter.statePre.at<float>(6) = 0;
  filter.statePre.at<float>(7) = 0;
  filter.statePre.at<float>(8) = 0;

  //  Eigen::Matrix3d tmp_mat(3, 3);

  //  for (auto i = 0; i < 3; ++i) {
  //    for (auto j = 0; j < 3; ++j) tmp_mat(i, j) = projection(i, j);
  //  }

  //  Eigen::Vector3d angles = tmp_mat.eulerAngles(2, 1, 0);

  //  file << "pnp kalman init angle \n" << angles(0) << " " << angles(1) << " "
  //       << angles(2) << "\n";

  filter.statePre.at<float>(9) = beta.at(3);   // angles(2);
  filter.statePre.at<float>(10) = beta.at(4);  // angles(1);
  filter.statePre.at<float>(11) = beta.at(5);  // angles(0);
  filter.statePre.at<float>(12) = 0;
  filter.statePre.at<float>(13) = 0;
  filter.statePre.at<float>(14) = 0;
  filter.statePre.at<float>(15) = 0;
  filter.statePre.at<float>(16) = 0;
  filter.statePre.at<float>(17) = 0;

  //  kalman_pose_flow_.transitionMatrix =
  //      kalman_pose_pnp_.transitionMatrix.clone();
  //  kalman_pose_flow_.statePre = kalman_pose_pnp_.statePre.clone();
  //  kalman_pose_flow_.measurementMatrix =
  //      kalman_pose_pnp_.measurementMatrix.clone();
  //  kalman_pose_flow_.measurementNoiseCov =
  //      kalman_pose_pnp_.measurementNoiseCov.clone();
  //  kalman_pose_flow_.processNoiseCov =
  //  kalman_pose_pnp_.processNoiseCov.clone();
  //  kalman_pose_flow_.errorCovPost = kalman_pose_pnp_.errorCovPost.clone();

  //  for (auto i = 0; i < A.rows; ++i) {
  //    for (auto j = 0; j < A.cols; ++j)
  //      file << fixed << setprecision(4) << A.at<float>(i, j) << " ";
  //    file << "\n";
  //  }

  //  file << "\n";
  //  for (auto i = 0; i < kalman_pose_pnp_.measurementMatrix.rows; ++i) {
  //    for (auto j = 0; j < kalman_pose_pnp_.measurementMatrix.cols; ++j)
  //      file << fixed << setprecision(4)
  //           << kalman_pose_pnp_.measurementMatrix.at<float>(i, j) << " ";
  //    file << "\n";
  //  }

  //  file << "\n";
  //  for (auto i = 0; i < kalman_pose_pnp_.measurementNoiseCov.rows; ++i) {
  //    for (auto j = 0; j < kalman_pose_pnp_.measurementNoiseCov.cols; ++j)
  //      file << fixed << setprecision(4)
  //           << kalman_pose_pnp_.measurementNoiseCov.at<float>(i, j) << " ";
  //    file << "\n";
  //  }

  //  file << "\n initialization \n";
  //  file << "pnp statePre\n";
  //  for (auto i = 0; i < kalman_pose_pnp_.statePre.rows; ++i) {
  //    for (auto j = 0; j < kalman_pose_pnp_.statePre.cols; ++j)
  //      file << fixed << setprecision(4)
  //           << kalman_pose_pnp_.statePre.at<float>(i, j) << " ";
  //  }
  //  file << "\n";

  //  file << "flow statePre\n";
  //  for (auto i = 0; i < kalman_pose_flow_.statePre.rows; ++i) {
  //    for (auto j = 0; j < kalman_pose_flow_.statePre.cols; ++j)
  //      file << fixed << setprecision(4)
  //           << kalman_pose_flow_.statePre.at<float>(i, j) << " ";
  //  }

  //  file << "\n initialization done \n\n";
  //  file.close();

  cout <<  "initialization done " << endl;
}

void TrackerMB::predictPose() {
  ofstream file("/home/alessandro/debug/object_pose.txt",
                ofstream::out | ofstream::app);

  // file << "----------------- PNP PREDICTION ---------------- \n\n";

  auto toString = [](cv::Mat& mat) {
    stringstream ss;
    ss << fixed << setprecision(5);
    for (auto i = 0; i < mat.rows; ++i) {
      ss << mat.at<float>(i, 0) << " ";
    }
    return ss.str();
  };

  // file << "state pred \n" << toString(kalman_pose_pnp_.statePre) << "\n";

  Mat prediction = kalman_pose_pnp_.predict();

  // file << "pnp pred \n";
  file << "pred: " << prediction.at<float>(9) << " " << prediction.at<float>(10)
       << " " << prediction.at<float>(11) << "\n";

  Pose& pnp_pose = target_object_.pnp_pose;

  Eigen::Matrix3d tmp_mat(3, 3);

  // file << "rotation matrix \n";
  for (auto i = 0; i < 3; ++i) {
    for (auto j = 0; j < 3; ++j) {
      tmp_mat(i, j) = target_object_.rotation.at<double>(i, j);
      // file << tmp_mat(i, j) << " ";
    }
    // file << "\n";
  }

  Eigen::Vector3d angles = tmp_mat.eulerAngles(2, 1, 0);

  vector<double> beta = pnp_pose.getBeta();

  file << "euler angles " << angles(0) << " " << angles(1) << " " << angles(2)
       << "\n";

  Mat_<float> measurement(6, 1);
  measurement(0) = beta.at(0);  // target_object_.translation.at<double>(0);
  measurement(1) = beta.at(1);  // target_object_.translation.at<double>(1);
  measurement(2) = beta.at(2);  // target_object_.translation.at<double>(2);
  measurement(3) = beta.at(3);  //;
  measurement(4) = beta.at(4);  //;
  measurement(5) = beta.at(5);  //
  // file << "pnp meas \n";
  // file << toString(measurement) << "\n";

  Mat estimated = kalman_pose_pnp_.correct(measurement);
  // file << "pnp estimated \n";
  // file << toString(estimated) << "\n";

  Mat_<double> rot(3, 1);
  Mat_<double> tr(3, 1);

  tr(0) = estimated.at<float>(0);
  tr(1) = estimated.at<float>(1);
  tr(2) = estimated.at<float>(2);

  rot(0) = estimated.at<float>(9);
  rot(1) = estimated.at<float>(10);
  rot(2) = estimated.at<float>(11);

  vector<double> tmp = {estimated.at<float>(0),  estimated.at<float>(1),
                        estimated.at<float>(2),  estimated.at<float>(9),
                        estimated.at<float>(10), estimated.at<float>(11)};

  target_object_.kal_pnp_pose = Pose(tmp);

  target_object_.translation_kalman = tr;

  file << "corr: " << estimated.at<float>(9) << " " << estimated.at<float>(10)
       << " " << estimated.at<float>(11) << "\n\n";

  Eigen::Matrix3f m1;
  m1 = Eigen::AngleAxisf(angles(0), Eigen::Vector3f::UnitZ()) *
       Eigen::AngleAxisf(angles(1), Eigen::Vector3f::UnitY()) *
       Eigen::AngleAxisf(angles(2), Eigen::Vector3f::UnitX());

  Eigen::Matrix3f m;
  m = Eigen::AngleAxisf(rot(2), Eigen::Vector3f::UnitZ()) *
      Eigen::AngleAxisf(rot(1), Eigen::Vector3f::UnitY()) *
      Eigen::AngleAxisf(rot(0), Eigen::Vector3f::UnitX());

  // file << "matrices original - euler - kalman \n";
  for (auto i = 0; i < 3; ++i) {
    stringstream ss, ss1, ss2;
    ss << fixed << setprecision(3);
    ss1 << fixed << setprecision(3);
    ss2 << fixed << setprecision(3);
    for (auto j = 0; j < 3; ++j) {
      ss << tmp_mat(i, j) << " ";
      ss1 << m1(i, j) << " ";
      ss2 << m(i, j) << " ";
    }
    // file << ss.str() << " - " << ss1.str() << " - " << ss2.str() << "\n";
    // file << "\n";
  }

  for (auto i = 0; i < 3; ++i) {
    for (auto j = 0; j < 3; ++j) {
      target_object_.rotation_kalman.at<double>(i, j) = m(i, j);
    }
  }

  // file << "\n\n--------------------------------- \n\n";

  //  try {
  //    Rodrigues(rot, target_object_.rotation_kalman);
  //  } catch (cv::Exception& e) {
  //    cout << "Error estimating ransac rotation: " << e.what() << endl;
  //  }
}

void TrackerMB::predictPoseFlow(std::vector<float>& t, std::vector<float> r) {
  //  ofstream file("/home/alessandro/debug/object_pose.txt",
  //                ofstream::out | ofstream::app);

  Pose tmp_pose = target_object_.flow_pose;
  vector<double> tmp_bt = {t[0],t[1],t[2],r[0],r[1],r[2]};
  tmp_pose.transform(tmp_bt);

  Eigen::Matrix3d rot_view;
  rot_view = Eigen::AngleAxisd(r.at(2), Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(r.at(1), Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(r.at(0), Eigen::Vector3d::UnitX());

  Eigen::Matrix3d tmp_rot = rot_view;

  Eigen::Matrix4d proj_mat(4, 4);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      proj_mat(i, j) = rot_view(i, j);
    }
    proj_mat(i, 3) = t[i];
    proj_mat(3, i) = 0;
  }
  proj_mat(3, 3) = 1;

  Eigen::Matrix4d tmp = proj_mat * target_object_.kalman_pose_;

  for (auto i = 0; i < 3; ++i) {
    for (auto j = 0; j < 3; ++j) rot_view(i, j) = tmp(i, j);
  }

  Eigen::Vector3d rot_vect = rot_view.eulerAngles(2, 1, 0);

  Eigen::Matrix3d rot_view1;
  rot_view1 = Eigen::AngleAxisd(rot_vect(0), Eigen::Vector3d::UnitZ()) *
              Eigen::AngleAxisd(rot_vect(1), Eigen::Vector3d::UnitY()) *
              Eigen::AngleAxisd(rot_vect(2), Eigen::Vector3d::UnitX());

  // file << "rotation " << endl;
  for (auto i = 0; i < 3; ++i) {
    stringstream ss, ss1;
    ss << fixed << setprecision(5);
    ss1 << fixed << setprecision(5);
    for (auto j = 0; j < 3; ++j) {
      ss << rot_view(i, j) << " ";
      ss1 << rot_view1(i, j) << " ";
    }
    // file << ss.str() << "   " << ss1.str() << "\n";
  }
  // file << "\n";

  auto toString = [](cv::Mat& mat) {
    stringstream ss;
    ss << fixed << setprecision(5);
    for (auto i = 0; i < mat.rows; ++i) {
      ss << mat.at<float>(i, 0) << " ";
    }
    return ss.str();
  };

  //  file << "rot vector \n" << rot_vect(0) << " " << rot_vect(1) << " "
  //       << rot_vect(2) << " "
  //       << "\n";
  Mat prediction = kalman_pose_flow_.predict();
  // file << "flow prediction \n" << toString(prediction) << "\n";

  vector<double> beta = tmp_pose.getBeta();

  Mat_<float> measurement(6, 1);
  measurement(0) = beta[0];//tmp(0, 3);
  measurement(1) = beta[1];//tmp(1, 3);
  measurement(2) = beta[2];//tmp(2, 3);
  measurement(3) = beta[3];//rot_vect(2);
  measurement(4) = beta[4];//rot_vect(1);
  measurement(5) = beta[5];//rot_vect(0);
  // file << "flow measurement \n" << toString(measurement) << "\n";
  Mat estimated = kalman_pose_flow_.correct(measurement);
  // file << "flow estimated \n" << toString(estimated) << "\n";

  //  cout << "measuered " << endl;
  //  cout << fixed << setprecision(2) << t.at(0) << " " << t.at(1) << " "
  //       << t.at(2) << endl;
  //  cout << "estimated " << endl;
  //  cout << fixed << setprecision(2) << estimated.at<float>(0) << " "
  //       << estimated.at<float>(1) << " " << estimated.at<float>(2) << endl;


  beta = {estimated.at<float>(0),  estimated.at<float>(1),
                        estimated.at<float>(2),  estimated.at<float>(9),
                        estimated.at<float>(10), estimated.at<float>(11)};

  target_object_.flow_pose.transform(beta);

  rot_view =
      Eigen::AngleAxisd(estimated.at<float>(11), Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(estimated.at<float>(10), Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(estimated.at<float>(9), Eigen::Vector3d::UnitX());

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      target_object_.kalman_pose_(i, j) = rot_view(i, j);
    }
    target_object_.kalman_pose_(i, 3) = estimated.at<float>(i);
  }
}

}  // end namespace pinot
