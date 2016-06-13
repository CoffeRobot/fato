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

//#include "../include/target.hpp"
#include <iostream>
#include <fstream>
#include <opencv2/calib3d/calib3d.hpp>
#include <target.hpp>
#include <stdexcept>
#include <Eigen/Dense>

using namespace cv;
using namespace std;

namespace fato {

void Target::init(std::vector<cv::Point3f> &points, cv::Mat &descriptors) {
  model_points_ = points;
  descriptors_ = descriptors.clone();

  centroid_ = Point3f(0, 0, 0);

  rel_distances_.reserve(model_points_.size());
  point_status_.resize(model_points_.size(), KpStatus::LOST);
  projected_depth_.resize(model_points_.size(),
                          std::numeric_limits<float>::quiet_NaN());

  for (auto pt : model_points_) {
    rel_distances_.push_back(centroid_ - pt);
  }

  resetPose();
}

void Target::resetPose() {
  //cout << "Object status resetting " << endl;
  int reserved_memory = model_points_.size() / 10;

  active_to_model_.clear();
  active_points.clear();

  active_to_model_.reserve(reserved_memory);
  active_points.reserve(reserved_memory);

  rotation = Mat(3, 3, CV_64FC1, 0.0f);
  rotation_custom = Mat(3, 3, CV_64FC1, 0.0f);
  rotation_vec = Mat(1, 3, CV_64FC1, 0.0f);
  translation = Mat(1, 3, CV_64FC1, 0.0f);
  translation_custom = Mat(1, 3, CV_64FC1, 0.0f);

  target_found_ = false;

  pose_ = Eigen::MatrixXd(4, 4);

  Eigen::Matrix3d rot_view;
  rot_view = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX());

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      pose_(i, j) = rot_view(i, j);
    }
  }
  pose_(2, 3) = 0;
  pose_(3, 3) = 1;

  for (auto &p : point_status_) p = KpStatus::LOST;

  for (auto &d : projected_depth_) d = numeric_limits<float>::quiet_NaN();

  //cout << "Object status reset " << endl;
}

void Target::removeInvalidPoints(const std::vector<int> &ids) {
  int last_valid = active_points.size() - 1;

  for (auto id : ids) {
    int mid = active_to_model_.at(id);
    point_status_.at(mid) = KpStatus::LOST;
    projected_depth_.at(mid) = std::numeric_limits<float>::quiet_NaN();
  }

  vector<int> elem_ids(active_points.size(), 0);
  for (int i = 0; i < elem_ids.size(); ++i) elem_ids[i] = i;

  for (auto id : ids) {
    auto &el_id = elem_ids.at(id);

    std::swap(active_points.at(el_id), active_points.at(last_valid));
    std::swap(prev_points_.at(el_id), prev_points_.at(last_valid));
    std::swap(active_to_model_.at(el_id), active_to_model_.at(last_valid));
    std::swap(elem_ids.at(el_id), elem_ids.at(last_valid));
    last_valid--;
  }

  int resized = last_valid + 1;
  active_points.resize(resized);
  active_to_model_.resize(resized);
  prev_points_.resize(resized);

  isConsistent();
}

bool Target::isConsistent() {
  bool consistent_check = true;

  for (auto i : active_to_model_) {
    if (point_status_.at(i) != KpStatus::TRACK &&
        point_status_.at(i) != KpStatus::MATCH &&
        point_status_.at(i) != KpStatus::PNP)
      throw std::runtime_error("points status not consistent in target");
  }

  return consistent_check;
}

void Target::projectVectors(Mat &camera, Mat &out) {
  std::vector<Point3f *> model_pts;
  std::vector<Point3f *> rel_pts;

  for (auto i = 0; i < active_to_model_.size(); ++i) {
    model_pts.push_back(&model_points_.at(active_to_model_.at(i)));
    rel_pts.push_back(&rel_distances_.at(active_to_model_.at(i)));
  }

  std::vector<Point2f> model_2d, rel_2d;
  projectPoints(model_pts, rotation, translation, camera, Mat(), model_2d);
  projectPoints(rel_pts, rotation, translation, camera, Mat(), rel_2d);

  for (auto i = 0; i < model_2d.size(); ++i) {
    line(out, model_2d.at(i), rel_2d.at(i), Scalar(0, 255, 0));
  }
}
}
