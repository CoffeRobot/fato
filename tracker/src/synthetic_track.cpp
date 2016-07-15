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
#include "../include/synthetic_track.hpp"
#include "../include/utilities.h"
#include "../include/pose_estimation.h"

#include <utility_kernels.h>
#include <utility_kernels_pose.h>
#include <device_1d.h>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

namespace fato {

SyntheticTrack::SyntheticTrack() {}

void SyntheticTrack::init(double nodal_x, double nodal_y, double focal_x,
                          double focal_y, int img_w, int img_h,
                          pose::MultipleRigidModelsOgre* rendering_engine) {
  rendering_engine_ = rendering_engine;

  img_w_ = img_w;
  img_h_ = img_h;
  nodal_x_ = nodal_x;
  nodal_y_ = nodal_y;
  focal_x_ = focal_x;
  focal_y_ = focal_y;

  namedWindow("synthetic_pose");
}

pair<bool,vector<double>> SyntheticTrack::poseFromSynth(Pose prev_pose, cv::Mat& curr_img) {
  Mat rendered_image;
  vector<float> z_buffer;
  renderObject(prev_pose, rendered_image, z_buffer);

  Mat rendered_gray;
  cvtColor(rendered_image, rendered_gray, CV_BGR2GRAY);

  Mat gray_next;
  cvtColor(curr_img, gray_next, CV_BGR2GRAY);

  vector<Point2f> prev_pts, next_pts;

  trackCorners(rendered_gray, gray_next, prev_pts, next_pts);

  vector<float> depth_pts;
  depth_pts.reserve(prev_pts.size());
  for (auto pt : prev_pts) {
    int x = floor(pt.x);
    int y = floor(pt.y);
    float depth = z_buffer.at(x + y * img_w_);

    depth_pts.push_back(depth);
  }

//   cout << "before debug" << endl;
  // debug(rendered_gray, gray_next, prev_pts, next_pts, prev_pose, prev_pose);
//  cout << "after debug" << endl;

  vector<float> translation;
  vector<float> rotation;
  vector<int> outliers;


  Eigen::VectorXf beta;
  vector<double> std_beta(6,0);
  bool valid_pose = false;
  if(prev_pts.size() > 4)
  {
    beta = getPoseFromFlowRobust(prev_pts, depth_pts, next_pts, nodal_x_,
                                    nodal_y_, focal_x_, focal_y_, 10,
                                    translation, rotation, outliers);

    for(auto i = 0; i < 6; ++i)
        std_beta[i] = beta(i);

    valid_pose = true;
  }



  return pair<bool,vector<double>>(valid_pose, std_beta);
}

void SyntheticTrack::renderObject(Pose prev_pose, Mat& rendered_image,
                                  std::vector<float>& z_buffer) {
  auto eigen_pose = prev_pose.toEigen();

  double tra_render[3];
  double rot_render[9];
  Eigen::Map<Eigen::Vector3d> tra_render_eig(tra_render);
  Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> rot_render_eig(
      rot_render);
  tra_render_eig = eigen_pose.second;
  rot_render_eig = eigen_pose.first;

  std::vector<pose::TranslationRotation3D> TR(1);
  TR.at(0).setT(tra_render);
  TR.at(0).setR_mat(rot_render);

  rendering_engine_->render(TR);

  std::vector<uchar4> h_texture(img_w_ * img_h_);
  downloadRenderedImage(h_texture);
  downloadZBuffer(z_buffer);

  cv::Mat img_rgba(img_h_, img_w_, CV_8UC4, h_texture.data());

  cv::cvtColor(img_rgba, rendered_image, CV_RGBA2BGR);

}

void SyntheticTrack::downloadRenderedImage(std::vector<uchar4>& h_texture) {
  util::Device1D<uchar4> d_texture(img_h_ * img_w_);
  vision::convertFloatArrayToGrayRGBA(d_texture.data(),
                                      rendering_engine_->getTexture(), img_w_,
                                      img_h_, 1.0, 2.0);
  h_texture.resize(img_h_ * img_w_);
  d_texture.copyTo(h_texture);
}

void SyntheticTrack::downloadZBuffer(std::vector<float>& buffer) {
  util::Device1D<float> d_z(img_h_ * img_w_);
  pose::convertZbufferToZ(d_z.data(), rendering_engine_->getZBuffer(), img_w_,
                          img_h_, nodal_x_, nodal_y_, 0.01, 10.0);
  buffer.resize(img_h_ * img_w_, 0);
  d_z.copyTo(buffer);
}

void SyntheticTrack::trackCorners(cv::Mat& rendered_image, cv::Mat& next_img,
                                  std::vector<cv::Point2f>& prev_pts,
                                  std::vector<cv::Point2f>& next_pts) {
  vector<KeyPoint> kps;
  FAST(rendered_image, kps, 7, true);

  if (kps.empty()) return;

  vector<Point2f> tmp_next_pts, tmp_prev_pts, backflow_pts;

  for (auto kp : kps) tmp_prev_pts.push_back(kp.pt);

  vector<uchar> next_status, prev_status;
  vector<float> next_errors, prev_errors;

  calcOpticalFlowPyrLK(rendered_image, next_img, tmp_prev_pts, tmp_next_pts,
                       next_status, next_errors);
  calcOpticalFlowPyrLK(next_img, rendered_image, tmp_next_pts, backflow_pts,
                       prev_status, prev_errors);

  for (int i = 0; i < tmp_prev_pts.size(); ++i) {
    float error = getDistance(backflow_pts[i], tmp_prev_pts[i]);

    if (prev_status[i] == 1 && error < 5) {
      // const int& id = ids[i];
      prev_pts.push_back(tmp_prev_pts[i]);
      next_pts.push_back(tmp_next_pts[i]);
    }
  }
}

void SyntheticTrack::debug(Mat& rendered_img, Mat& next_img,
                           std::vector<Point2f>& prev_pts,
                           std::vector<Point2f>& next_pts, Pose& prev_pose,
                           Pose& next_pose) {
  Mat debug_img(rendered_img.rows, rendered_img.cols + next_img.cols, CV_8UC1);

  rendered_img.copyTo(debug_img(Rect(0, 0, img_w_, img_h_)));
  next_img.copyTo(debug_img(Rect(img_w_, 0, img_w_, img_h_)));
  cvtColor(debug_img, debug_img, CV_GRAY2BGR);

  for (auto i = 0; i < prev_pts.size(); ++i) {
    cv::circle(debug_img, prev_pts.at(i), 3, Scalar(0, 255, 0), 1);
    Point2f pt = next_pts.at(i);
    pt.x += img_w_;
    cv::circle(debug_img, pt, 3, Scalar(0, 255, 0), 1);
    line(debug_img, prev_pts.at(i), pt, Scalar(0, 255, 0), 1);
  }

  imshow("synthetic_pose", debug_img);
  waitKey(10);
}
}
