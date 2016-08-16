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

#include "../include/tracker_model_vx.h"

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#define _USE_MATH_DEFINES
#include <math.h>
#include <limits>
#include <algorithm>
#include <utility>
#include <iomanip>

#include <NVX/nvxcu.h>
#include <NVX/nvx_opencv_interop.hpp>
#include <NVX/nvx_timer.hpp>

#include <hdf5_file.h>

#include "../include/pose_estimation.h"
#include "../../utilities/include/utilities.h"
#include "../include/utilities.h"
#include "../include/nvx_utilities.hpp"
#include "../../cuda/include/utility_kernels.h"
#include "../../cuda/include/utility_kernels_pose.h"

#define VERBOSE_DEBUG

using namespace std;
using namespace cv;

namespace fato {

mutex io_mutex;

void safePrint(string message) {
  unique_lock<mutex> lck(io_mutex);
  cout << message << endl;
}

TrackerVX::TrackerVX(const Params& params,
                     std::unique_ptr<FeatureMatcher> matcher)
    : rendered_depth_(params_.image_height * params_.image_width),
      host_rendered_depth_(params_.image_height * params_.image_width, 0) {
  params_ = params;

  profile_ = Profile();

  feature_detector_ = std::move(matcher);

  if (params_.image_width == 0 || params_.image_height == 0)
    throw runtime_error("image parameters not set");

  loadDescriptors(params_.descriptors_file);

  initSynthTracking(params_.model_file, params_.fx, params_.fy, params_.cx,
                    params_.cy, params_.image_width, params_.image_height);

  initializeContext();

  // set camera matrix
  camera_matrix_ = Mat(3, 3, CV_64FC1);
  camera_matrix_.at<double>(0, 0) = params_.fx;
  camera_matrix_.at<double>(1, 1) = params_.fy;
  camera_matrix_.at<double>(0, 2) = params_.cx;
  camera_matrix_.at<double>(1, 2) = params_.cy;
  camera_matrix_.at<double>(2, 2) = 1;

  image_w_ = params_.image_width;
  image_h_ = params_.image_height;

  // launch workers for paraller detection and tracking
  if (params.parallel) {
    task_completed_ = false;
    image_received_ = false;
    detector_thread_ = thread(&TrackerVX::detectorWorker, this);
  }
}

TrackerVX::~TrackerVX() { release(); }

void TrackerVX::loadDescriptors(const string& h5_file) {
  util::HDF5File out_file(h5_file);
  std::vector<uchar> descriptors;
  std::vector<int> dscs_size, point_size;
  std::vector<float> points;
  // int test_feature_size;
  out_file.readArray<uchar>("descriptors", descriptors, dscs_size);
  out_file.readArray<float>("positions", points, point_size);

  Mat init_descriptors;
  vectorToMat(descriptors, dscs_size, init_descriptors);

  vector<Point3f> pts(point_size[0], cv::Point3f(0, 0, 0));

  for (int i = 0; i < point_size[0]; ++i) {
    pts.at(i) =
        cv::Point3f(points[3 * i], points[3 * i + 1], points[3 * i + 2]);
  }

  target_object_.init(pts, init_descriptors);

  feature_detector_->setTarget(init_descriptors);

  std::cout << "Model loaded with " << target_object_.model_points_.size()
            << " key points and " << init_descriptors.rows << " descriptors "
            << std::endl;
}

void TrackerVX::initSynthTracking(const string& object_model, double fx,
                                  double fy, double cx, double cy,
                                  int img_width, int img_height) {
  rendering_engine_ = unique_ptr<pose::MultipleRigidModelsOgre>(
      new pose::MultipleRigidModelsOgre(img_width, img_height, fx, fy, cx, cy,
                                        0.01, 10.0));

  rendering_engine_->addModel(object_model);

  synth_track_.init(cx, cy, fx, fy, img_width, img_height,
                    rendering_engine_.get());
}

void TrackerVX::initializeContext() {
  gpu_context_ = vxCreateContext();

  vx_image frameGray = vxCreateImage(gpu_context_, params_.image_width,
                                     params_.image_height, VX_DF_IMAGE_U8);

  camera_img_delay_ = vxCreateDelay(gpu_context_, (vx_reference)frameGray, 2);
  renderer_img_delay_ = vxCreateDelay(gpu_context_, (vx_reference)frameGray, 2);

  vx::FeatureTracker::Params params;
  params.array_capacity = params_.array_capacity;
  params.detector_cell_size = params_.detector_cell_size;
  params.fast_thresh = params_.fast_thresh;
  params.fast_type = params_.fast_type;
  params.harris_k = params_.harris_k;
  params.harris_thresh = params_.harris_thresh;
  params.lk_num_iters = params_.lk_num_iters;
  params.lk_win_size = params_.lk_win_size;
  params.pyr_levels = params_.pyr_levels;
  params.use_harris_detector = params_.use_harris_detector;
  params.img_w = params_.image_width;
  params.img_h = params_.image_height;
  params.lk_epsilon = params.lk_epsilon;

  synth_graph_ = unique_ptr<vx::FeatureTrackerSynth>(
      new vx::FeatureTrackerSynth(gpu_context_, params));

  real_graph_ = unique_ptr<vx::FeatureTrackerReal>(
      new vx::FeatureTrackerReal(gpu_context_, params));

  vxReleaseImage(&frameGray);
}

void TrackerVX::detectorWorker() {
  cout << "detector worker started" << endl;
  det_worker_ready_ = true;

  while (true) {
    {
      unique_lock<mutex> lock(detector_mutex_);
      while (!image_received_) {
        detector_condition_.wait(lock);
      }
      det_worker_ready_ = false;
      //safePrint("detector working");
    }

    if (task_completed_) break;

    auto begin = chrono::high_resolution_clock::now();
    detectParallel();
    auto end = chrono::high_resolution_clock::now();
    profile_.match_time =
        chrono::duration_cast<chrono::nanoseconds>(end - begin).count();

    {
      //safePrint("detector done");
      unique_lock<mutex> lock(detector_mutex_);
      det_worker_ready_ = true;
      detector_done_ = true;
      tracker_condition_.notify_one();
    }
  }

  cout << "detector worker stopped" << endl;
}

void TrackerVX::resetTarget() { target_object_.resetPose(); }

void TrackerVX::getOpticalFlow(const Mat& prev, const Mat& next,
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
    float dx = target.active_points.at(i).x - prev_points[i].x;
    float dy = target.active_points.at(i).y - prev_points[i].y;
    float error = dx * dx + dy * dy;

    const int& id = target.active_to_model_.at(i);
    KpStatus& s = target.point_status_.at(id);

    if (prev_status[i] == 1 && error < params_.flow_threshold) {
      // const int& id = ids[i];

      if (s == KpStatus::MATCH) s = KpStatus::TRACK;

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

void TrackerVX::getOpticalFlowVX(vx_image prev, vx_image next, Target& target) {
  if (target_object_.active_points.size() <= 0) return;

  vector<int> valid_id;

  real_graph_->track(target.active_points, next);
  real_graph_->getValidPoints(target, params_.flow_threshold);
}

void TrackerVX::next(Mat& rgb) {
  // TODO: visionworks is slow right now due to a bug, grayscale conversion must
  // be moved to gpu later
  auto begin = chrono::high_resolution_clock::now();
  cvtColor(rgb, next_gray_, CV_BGR2GRAY);
  vx_image vxiSrc;
  vxiSrc = nvx_cv::createVXImageFromCVMat(gpu_context_, rgb);
  NVXIO_SAFE_CALL(
      vxuColorConvert(gpu_context_, vxiSrc,
                      (vx_image)vxGetReferenceFromDelay(camera_img_delay_, 0)));
  auto end = chrono::high_resolution_clock::now();
  profile_.img_load_time =
      chrono::duration_cast<chrono::nanoseconds>(end - begin).count();

  // compute flow
  begin = chrono::high_resolution_clock::now();
  trackSequential();
  end = chrono::high_resolution_clock::now();
  profile_.track_time =
      chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
  // render pose
  begin = chrono::high_resolution_clock::now();
  renderPredictedPose();
  end = chrono::high_resolution_clock::now();
  profile_.render_time =
      chrono::duration_cast<chrono::nanoseconds>(end - begin).count();

  if (target_object_.target_found_) {
    // download to host depth buffer
    begin = chrono::high_resolution_clock::now();
    rendered_depth_.copyTo(host_rendered_depth_);
    end = chrono::high_resolution_clock::now();
    profile_.depth_to_host_time =
        chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
    // update position of tracked points
    begin = chrono::high_resolution_clock::now();
    updatePointsDepthFromZBuffer(target_object_, target_object_.weighted_pose);
    end = chrono::high_resolution_clock::now();
    profile_.depth_update_time =
        chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
  }

  // compute match
  begin = chrono::high_resolution_clock::now();
  detectSequential();
  end = chrono::high_resolution_clock::now();
  profile_.match_time =
      chrono::duration_cast<chrono::nanoseconds>(end - begin).count();

  // age dealy of images in visionworks
  vxAgeDelay(camera_img_delay_);
  vxAgeDelay(renderer_img_delay_);
  next_gray_.copyTo(prev_gray_);
  vxReleaseImage(&vxiSrc);
}

void TrackerVX::parNext(cv::Mat& rgb) {
  {
    unique_lock<mutex> lock(detector_mutex_);
    if (!det_worker_ready_) {
      //cout << "worker not ready, skipping img" << endl;
      return;
    }
  }

  // loading image on the gpu and notify detector
  {
    //safePrint("new image ready");
    unique_lock<mutex> lock(detector_mutex_);
    loadImg(rgb);
    image_received_ = true;
    tracking_done_ = false;
    detector_done_ = false;
    detector_condition_.notify_one();
  }
  // doing the tracking stuff
  trackParallel();

  // notify detector tracking is done
  {
    //safePrint("tracking done");
    unique_lock<mutex> lock(detector_mutex_);
    tracking_done_ = true;
    update_condition_.notify_one();
  }
  // waiting for the detector to be finished
  {
    //safePrint("waiting for detector to be done");
    unique_lock<mutex> lock(detector_mutex_);
    while (!detector_done_) tracker_condition_.wait(lock);
  }

  image_received_ = false;
  vxAgeDelay(camera_img_delay_);
  vxAgeDelay(renderer_img_delay_);
  next_gray_.copyTo(prev_gray_);
}

void TrackerVX::loadImg(Mat& rgb) {


  auto begin = chrono::high_resolution_clock::now();
  cvtColor(rgb, next_gray_, CV_BGR2GRAY);
  vx_image vxiSrc;
  vxiSrc = nvx_cv::createVXImageFromCVMat(gpu_context_, rgb);
  NVXIO_SAFE_CALL(
      vxuColorConvert(gpu_context_, vxiSrc,
                      (vx_image)vxGetReferenceFromDelay(camera_img_delay_, 0)));
  auto end = chrono::high_resolution_clock::now();
  profile_.img_load_time =
      chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
  vxReleaseImage(&vxiSrc);

}

void TrackerVX::release() {
  vxReleaseDelay(&camera_img_delay_);
  vxReleaseDelay(&renderer_img_delay_);
  vxReleaseContext(&gpu_context_);

  if (params_.parallel) {
    {
      unique_lock<mutex> lock(detector_mutex_);
      task_completed_ = true;
      image_received_ = true;
    }

    tracker_condition_.notify_one();
    detector_condition_.notify_one();
    detector_thread_.join();
  }
}

void TrackerVX::trackSequential() {
  /*************************************************************************************/
  /*                       TRACKING */
  /*************************************************************************************/
  if (!real_graph_->isInitialized())
    real_graph_->init((vx_image)vxGetReferenceFromDelay(camera_img_delay_, 0));

  auto begin = chrono::high_resolution_clock::now();
  real_graph_->track(target_object_.active_points,
                     (vx_image)vxGetReferenceFromDelay(camera_img_delay_, 0));
  real_graph_->getValidPoints(target_object_, params_.flow_threshold);
  auto end = chrono::high_resolution_clock::now();
  profile_.cam_flow_time =
      chrono::duration_cast<chrono::nanoseconds>(end - begin).count();

  //  cout << "cpu active " << target_object_.active_points.size() << " gpu "
  //       << tmp_t.active_points.size() << endl;

  if (!target_object_.isConsistent())
    cout << "OF: Error status of object is incosistent!!" << endl;
  /*************************************************************************************/
  /*                             POSE ESTIMATION /
  /*************************************************************************************/
  // finding all the points in the model that are active
  begin = chrono::high_resolution_clock::now();
  vector<Point3f> model_pts;
  for (int i = 0; i < target_object_.active_points.size(); ++i) {
    const int& id = target_object_.active_to_model_.at(i);
    model_pts.push_back(target_object_.model_points_.at(id));
  }
  end = chrono::high_resolution_clock::now();
  profile_.active_transf_time =
      chrono::duration_cast<chrono::nanoseconds>(end - begin).count();

  if (model_pts.size() > 4) {
    if (!target_object_.target_found_) {
      // removing outliers using pnp ransac

      vector<int> inliers;
      begin = chrono::high_resolution_clock::now();
      poseFromPnP(model_pts, inliers);
      end = chrono::high_resolution_clock::now();
      profile_.pnp_time =
          chrono::duration_cast<chrono::nanoseconds>(end - begin).count();

      target_object_.pnp_pose =
          Pose(target_object_.rotation, target_object_.translation);

      // initializing the synth pose with PNP
      target_object_.synth_pose = target_object_.pnp_pose;
      target_object_.weighted_pose = target_object_.pnp_pose;

      target_object_.target_history_.clear();
      target_object_.target_history_.init(target_object_.pnp_pose);

      auto inliers_count = inliers.size();
      removeOutliers(inliers);
      validateInliers(inliers);

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
      }
      if (inliers_count > 4) {
        target_object_.target_found_ = true;
        target_object_.pose_ = projection_matrix * target_object_.pose_;
        target_object_.kalman_pose_ =
            projection_matrix * target_object_.kalman_pose_;
      }

      target_object_.flow_pose =
          Pose(target_object_.rotation, target_object_.translation);

    } else {
      begin = chrono::high_resolution_clock::now();
      pair<int, vector<double>> synth_beta = poseFromSynth();
      end = chrono::high_resolution_clock::now();
      profile_.synth_time_vx =
          chrono::duration_cast<chrono::nanoseconds>(end - begin).count();

      std::pair<int, vector<double>> beta = poseFromFlow();

      target_object_.flow_pose.transform(beta.second);

      vector<double> weigthed_beta(6, 0);
      float total_features = static_cast<float>(beta.first + synth_beta.first);

      if (total_features == 0)
        target_object_.target_found_ = false;
      else {
        float ratio = beta.first / total_features;

        if (synth_beta.first > 0) {
          target_object_.synth_pose.transform(synth_beta.second);
        }

        for (auto i = 0; i < 6; ++i) {
          weigthed_beta[i] =
              ((1 - ratio) * synth_beta.second[i] + ratio * beta.second[i]);
        }

        target_object_.weighted_pose.transform(weigthed_beta);
        target_object_.target_history_.update(target_object_.weighted_pose);

        if (!target_object_.isPoseReliable())
          target_object_.target_found_ = false;
      }
    }
  } else {
    target_object_.rotation = Mat(3, 3, CV_64FC1, 0.0f);
    setIdentity(target_object_.rotation);
    target_object_.translation = Mat(1, 3, CV_64FC1, 0.0f);
    target_object_.rotation_vec = Mat(1, 3, CV_64FC1, 0.0f);
    target_object_.target_found_ = false;
    target_object_.resetPose();
  }
}

void TrackerVX::detectSequential() {
  // if (stop_matcher) return;

  vector<KeyPoint> keypoints;
  Mat descriptors;
  /*************************************************************************************/
  /*                       FEATURE MATCHING */
  /*************************************************************************************/
  vector<vector<DMatch>> matches;
  auto begin = chrono::high_resolution_clock::now();
  feature_detector_->match(next_gray_, keypoints, descriptors, matches);
  auto end = chrono::high_resolution_clock::now();
  profile_.feature_extraction =
      chrono::duration_cast<chrono::nanoseconds>(end - begin).count();

  /*************************************************************************************/
  /*                       ADD FEATURES TO TRACKER */
  /*************************************************************************************/
  updatedDetectedPoints(keypoints, matches);
}

void TrackerVX::detectParallel() {
  // if (stop_matcher) return;

  vector<KeyPoint> keypoints;
  Mat descriptors;
  /*************************************************************************************/
  /*                       FEATURE MATCHING */
  /*************************************************************************************/
  vector<vector<DMatch>> matches;
  auto begin = chrono::high_resolution_clock::now();
  pair<float,float> times =feature_detector_->matchP(next_gray_, keypoints, descriptors, matches);
  //feature_detector_->match(next_gray_, keypoints, descriptors, matches);
  auto end = chrono::high_resolution_clock::now();
  profile_.feature_extraction =
      chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
  profile_.feature_detection = times.first;
  profile_.feature_matching = times.second;
  /*************************************************************************************/
  /*                       SYNCRONIZATION */
  /*************************************************************************************/
  unique_lock<mutex> lock(detector_mutex_);
  while (!tracking_done_) {
    update_condition_.wait(lock);
  }
  /*************************************************************************************/
  /*                       ADD FEATURES TO TRACKER */
  /*************************************************************************************/
  //safePrint("updating points");
  updatedDetectedPoints(keypoints, matches);
}

void TrackerVX::trackParallel() {
  // compute flow
  auto begin = chrono::high_resolution_clock::now();
  trackSequential();
  auto end = chrono::high_resolution_clock::now();
  profile_.track_time =
      chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
  // render pose
  begin = chrono::high_resolution_clock::now();
  renderPredictedPose();
  end = chrono::high_resolution_clock::now();
  profile_.render_time =
      chrono::duration_cast<chrono::nanoseconds>(end - begin).count();

  if (target_object_.target_found_) {
    // download to host depth buffer
    begin = chrono::high_resolution_clock::now();
    rendered_depth_.copyTo(host_rendered_depth_);
    end = chrono::high_resolution_clock::now();
    profile_.depth_to_host_time =
        chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
    // update position of tracked points
    begin = chrono::high_resolution_clock::now();
    updatePointsDepthFromZBuffer(target_object_, target_object_.weighted_pose);
    end = chrono::high_resolution_clock::now();
    profile_.depth_update_time =
        chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
  }
}

void TrackerVX::updatedDetectedPoints(
    const std::vector<KeyPoint>& keypoints,
    const std::vector<std::vector<DMatch>>& matches) {
  //cout << "matches size: " << matches.size() << endl;
  auto begin = chrono::high_resolution_clock::now();
  if (matches.size() < 2) return;
  if (matches.at(0).size() != 2) return;

  vector<cv::Point2f>& active_points = target_object_.active_points;
  vector<KpStatus>& point_status = target_object_.point_status_;
  Mat& target_descriptors = feature_detector_->getTargetDescriptors();

  float max_distance = feature_detector_->maxDistance();
  int valid = 0;

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
            target_descriptors.rows)  // target_object_.model_points_.size())
      continue;

    float confidence = 1 - (matches[i][0].distance / max_distance);

    if (confidence < params_.match_confidence) continue;

    auto ratio = (fst_match.distance / scd_match.distance);

    if (ratio > params_.match_ratio) continue;

    KpStatus& s = point_status.at(model_idx);

    if (s != KpStatus::LOST) continue;

    if (point_status.at(model_idx) == KpStatus::LOST) {
      active_points.push_back(keypoints.at(match_idx).pt);
      target_object_.active_to_model_.push_back(model_idx);
      point_status.at(model_idx) = KpStatus::MATCH;
      valid++;
    }
  }
  auto end = chrono::high_resolution_clock::now();
  profile_.matching_update =
      chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
  //cout << "valid matches " << valid << endl;
}

void TrackerVX::renderPredictedPose() {
  // setting up pose according to ogre requirements
  auto eigen_pose = target_object_.weighted_pose.toEigen();
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
  // mapping opengl rendered image to visionworks vx_image
  vx_image rendered_next =
      (vx_image)vxGetReferenceFromDelay(renderer_img_delay_, 0);
  vx_rectangle_t rect;
  vxGetValidRegionImage(rendered_next, &rect);
  vx_uint8* dst_ptr = NULL;  // should be NULL to work in MAP mode
  vx_imagepatch_addressing_t dst_addr;
  vxAccessImagePatch(rendered_next, &rect, 0, &dst_addr, (void**)&dst_ptr,
                     NVX_WRITE_ONLY_CUDA);
  vision::convertFloatArrayToGrayVX(
      (uchar*)dst_ptr, rendering_engine_->getTexture(), params_.image_width,
      params_.image_height, dst_addr.stride_y, 1.0, 2.0);
  vxCommitImagePatch(rendered_next, &rect, 0, &dst_addr, dst_ptr);
  // saving zbuffer to adjust points depth in the real image
  pose::convertZbufferToZ(rendered_depth_.data(),
                          rendering_engine_->getZBuffer(), params_.image_width,
                          params_.image_height, params_.cx, params_.cy, 0.01,
                          10.0);
  // rendered_depth_.copyTo(host_rendered_depth_);
}

void TrackerVX::projectPointsToModel(const Point2f& model_centroid,
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

void TrackerVX::removeOutliers(vector<int>& inliers) {
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

void TrackerVX::validateInliers(std::vector<int>& inliers) {
  for (auto i = 0; i < target_object_.active_points.size(); ++i) {
    int id = target_object_.active_to_model_.at(i);
    KpStatus& s = target_object_.point_status_.at(id);
    if (s == KpStatus::PNP) s = KpStatus::TRACK;
  }
}

void TrackerVX::poseFromPnP(vector<Point3f>& model_pts, vector<int>& inliers) {
  float ransac_error = 2.0f;
  int num_iters = 50;

  solvePnPRansac(model_pts, target_object_.active_points, camera_matrix_,
                 Mat::zeros(1, 8, CV_64F), target_object_.rotation_vec,
                 target_object_.translation, true, num_iters, ransac_error,
                 model_pts.size(), inliers, CV_ITERATIVE);

  try {
    Rodrigues(target_object_.rotation_vec, target_object_.rotation);
  } catch (cv::Exception& e) {
    cout << "Error estimating ransac rotation: " << e.what() << endl;
  }
}

pair<int, vector<double>> TrackerVX::poseFromFlow() {
  vector<Point2f> prev_points;
  vector<Point2f> curr_points;
  vector<float> depths;

  prev_points.reserve(target_object_.active_points.size());
  curr_points.reserve(target_object_.active_points.size());
  depths.reserve(target_object_.active_points.size());

  for (auto i = 0; i < target_object_.active_points.size(); ++i) {
    int id = target_object_.active_to_model_.at(i);
    if (!is_nan(target_object_.projected_depth_.at(id))) {
      prev_points.push_back(target_object_.prev_points_.at(i));
      curr_points.push_back(target_object_.active_points.at(i));
      depths.push_back(target_object_.projected_depth_.at(id));
    }
  }

  if (prev_points.size() == 0) {
    vector<double> bad_pose{0, 0, 0, 0, 0, 0};
    return pair<bool, vector<double>>(false, bad_pose);
  }

  float nodal_x = camera_matrix_.at<double>(0, 2);
  float nodal_y = camera_matrix_.at<double>(1, 2);
  float focal_x = camera_matrix_.at<double>(0, 0);
  float focal_y = camera_matrix_.at<double>(1, 1);

  vector<double> std_beta(6, 0);

  if (prev_points.size() > 4) {
    vector<float> t, r;
    vector<int> outliers;
    Eigen::VectorXf beta;
    beta = getPoseFromFlowRobust(prev_points, depths, curr_points, nodal_x,
                                 nodal_y, focal_x, focal_y, 10, t, r, outliers);
    target_object_.removeInvalidPoints(outliers);

    for (auto i = 0; i < 6; ++i) {
      std_beta.at(i) = beta(i);
    }

    cout << prev_points.size() << " " << outliers.size() << " "
         << target_object_.active_points.size() << endl;
  }

  return pair<int, vector<double>>(target_object_.active_points.size(),
                                   std_beta);
}

pair<int, vector<double>> TrackerVX::poseFromSynth() {
  synth_graph_->track(
      (vx_image)vxGetReferenceFromDelay(renderer_img_delay_, -1),
      (vx_image)vxGetReferenceFromDelay(camera_img_delay_, 0));

  vector<Point2f> prev_pts, next_pts;

  synth_graph_->getValidPoints(params_.flow_threshold, prev_pts, next_pts);

  vector<float> depth_pts;
  depth_pts.reserve(prev_pts.size());
  for (auto pt : prev_pts) {
    int x = floor(pt.x);
    int y = floor(pt.y);
    float depth = host_rendered_depth_.at(x + y * params_.image_width);

    depth_pts.push_back(depth);
  }

  Eigen::VectorXf beta;
  vector<double> std_beta(6, 0);

  vector<float> translation;
  vector<float> rotation;
  vector<int> outliers;

  if (prev_pts.size() > 4) {
    beta = getPoseFromFlowRobust(prev_pts, depth_pts, next_pts, params_.cx,
                                 params_.cy, params_.fx, params_.fy, 10,
                                 translation, rotation, outliers);
    for (auto i = 0; i < 6; ++i) std_beta[i] = beta(i);
  }

  return pair<int, vector<double>>(prev_pts.size(), std_beta);
}

void TrackerVX::projectPointsDepth(std::vector<Point3f>& points,
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

void TrackerVX::initFilter(KalmanFilter& filter, Eigen::MatrixXd& projection) {
  filter = KalmanFilter(18, 6);

  Mat A = Mat::eye(18, 18, CV_32FC1);

  double dt = 1 / 30.0;

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
  setIdentity(filter.measurementNoiseCov, Scalar::all(1e-4));
  setIdentity(filter.errorCovPost, Scalar::all(1e-1));

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

  filter.statePre.at<float>(9) = beta.at(3);   // angles(2);
  filter.statePre.at<float>(10) = beta.at(4);  // angles(1);
  filter.statePre.at<float>(11) = beta.at(5);  // angles(0);
  filter.statePre.at<float>(12) = 0;
  filter.statePre.at<float>(13) = 0;
  filter.statePre.at<float>(14) = 0;
  filter.statePre.at<float>(15) = 0;
  filter.statePre.at<float>(16) = 0;
  filter.statePre.at<float>(17) = 0;

  Mat_<float> measurement(6, 1);
  measurement(0) = beta.at(0);  // target_object_.translation.at<double>(0);
  measurement(1) = beta.at(1);  // target_object_.translation.at<double>(1);
  measurement(2) = beta.at(2);  // target_object_.translation.at<double>(2);
  measurement(3) = beta.at(3);  //;
  measurement(4) = beta.at(4);  //;
  measurement(5) = beta.at(5);  //
}

void TrackerVX::updatePointsDepth(Target& t, Pose& p) {
  vector<Point3f> model_pts;
  for (int i = 0; i < t.active_points.size(); ++i) {
    const int& id = t.active_to_model_.at(i);
    model_pts.push_back(t.model_points_.at(id));
  }
  vector<float> projected_depth;
  // projectPointsDepth(model_pts, target_object_.pose_, projected_depth);

  Eigen::MatrixXd pose(4, 4);
  // Eigen::Matrix4d tmp_pose = target_object_.flow_pose.getPose();
  Eigen::Matrix4d tmp_pose = p.getPose();

  for (auto i = 0; i < 4; ++i) {
    for (auto j = 0; j < 4; ++j) pose(i, j) = tmp_pose(i, j);
  }

  projectPointsDepth(model_pts, pose, projected_depth);

  for (auto i = 0; i < t.active_points.size(); ++i) {
    const int& id = t.active_to_model_.at(i);
    t.projected_depth_.at(id) = projected_depth.at(i);
  }
}

void TrackerVX::updatePointsDepthFromZBuffer(Target& t, Pose& p) {
  //  std::vector<float> z_buffer;
  //  Mat out;
  //  synth_track_.renderObject(p, out, z_buffer);

  vector<int> invalid_ids;

  for (auto i = 0; i < t.active_points.size(); ++i) {
    const int& id = t.active_to_model_.at(i);
    Point2f& pt = t.active_points.at(i);

    int x = floor(pt.x);
    int y = floor(pt.y);

    if (x < 0 || x > image_w_ || y < 0 || y > image_h_) {
      invalid_ids.push_back(i);
      continue;
    }

    float depth = host_rendered_depth_.at(x + y * image_w_);

    if (is_nan(depth)) {
      invalid_ids.push_back(i);
      continue;
    }

    t.projected_depth_.at(id) = depth;
  }

  t.removeInvalidPoints(invalid_ids);
}

cv::Mat TrackerVX::getRenderedPose() {
  vx_image src = (vx_image)vxGetReferenceFromDelay(renderer_img_delay_, -1);
  vx_rectangle_t rect;
  vxGetValidRegionImage(src, &rect);

  vx_uint32 plane_index = 0;
  // vx_rectangle_t rect = {0u, 0u, 640u, 480u};
  void* ptr = NULL;
  vx_imagepatch_addressing_t addr;

  nvx_cv::VXImageToCVMatMapper map(src, plane_index, &rect, VX_READ_ONLY,
                                   VX_IMPORT_TYPE_HOST);
  return map.getMat();
}

cv::Mat TrackerVX::downloadImage(vx_image image) {
  vx_rectangle_t rect;
  vxGetValidRegionImage(image, &rect);

  vx_uint32 plane_index = 0;
  void* ptr = NULL;
  vx_imagepatch_addressing_t addr;

  nvx_cv::VXImageToCVMatMapper map(image, plane_index, &rect, VX_READ_ONLY,
                                   VX_IMPORT_TYPE_HOST);
  return map.getMat();
}

void TrackerVX::printProfile() { cout << profile_.str() << endl; }

TrackerVX::Params::Params() {
  parallel = false;

  image_width = 0;
  image_height = 0;

  fx = 0;
  fy = 0;
  cx = 0;
  cy = 0;

  descriptors_file = "";
  model_file = "";

  match_confidence = 0.8;
  match_ratio = 0.8;

  flow_threshold = 25.0f;  // distance * distance in pixel

  // Parameters for optical flow node
  pyr_levels = 3;
  lk_num_iters = 5;
  lk_win_size = 21;
  lk_epsilon = 0.01;

  // Common parameters for corner detector node
  array_capacity = 2000;
  detector_cell_size = 7;
  use_harris_detector = false;

  // Parameters for harris_track node
  harris_k = 0.04f;
  harris_thresh = 100.0f;

  // Parameters for fast_track node
  fast_type = 9;
  fast_thresh = 7;
}

string TrackerVX::Profile::str() {
  stringstream ss;

  ss << "Tracker profile: \n";
  ss << "\t img:  " << img_load_time / 1000000.0f << "\n";
  ss << "\t pnp:  " << pnp_time / 1000000.0f << "\n";
  ss << "\t matcher:  " << match_time / 1000000.0f << "\n";
  ss << "\t\t features:  " << feature_extraction / 1000000.0f << "\n";
  ss << "\t\t\t detection:  " << feature_detection / 1000000.0f << "\n";
  ss << "\t\t\t matching:  " << feature_matching / 1000000.0f << "\n";
  ss << "\t\t update:  " << matching_update / 1000000.0f << "\n";
  ss << "\t tracker:  " << track_time / 1000000.0f << "\n";
  ss << "\t\t cam_flow: " << cam_flow_time / 1000000.0f << "\n";
  ss << "\t\t active pts: " << active_transf_time / 1000000.0f << "\n";
  ss << "\t\t synth_vx: " << synth_time_vx / 1000000.0f << "\n";
  ss << "\t\t estimator: " << m_est_time / 1000000.0f << "\n";
  ss << "\t renderer: " << render_time / 1000000.0f << "\n";
  ss << "\t depth2host: " << depth_to_host_time / 1000000.0f << "\n";
  ss << "\t depth update: " << depth_update_time / 1000000.0f << "\n";

  return ss.str();
}

}  // end namespace pinot
