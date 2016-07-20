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

#ifndef TRACKER_MODEL_H
#define TRACKER_MODEL_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

#include <vector>
#include <queue>
#include <memory>
#include <future>
#include <queue>
#include <condition_variable>
#include <fstream>
#include <set>
#include <memory>
#include <string>

#include "target.hpp"
#include "matcher.h"
#include "config.h"
#include "feature_matcher.hpp"
#include "tracker_2d_v2.h"
#include "synthetic_track.hpp"

namespace fato {

class TrackerMB {
 public:
  TrackerMB();

  TrackerMB(const Config& params, const cv::Mat& camera_matrix,
            std::unique_ptr<FeatureMatcher> matcher);

  TrackerMB(Config& params, int descriptor_type,
            std::unique_ptr<FeatureMatcher> matcher);

  ~TrackerMB();

  void addModel(const cv::Mat& descriptors,
                const std::vector<cv::Point3f>& points);

  void addModel(const std::string& h5_file);

  void setParameters(cv::Mat& camera_matrix, int image_w, int image_h);
  // TODO: merge the initialization to force order of init things, synth depends
  // from camera
  void initSynthTracking(const std::string& object_model, double fx, double fy,
                         double cx, double cy, int img_width, int img_height);

  void learnBackground(const cv::Mat& rgb);

  void resetTarget();

  void setFeatureExtractionParameters(int num_features, float scale_factor,
                                      int num_levels, int edge_threshold,
                                      int first_level, int patch_size);

  void setMatcerParameters(float confidence, float second_ratio);

  void computeNext(const cv::Mat& rgb);

  void computeNextSequential(cv::Mat& rgb);

  std::vector<cv::Point2f> getActivePoints();

  const std::vector<cv::Point2f>* getInitPoints() { return &m_points; }

  const Target& getTarget() { return target_object_; }

  void getRenderedPose(const Pose& p, cv::Mat& out);


  void taskFinished();

  float getTrackerTime() {
    return m_trackerTime / static_cast<float>(m_trackerFrameCount);
  }

  float getDetectorTime() {
    return m_detectorTime / static_cast<float>(m_detectorFrameCount);
  }


  bool isLost() { return m_is_object_lost; }

  /****************************************************************************/
  /*                       STATS VARIABLES                                    */
  /****************************************************************************/
  std::atomic_int m_flow_counter;
  std::atomic_int m_match_counter;
  int m_original_model_size;
  std::vector<KpStatus> m_points_status_debug;

  bool stop_matcher;

  /****************************************************************************/
  /*                       KALMAN POSE                                        */
  /****************************************************************************/
  cv::KalmanFilter kalman_pose_pnp_;
  cv::KalmanFilter kalman_pose_flow_;

 private:
  int runTracker();

  int runDetector();

  void getOpticalFlow(const cv::Mat& prev, const cv::Mat& next, Target& target);

  void projectPointsToModel(const cv::Point2f& model_centroid,
                            const cv::Point2f& upd_centroid, const float angle,
                            const float scale,
                            const std::vector<cv::Point2f>& pts,
                            std::vector<cv::Point2f>& proj_pts);

  void removeOutliers(std::vector<int>& inliers);

  void validateInliers(std::vector<int>& inliers);

  void getTrackedPoints(std::vector<cv::Point2f>& model_pts,
                        std::vector<cv::Point2f>& current_pts,
                        const std::vector<int>& ids,
                        std::vector<cv::Point2f*>& model_valid_pts,
                        std::vector<cv::Point2f*>& current_valid_pts);

  void getTrackedPoints(std::vector<cv::Point2f>& model_pts,
                        std::vector<cv::Point2f>& current_pts,
                        const std::vector<int>& ids,
                        std::vector<cv::Point3f>& model_valid_pts,
                        std::vector<cv::Point2f>& current_valid_pts);

  void trackSequential(cv::Mat& next);

  void detectSequential(cv::Mat& next);

  void poseFromPnP(std::vector<cv::Point3f>& model_pts,
                   std::vector<int>& inliers);

  std::pair<int, std::vector<double> > poseFromFlow();

  void projectPointsDepth(std::vector<cv::Point3f>& points,
                          Eigen::MatrixXd& projection,
                          std::vector<float>& projected_depth);

  void initFilter(cv::KalmanFilter& filter, Eigen::MatrixXd& projection);

  void predictPose();

  void predictPoseFlow(std::vector<float>& t, std::vector<float> r);

  void updatePointsDepth(Target& t, Pose& p);

  void updatePointsDepthFromZBuffer(Target& t, Pose& p);

  int image_w_, image_h_;

  std::future<int> m_trackerStatus, m_detectorStatus;

  cv::Mat m_nextRgb, m_init_rgb_img, prev_gray_;
  /****************************************************************************/
  /*                       ESTIMATION VARIABLES                               */
  /****************************************************************************/
  cv::Mat camera_matrix_;
  float m_angle, m_scale;
  bool m_is_object_lost;
  int ransac_iterations_;
  float ransac_distance_;
  /****************************************************************************/
  /*                       CONCURRENCY VARIABLES                              */
  /****************************************************************************/
  std::condition_variable m_trackerCondition, m_detectorCondition;
  std::mutex m_trackerMutex, m_detectorMutex, m_mutex;
  std::atomic_int m_completed;
  std::atomic_bool m_isRunning;
  std::atomic_bool m_trackerDone;
  std::atomic_bool m_matcherDone;
  /****************************************************************************/
  /*                       DETECTOR VARIABLES                                 */
  /****************************************************************************/
  cv::Mat m_initDescriptors;
  std::vector<cv::KeyPoint> m_initKeypoints;
  std::vector<cv::Point2f> m_points;
  std::unique_ptr<FeatureMatcher> feature_detector_;

  /****************************************************************************/
  /*                       TRACKER VARIABLES                                  */
  /****************************************************************************/
  std::vector<cv::Point2f> m_relativeDistances;


  /****************************************************************************/
  /*                       PROFILINGVARIABLES                                 */
  /****************************************************************************/
  float m_trackerTime;
  float m_detectorTime;
  float m_trackerFrameCount;
  float m_detectorFrameCount;
  /****************************************************************************/
  /*                       PNP RANSAC REQUIREMENTS                            */
  /****************************************************************************/
  int pnp_iterations_;
  /****************************************************************************/
  /*                       MATCHER PARAMS                                     */
  /****************************************************************************/
  float matcher_confidence_;
  float matcher_ratio_;
  /****************************************************************************/
  /*                       TARGETS TO TRACK                                   */
  /****************************************************************************/
  Target target_object_;

  SyntheticTrack synth_track_;
  std::unique_ptr<pose::MultipleRigidModelsOgre> rendering_engine_;

  std::string file_name_pose;
};

}  // end namespace

#endif  // TRACKER_H
