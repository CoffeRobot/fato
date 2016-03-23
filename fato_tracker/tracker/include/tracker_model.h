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

#ifndef TRACKER_V2_H
#define TRACKER_V2_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <queue>
#include <memory>
#include <future>
#include <queue>
#include <condition_variable>
#include <fstream>
#include <set>
#include <memory>

#include "profiler.h"
#include "matcher.h"
#include "config.h"
#include "feature_matcher.hpp"

namespace fato {

enum struct Status {
  INIT = 0,
  MATCH,
  NOMATCH,
  TRACK,
  BOTH,
  BACKGROUND,
  LOST,
  NOCLUSTER,
  LEARN
};


class TrackerMB {
 public:
  TrackerMB();

  TrackerMB(const Config& params, const cv::Mat& camera_matrix, std::unique_ptr<FeatureMatcher> matcher);

  TrackerMB(Config &params, int descriptor_type, std::unique_ptr<FeatureMatcher> matcher);

  ~TrackerMB();

  void addModel(const cv::Mat& descriptors, const std::vector<cv::Point3f>& points);

  void init(const cv::Mat &rgb, const cv::Point2d &fst, const cv::Point2d &scd);

  void init(const cv::Mat& rgb, const cv::Mat& mask);

  void setFeatureExtractionParameters(int num_features, float scale_factor,
                                      int num_levels, int edge_threshold,
                                      int first_level, int patch_size);

  void setMatcerParameters(float confidence, float second_ratio);

  void computeNext(const cv::Mat& rgb);

  const std::vector<cv::Point2f>* getPoints() { return &m_updatedPoints; }

  const std::vector<Status>* getPointsStatus() { return &m_pointsStatus; }

  const std::vector<int>* getPointsIds() { return &m_upd_to_init_ids; }

  const std::vector<cv::Point2f>* getInitPoints() { return &m_points; }

  cv::Point2f getCentroid() { return m_updatedCentroid; }

  cv::Point2f getInitCentroid() { return m_initCentroid; }

  std::vector<cv::Point2f> getBoundingBox() { return m_boundingBoxUpdated; }

  bool getEstimatedPosition(std::vector<cv::Point2f>& bounding_box);

  const std::vector<cv::Point2f>* getVotes() { return &m_votes; }

  bool isNewPose() { return m_learn_new_pose; }

  void taskFinished();

  float getTrackerTime() {
    return m_trackerTime / static_cast<float>(m_trackerFrameCount);
  }

  float getDetectorTime() {
    return m_detectorTime / static_cast<float>(m_detectorFrameCount);
  }

  float getAngle() { return m_angle; }

  float getScale() { return m_scale; }

  bool isLost() { return m_is_object_lost; }

  /****************************************************************************/
  /*                       STATS VARIABLES                                    */
  /****************************************************************************/
  std::atomic_int m_flow_counter;
  std::atomic_int m_match_counter;
  int m_original_model_size;
  std::vector<Status> m_points_status_debug;

 private:
  int runTracker();

  int runDetector();

  cv::Point2f initCentroid(const std::vector<cv::Point2f>& points);

  void initRelativeDistance(const std::vector<cv::Point2f>& points,
                            const cv::Point2f& centroid,
                            std::vector<cv::Point2f>& relDistances);

  void initBoundingBox(const cv::Mat& mask, const cv::Point2f& centroid,
                       std::vector<cv::Point2f>& initBox,
                       std::vector<cv::Point2f>& relativeBox,
                       std::vector<cv::Point2f>& updBox);

  void getOpticalFlow(const cv::Mat& prev,
                      const cv::Mat& next,
                      std::vector<cv::Point2f>& points, std::vector<int>& ids,
                      std::vector<Status>& status);

  float getMedianRotation(const std::vector<cv::Point2f>& initPoints,
                          const std::vector<cv::Point2f>& updPoints,
                          const std::vector<int>& ids);

  float getMedianScale(const std::vector<cv::Point2f>& initPoints,
                       const std::vector<cv::Point2f>& updPoints,
                       const std::vector<int>& ids);

  void voteForCentroid(const std::vector<cv::Point2f>& relativeDistances,
                       const std::vector<cv::Point2f>& updPoints,
                       const float& angle, const float& scale,
                       std::vector<cv::Point2f>& votes);

  void updatePointsStatus(const std::vector<bool>& isClustered,
                          std::vector<cv::Point2f>& points,
                          std::vector<cv::Point2f>& votes,
                          std::vector<cv::Point2f>& relDistances,
                          std::vector<int>& ids,
                          std::vector<Status>& pointsStatus);

  void labelNotClusteredPts(const std::vector<bool>& isClustered,
                            std::vector<cv::Point2f>& points,
                            std::vector<cv::Point2f>& votes,
                            std::vector<cv::Point2f>& relDistances,
                            std::vector<int>& ids,
                            std::vector<Status>& pointsStatus);

  void discardNotClustered(std::vector<cv::Point2f>& upd_points,
                           std::vector<cv::Point2f>& init_pts,
                           cv::Point2f& upd_centroid,
                           cv::Point2f& init_centroid, std::vector<int>& ids,
                           std::vector<Status>& pointsStatus);

  void removeLostPoints(const std::vector<bool>& isClustered,
                        std::vector<cv::Point2f>& points,
                        std::vector<cv::Point2f>& votes,
                        std::vector<cv::Point2f>& relDistances,
                        std::vector<int>& ids,
                        std::vector<Status>& pointsStatus);

  void updateCentroid(const float& angle, const float& scale,
                      const std::vector<cv::Point2f>& votes,
                      const std::vector<bool>& isClustered,
                      cv::Point2f& updCentroid);

  void updateBoundingBox(const float& angle, const float& scale,
                         const std::vector<cv::Point2f>& boundingBoxRelative,
                         const cv::Point2f& updCentroid,
                         std::vector<cv::Point2f>& updBox);

  void clusterVotes(std::vector<cv::Point2f>& centroidVotes,
                    std::vector<bool>& isClustered);

  bool evaluatePose(const float& angle, const float& scale);

  void projectPointsToModel(const cv::Point2f& model_centroid,
                            const cv::Point2f& upd_centroid, const float angle,
                            const float scale,
                            const std::vector<cv::Point2f>& pts,
                            std::vector<cv::Point2f>& proj_pts);

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

  bool isPointValid(const int& id);

  void trackNext(cv::Mat next);

  void detectNext(cv::Mat next);

  int m_height, m_width;

  std::future<int> m_trackerStatus, m_detectorStatus;

  cv::Mat m_nextRgb, m_init_rgb_img, prev_gray_;
  /****************************************************************************/
  /*                       ESTIMATION VARIABLES                               */
  /****************************************************************************/
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
  std::vector<Status> m_pointsStatus;
  std::unique_ptr<FeatureMatcher> feature_detector_;

  /****************************************************************************/
  /*                       TRACKER VARIABLES                                  */
  /****************************************************************************/
  std::vector<cv::Point2f> m_updatedPoints;
  std::vector<cv::Point2f> m_relativeDistances;
  std::vector<cv::Point2f> m_votes;
  std::vector<int> m_upd_to_init_ids;
  std::vector<int> m_init_to_upd_ids;
//  cv::gpu::PyrLKOpticalFlow m_dPyrLK;
//  cv::gpu::GpuMat dm_prev, dm_prevGray;
  /****************************************************************************/
  /*                       TRACKED MODEL                                      */
  /****************************************************************************/
  cv::Point2f m_initCentroid;
  cv::Point2f m_updatedCentroid;
  std::vector<cv::Point2f> m_boundingBoxRelative;
  std::vector<cv::Point2f> m_boundingBox;
  std::vector<cv::Point2f> m_boundingBoxUpdated;
  /****************************************************************************/
  /*                       LEARN MODEL                                        */
  /****************************************************************************/
  std::set<std::pair<float, float>> m_learned_poses;
  bool m_learn_new_pose;
  float m_scale_old;
  float m_angle_old;
  cv::Point2f m_centroid_old;
  std::deque<float> m_scale_history;
  std::deque<float> m_angle_history;
  std::deque<cv::Point2f> m_center_history;
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
  cv::Mat camera_matrix_;
  int pnp_iterations_;
  /****************************************************************************/
  /*                       MATCHER PARAMS                                     */
  /****************************************************************************/
  float matcher_confidence_;
  float matcher_ratio_;

};

} // end namespace

#endif  // TRACKER_H
