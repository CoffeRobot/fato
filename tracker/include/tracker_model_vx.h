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

#ifndef TRACKER_MODEL__VX_H
#define TRACKER_MODEL__VX_H

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
#include <NVX/nvxcu.h>
#include <NVX/nvx_opencv_interop.hpp>

#include "target.hpp"
#include "matcher.h"
#include "config.h"
#include "feature_matcher.hpp"
#include "synthetic_track.hpp"
#include "flow_graph.hpp"

#include "../../fato_rendering/include/device_1d.h"
#include "../../fato_rendering/include/renderer.h"

namespace fato {

class TrackerVX {
 public:
  struct Params {
    bool parallel;

    int image_width;
    int image_height;

    float fx;
    float fy;
    float cx;
    float cy;

    std::string descriptors_file;
    std::string model_file;

    float match_confidence;
    float match_ratio;

    float flow_threshold;

    // VISIONWORKS PARAMETERS
    // parameters for optical flow node
    vx_uint32 pyr_levels;
    vx_uint32 lk_num_iters;
    vx_uint32 lk_win_size;
    vx_float32 lk_epsilon;

    // common parameters for corner detector node
    vx_uint32 array_capacity;
    vx_uint32 detector_cell_size;
    bool use_harris_detector;

    // parameters for harris_track node
    vx_float32 harris_k;
    vx_float32 harris_thresh;

    // parameters for fast_track node
    vx_uint32 fast_type;
    vx_uint32 fast_thresh;

    // parameters used by the m-estimators
    int iterations_m_real;
    int iterations_m_synth;

    Params();
  };

  struct Profile {
    float img_load_time;
    float pnp_time;
    float match_time;
    float feature_extraction;
    float feature_detection;
    float feature_matching;
    float matching_update;
    float track_time;
    float render_time;
    float synth_time;
    float synth_time_vx;
    float synth_graph;
    float synth_points;
    float synth_depth;
    float synth_estimator;
    float cam_flow_time;
    float active_transf_time;
    float corner_time;
    float m_est_time;
    float depth_to_host_time;
    float depth_update_time;

    Profile()
        : img_load_time(0),
          pnp_time(0),
          match_time(0),
          feature_extraction(0),
          feature_detection(0),
          feature_matching(0),
          matching_update(0),
          render_time(0),
          track_time(0),
          synth_time(0),
          synth_time_vx(0),
          synth_graph(0),
          synth_points(0),
          synth_depth(0),
          synth_estimator(0),
          corner_time(0),
          m_est_time(0),
          depth_to_host_time(0),
          depth_update_time(0),
          cam_flow_time(0),
          active_transf_time(0) {}

    std::string str();
  };

  TrackerVX(const Params& params, std::unique_ptr<FeatureMatcher> matcher);

  ~TrackerVX();

  void learnBackground(const cv::Mat& rgb);

  void resetTarget();

  void next(cv::Mat& rgb);

  void parNext(cv::Mat& rgb);

  void loadImg(cv::Mat& rgb);

  std::vector<cv::Point2f> getActivePoints();

  const Target& getTarget() { return target_object_; }

  cv::Mat getRenderedPose();

  cv::Mat downloadImage(vx_image image);

  void release();

  void printProfile();

  cv::Mat getDepthBuffer();

  bool isLost() { return m_is_object_lost; }
  /****************************************************************************/
  /*                       RENDERING                                          */
  /****************************************************************************/
  std::unique_ptr<pose::MultipleRigidModelsOgre> ogre_renderer_;
  std::unique_ptr<rendering::Renderer> renderer_;
  /****************************************************************************/
  /*                       STATS VARIABLES                                    */
  /****************************************************************************/
  bool stop_matcher;

  /****************************************************************************/
  /*                       KALMAN FILTER                                      */
  /****************************************************************************/
  cv::KalmanFilter kalman_pose_flow_;

 private:
  /**
   * @brief loadDescriptors load the descriptors extraced from the 3d model file
   * @param h5_file
   */
  void loadDescriptors(const std::string& h5_file);

  /**
   * @brief initSynthTracking initializes the rendering engine
   * @param object_model model file save in obj format, need to render the
   * object
   * @param fx
   * @param fy
   * @param cx
   * @param cy
   * @param img_width
   * @param img_height
   */
  void initSynthTracking(const std::string& object_model, double fx, double fy,
                         double cx, double cy, int img_width, int img_height);

  /**
   * @brief initializeContext initializes visionworks context and variables
   */
  void initializeContext();


  void detectorWorker();

  void getOpticalFlow(const cv::Mat& prev, const cv::Mat& next, Target& target);

  void getOpticalFlowVX(vx_image prev, vx_image next, Target& target);

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

  void trackSequential();

  void detectSequential();

  void detectParallel();

  void trackParallel();

  void updatedDetectedPoints(
      const std::vector<cv::KeyPoint>& keypoints,
      const std::vector<std::vector<cv::DMatch>>& matches);

  void renderPredictedPose();

  void poseFromPnP(std::vector<cv::Point3f>& model_pts,
                   std::vector<int>& inliers);

  std::pair<int, std::vector<double>> poseFromFlow();

  std::pair<int, std::vector<double>> poseFromSynth();

  void projectPointsDepth(std::vector<cv::Point3f>& points,
                          Eigen::MatrixXd &projection,
                          std::vector<float>& projected_depth);

  void initFilter(cv::KalmanFilter& filter, Eigen::MatrixXd& projection);

  void updatePointsDepth(Target& t, Pose& p);

  void updatePointsDepthFromZBuffer(Target& t, Pose& p);

  int image_w_, image_h_;

  cv::Mat prev_gray_, next_gray_;
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
  std::condition_variable tracker_condition_, detector_condition_, update_condition_;
  std::mutex detector_mutex_;

  bool image_received_, tracking_done_, detector_done_, det_worker_ready_,
       task_completed_;

  std::thread detector_thread_;
  std::thread tracker_thread_;
  /****************************************************************************/
  /*                       DETECTOR VARIABLES                                 */
  /****************************************************************************/
  std::unique_ptr<FeatureMatcher> feature_detector_;

  /****************************************************************************/
  /*                       PROFILINGVARIABLES                                 */
  /****************************************************************************/
  Profile profile_;
  /****************************************************************************/
  /*                           PARAMS                                        */
  /****************************************************************************/
  float matcher_confidence_;
  float matcher_ratio_;
  int pnp_iterations_;
  /****************************************************************************/
  /*                       TARGETS TO TRACK                                   */
  /****************************************************************************/
  Target target_object_;
  /****************************************************************************/
  /*                       GPU VARIABLES                                      */
  /****************************************************************************/
  vx_delay camera_img_delay_, renderer_img_delay_;
  vx_context gpu_context_;
  std::unique_ptr<vx::FeatureTrackerSynth> synth_graph_;
  std::unique_ptr<vx::FeatureTrackerReal> real_graph_;

  util::Device1D<float> rendered_depth_;
  std::vector<float> host_rendered_depth_;
  // ERORR: why doesn't work if I move this!!!!
  Params params_;

  //SyntheticTrack synth_track_;


  std::string file_name_pose;
};

}  // end namespace

#endif  // TRACKER_H
