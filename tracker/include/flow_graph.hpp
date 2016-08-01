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
#ifndef FLOW_GRAPH_HPP
#define FLOW_GRAPH_HPP

#include <VX/vx.h>
#include <NVX/nvx.h>
#include <vector>
#include <opencv2/core.hpp>

namespace fato {

namespace vx {

class FeatureTracker {
 public:
  struct Params {
    // parameters for optical flow node
    vx_uint32 pyr_levels;
    vx_uint32 lk_num_iters;
    vx_uint32 lk_win_size;

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

    Params();
  };

  virtual ~FeatureTracker() {}

  virtual void printPerfs() const = 0;

  void uploadPoints(const std::vector<nvx_keypointf_t>& prev_pts);

  void downloadPoints(std::vector<nvx_keypointf_t>& prev_pts,
                      std::vector<nvx_keypointf_t>& next_pts,
                      std::vector<nvx_keypointf_t>& back_pts);

  void getValidPoints(float distance_thresh,
                      std::vector<nvx_keypointf_t>& prev_pts,
                      std::vector<nvx_keypointf_t>& next_pts);

 protected:

  vx_context context_;
  // Format for current frames
  vx_df_image format_;
  vx_uint32 width_;
  vx_uint32 height_;
  // Tracked points
  vx_array kp_next_list_;
  vx_array kp_prev_list_;
  vx_array kp_back_list_;
  // Main graph
  vx_graph main_graph_;
  // timing
  float dl_time_, valid_time_;

  bool is_initialized;
};

/**
 * @brief The FeatureTrackerReal class, used to track features from camera
 * images,the structure of the visionworks graph differs from the real graph
 */
class FeatureTrackerReal : public FeatureTracker {
 public:
  FeatureTrackerReal(vx_context context, const Params& params);
  ~FeatureTrackerReal();

  void init(vx_image firstFrame, std::vector<nvx_keypointf_t>& points);
  void track(vx_image newFrame, vx_image mask);

  void printPerfs() const;

 private:
  void createDataObjects();

  void processFirstFrame(vx_image frame);
  void createMainGraph(vx_image frame);

  void release();

  Params params_;
  // Pyramids for two successive frames
  vx_delay pyr_delay_;

  // Node from main graph (used to print performance results)
  vx_node pyr_node_;
  vx_node opt_flow_node_forward_;
  vx_node opt_flow_node_backward_;


};

/**
 * @brief The FeatureTrackerSynth class, used to track features from rendered
 * images, the structure of the visionworks graph differs from the real graph
 */
class FeatureTrackerSynth : public FeatureTracker {
 public:
  FeatureTrackerSynth(vx_context context, const Params& params);
  ~FeatureTrackerSynth();

  void init(vx_image sample_image);

  void track(vx_image rendered, vx_image curr_frame);

  void printPerfs() const;

 private:
  void createDataObjects();

  // void processFirstFrame(vx_image frame);
  void createMainGraph(vx_image frame);

  void release();

  Params params_;

  // Pyramids for two images
  vx_pyramid pyr_rendered, pyr_image;

  // Node from main graph (used to print performance results)
  vx_node pyr_node_synth_, pyr_node_img_;
  vx_node opt_flow_node_forward_;
  vx_node opt_flow_node_backward_;
  vx_node feature_track_node_;
};
}
}

#endif  // FLOW_GRAPH_HPP
