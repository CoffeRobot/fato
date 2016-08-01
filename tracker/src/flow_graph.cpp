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
#include "../include/flow_graph.hpp"

#include <climits>
#include <cfloat>
#include <iostream>
#include <iomanip>

#include <VX/vxu.h>
#include <NVX/nvx.h>
#include <vector>
#include <opencv2/core.hpp>
#include <iostream>
#include <chrono>

#include "../include/nvx_utilities.hpp"

//
// The feature_tracker.cpp contains the implementation of the  virtual void
// functions: track() and init()
//

using namespace std;

namespace fato {
namespace vx {

void FeatureTracker::getValidPoints(float distance_thresh,
                                    std::vector<nvx_keypointf_t>& prev_pts,
                                    std::vector<nvx_keypointf_t>& next_pts) {
  auto begin = chrono::high_resolution_clock::now();

  vx_size prev_size, next_size, back_size;
  vx_size prev_stride, next_stride, back_stride = 0;
  void* prev_data = NULL;
  void* next_data = NULL;
  void* back_data = NULL;

  vxQueryArray(kp_prev_list_, VX_ARRAY_ATTRIBUTE_NUMITEMS, &prev_size,
               sizeof(prev_size));
  vxQueryArray(kp_prev_list_, VX_ARRAY_ATTRIBUTE_NUMITEMS, &next_size,
               sizeof(next_size));
  vxQueryArray(kp_prev_list_, VX_ARRAY_ATTRIBUTE_NUMITEMS, &back_size,
               sizeof(back_size));

  cout << "prev " << prev_size << " next " << next_size << " back "
       << back_size << endl;

  prev_pts.reserve(prev_size);
  next_pts.reserve(prev_size);

  vxAccessArrayRange(kp_prev_list_, 0, prev_size, &prev_stride, &prev_data,
                     VX_READ_ONLY);
  vxAccessArrayRange(kp_next_list_, 0, next_size, &next_stride, &next_data,
                     VX_READ_ONLY);
  vxAccessArrayRange(kp_back_list_, 0, back_size, &back_stride, &back_data,
                     VX_READ_ONLY);
  for (auto i = 0; i < next_size; ++i) {
    nvx_keypointf_t prev_kp =
        vxArrayItem(nvx_keypointf_t, prev_data, i, prev_stride);
    nvx_keypointf_t next_kp =
        vxArrayItem(nvx_keypointf_t, next_data, i, next_stride);
    nvx_keypointf_t back_kp =
        vxArrayItem(nvx_keypointf_t, back_data, i, back_stride);

    if (next_kp.tracking_status == 0) {
      continue;
    }

    float dx = prev_kp.x - back_kp.x;
    float dy = prev_kp.y - back_kp.y;
    if(i < 20)
        cout << "dx "<< dx << " dy " << dy << " " << next_kp.tracking_status << endl;

    if (sqrt(dx * dx + dy * dy) > distance_thresh) continue;

    prev_pts.push_back(prev_kp);
    next_pts.push_back(next_kp);
  }
  vxCommitArrayRange(kp_prev_list_, 0, prev_size, prev_data);
  vxCommitArrayRange(kp_next_list_, 0, next_size, next_data);
  vxCommitArrayRange(kp_back_list_, 0, back_size, back_data);

  cout << "invalid points " << next_size - next_pts.size() << endl;

  auto end = chrono::high_resolution_clock::now();
  valid_time_ = chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
}

void FeatureTracker::downloadPoints(std::vector<nvx_keypointf_t>& prev_pts,
                                    std::vector<nvx_keypointf_t>& next_pts,
                                    std::vector<nvx_keypointf_t>& pred_pts) {
  auto begin = chrono::high_resolution_clock::now();

  vx_size prev_size, next_size, back_size;
  vx_size stride = 0;
  void* data = NULL;

  vxQueryArray(kp_prev_list_, VX_ARRAY_ATTRIBUTE_NUMITEMS, &prev_size,
               sizeof(prev_size));
  vxQueryArray(kp_next_list_, VX_ARRAY_ATTRIBUTE_NUMITEMS, &next_size,
               sizeof(prev_size));
  vxQueryArray(kp_back_list_, VX_ARRAY_ATTRIBUTE_NUMITEMS, &back_size,
               sizeof(back_size));

  prev_pts.resize(prev_size, nvx_keypointf_t());
  next_pts.resize(next_size, nvx_keypointf_t());
  pred_pts.resize(next_size, nvx_keypointf_t());

  vxAccessArrayRange(kp_next_list_, 0, next_size, &stride, &data, VX_READ_ONLY);
  for (auto i = 0; i < next_size; ++i) {
    next_pts[i] = vxArrayItem(nvx_keypointf_t, data, i, stride);
  }
  vxCommitArrayRange(kp_next_list_, 0, next_size, data);

  vxAccessArrayRange(kp_prev_list_, 0, prev_size, &stride, &data, VX_READ_ONLY);
  for (auto i = 0; i < prev_size; ++i) {
    prev_pts[i] = vxArrayItem(nvx_keypointf_t, data, i, stride);
  }
  vxCommitArrayRange(kp_prev_list_, 0, prev_size, data);

  vxAccessArrayRange(kp_back_list_, 0, back_size, &stride, &data, VX_READ_ONLY);
  for (auto i = 0; i < back_size; ++i) {
    pred_pts[i] = vxArrayItem(nvx_keypointf_t, data, i, stride);
  }
  vxCommitArrayRange(kp_back_list_, 0, back_size, data);

  auto end = chrono::high_resolution_clock::now();
  dl_time_ = chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
}

void FeatureTracker::uploadPoints(
    const std::vector<nvx_keypointf_t>& prev_pts) {
  vx_size size, max_capacity;

  vxQueryArray(kp_prev_list_, VX_ARRAY_ATTRIBUTE_NUMITEMS, &size, sizeof(size));
  vxQueryArray(kp_prev_list_, VX_ARRAY_ATTRIBUTE_CAPACITY, &max_capacity,
               sizeof(size));

  vx_size new_size = prev_pts.size();

  vx_size stride = 0;
  void* data = NULL;
  new_size = min(new_size, vx_size(max_capacity));
  int elems_to_update;

  if (new_size > size) {
    NVXIO_CHECK_REFERENCE(kp_prev_list_);
    vx_size to_add = new_size - size;
    NVXIO_SAFE_CALL(vxAddArrayItems(kp_prev_list_, to_add,
                                    prev_pts.data() + size,
                                    sizeof(nvx_keypointf_t)));
    elems_to_update = size;
  } else {
    vxTruncateArray(kp_prev_list_, new_size);
    elems_to_update = new_size;
  }

  vxQueryArray(kp_prev_list_, VX_ARRAY_ATTRIBUTE_NUMITEMS, &size, sizeof(size));
  vxQueryArray(kp_prev_list_, VX_ARRAY_ATTRIBUTE_CAPACITY, &max_capacity,
               sizeof(size));

  // TODO: check speed of this, maybe there is faster way to transfer all the
  // memory
  for (auto i = 0; i < elems_to_update; ++i) {
    nvx_keypointf_t* myptr = NULL;
    vxAccessArrayRange(kp_prev_list_, i, i + 1, &stride, (void**)&myptr,
                       VX_READ_AND_WRITE);
    myptr->x = prev_pts.at(i).x;
    myptr->y = prev_pts.at(i).y;
    myptr->tracking_status = 1;
    vxCommitArrayRange(kp_prev_list_, i, i + 1, (void*)myptr);
  }
}

FeatureTrackerReal::FeatureTrackerReal(vx_context context, const Params& params)
    : params_(params) {
  context_ = context;

  format_ = VX_DF_IMAGE_VIRT;
  width_ = 0;
  height_ = 0;

  dl_time_ = 0;
  valid_time_ = 0;

  pyr_delay_ = nullptr;
  kp_next_list_ = nullptr;
  kp_prev_list_ = nullptr;
  kp_back_list_ = nullptr;

  main_graph_ = nullptr;
  pyr_node_ = nullptr;
  opt_flow_node_forward_ = nullptr;
  opt_flow_node_backward_ = nullptr;

  dl_time_ = 0;
  valid_time_ = 0;

  is_initialized = false;
}

FeatureTrackerReal::~FeatureTrackerReal() { release(); }

void FeatureTrackerReal::release() {
  format_ = VX_DF_IMAGE_VIRT;
  width_ = 0;
  height_ = 0;

  vxReleaseDelay(&pyr_delay_);
  vxReleaseArray(&kp_next_list_);
  vxReleaseArray(&kp_back_list_);
  vxReleaseArray(&kp_prev_list_);

  vxReleaseNode(&pyr_node_);
  vxReleaseNode(&opt_flow_node_forward_);
  vxReleaseNode(&opt_flow_node_backward_);

  vxReleaseGraph(&main_graph_);
}

void FeatureTrackerReal::init(vx_image first_frame,
                              std::vector<nvx_keypointf_t>& points) {
  // Check input format

  vx_df_image format = VX_DF_IMAGE_VIRT;
  vx_uint32 width = 0;
  vx_uint32 height = 0;

  vxQueryImage(first_frame, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format));
  vxQueryImage(first_frame, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width));
  vxQueryImage(first_frame, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height));

  NVXIO_ASSERT(format == VX_DF_IMAGE_U8);

  // Re-create graph if the input size was changed

  if (width != width_ || height != height_) {
    release();

    format_ = format;
    width_ = width;
    height_ = height;

    createDataObjects();

    uploadPoints(points);

    createMainGraph(first_frame);
  }

  // Process first frame

  processFirstFrame(first_frame);
}

//
// For the subsequent frames, we call FeatureTracker::track() which
// essentially updates the input parameters passed to the graph. The previous
// pyramid and the tracked points in the previous frame are set by the
// vxAgeDelay(). The current Frame and the current mask are set by
// vxSetParameterByIndex. Finally vxProcessGraph() is called to execute the
// graph
//

void FeatureTrackerReal::track(vx_image newFrame, vx_image mask) {
  // Check input format

  vx_df_image format = VX_DF_IMAGE_VIRT;
  vx_uint32 width = 0;
  vx_uint32 height = 0;

  NVXIO_SAFE_CALL(vxQueryImage(newFrame, VX_IMAGE_ATTRIBUTE_FORMAT, &format,
                               sizeof(format)));
  NVXIO_SAFE_CALL(
      vxQueryImage(newFrame, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
  NVXIO_SAFE_CALL(vxQueryImage(newFrame, VX_IMAGE_ATTRIBUTE_HEIGHT, &height,
                               sizeof(height)));

  NVXIO_ASSERT(format == format_);
  NVXIO_ASSERT(width == width_);
  NVXIO_ASSERT(height == height_);

  if (mask) {
    vx_df_image mask_format = VX_DF_IMAGE_VIRT;
    vx_uint32 mask_width = 0;
    vx_uint32 mask_height = 0;

    NVXIO_SAFE_CALL(vxQueryImage(mask, VX_IMAGE_ATTRIBUTE_FORMAT, &mask_format,
                                 sizeof(mask_format)));
    NVXIO_SAFE_CALL(vxQueryImage(mask, VX_IMAGE_ATTRIBUTE_WIDTH, &mask_width,
                                 sizeof(mask_width)));
    NVXIO_SAFE_CALL(vxQueryImage(mask, VX_IMAGE_ATTRIBUTE_HEIGHT, &mask_height,
                                 sizeof(mask_height)));

    NVXIO_ASSERT(mask_format == VX_DF_IMAGE_U8);
    NVXIO_ASSERT(mask_width == width_);
    NVXIO_ASSERT(mask_height == height_);
  }

  // Update input parameters for next graph execution

  //  NVXIO_SAFE_CALL(
  //      vxSetParameterByIndex(feature_track_node_, 2, (vx_reference)mask));

  // Age the delay objects (pyramid, points to track) before graph execution
  NVXIO_SAFE_CALL(vxAgeDelay(pyr_delay_));

  NVXIO_SAFE_CALL(vxSetParameterByIndex(pyr_node_, 0, (vx_reference)newFrame));

  // Process graph
  NVXIO_SAFE_CALL(vxProcessGraph(main_graph_));
}

void FeatureTrackerReal::printPerfs() const {
  vx_size num_items = 0;
  NVXIO_SAFE_CALL(vxQueryArray(kp_next_list_, VX_ARRAY_ATTRIBUTE_NUMITEMS,
                               &num_items, sizeof(num_items)));
  std::cout << "Found " << num_items << " Features" << std::endl;

  vx_perf_t perf;

  NVXIO_SAFE_CALL(vxQueryGraph(main_graph_, VX_GRAPH_ATTRIBUTE_PERFORMANCE,
                               &perf, sizeof(perf)));
  std::cout << "Graph Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

  NVXIO_SAFE_CALL(vxQueryNode(pyr_node_, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf,
                              sizeof(perf)));
  std::cout << "\t Pyramid Time : " << perf.tmp / 1000000.0 << " ms"
            << std::endl;

  //  NVXIO_SAFE_CALL(vxQueryNode(
  //      feature_track_node_, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf,
  //      sizeof(perf)));
  //  std::cout << "\t Feature Track Time : " << perf.tmp / 1000000.0 << " ms"
  //            << std::endl;

  NVXIO_SAFE_CALL(vxQueryNode(opt_flow_node_forward_,
                              VX_NODE_ATTRIBUTE_PERFORMANCE, &perf,
                              sizeof(perf)));
  std::cout << "\t Optical Flow Time : " << perf.tmp / 1000000.0 << " ms"
            << std::endl;

  NVXIO_SAFE_CALL(vxQueryNode(opt_flow_node_backward_,
                              VX_NODE_ATTRIBUTE_PERFORMANCE, &perf,
                              sizeof(perf)));
  std::cout << "\t Optical Flow Time : " << perf.tmp / 1000000.0 << " ms"
            << std::endl;
}

//
// CreateDataObjects creates data objects that are not entirely linked to
// graphs. It creates two vx_delay references: pyr_delay_ and pts_delay_.
// pyr_delay_ holds image pyramids of two successive frames and pts_delay_
// holds the tracked points from the previous frame that will be used as an
// input to the pipeline, which is constructed by the createMainGraph()
//

void FeatureTrackerReal::createDataObjects() {
  //
  // Image pyramids for two successive frames are necessary for the computation.
  // A delay object with 2 slots is created for this purpose
  //

  vx_pyramid pyr_exemplar =
      vxCreatePyramid(context_, params_.pyr_levels, VX_SCALE_PYRAMID_HALF,
                      width_, height_, VX_DF_IMAGE_U8);
  NVXIO_CHECK_REFERENCE(pyr_exemplar);
  pyr_delay_ = vxCreateDelay(context_, (vx_reference)pyr_exemplar, 2);
  NVXIO_CHECK_REFERENCE(pyr_delay_);
  vxReleasePyramid(&pyr_exemplar);

  //
  // Create the list of tracked points. This is the output of the frame
  // processing
  //

  kp_next_list_ =
      vxCreateArray(context_, NVX_TYPE_KEYPOINTF, params_.array_capacity);
  NVXIO_CHECK_REFERENCE(kp_next_list_);
  kp_prev_list_ =
      vxCreateArray(context_, NVX_TYPE_KEYPOINTF, params_.array_capacity);
  NVXIO_CHECK_REFERENCE(kp_prev_list_);
  kp_back_list_ =
      vxCreateArray(context_, NVX_TYPE_KEYPOINTF, params_.array_capacity);
  NVXIO_CHECK_REFERENCE(kp_back_list_);
}

//
// The processFirstFrame() converts the first frame into grayscale,
// builds initial Gaussian pyramid, and detects initial keypoints.
//

void FeatureTrackerReal::processFirstFrame(vx_image frame) {
  NVXIO_CHECK_REFERENCE(frame);
  NVXIO_SAFE_CALL(vxuGaussianPyramid(
      context_, frame, (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, 0)));
}

//
// The createMainGraph() creates the pipeline. frame is passed as an input
// argument to createMainGraph(). It is subsequently overwritten in the function
// track() via the vxSetParameterByIndex()
//

void FeatureTrackerReal::createMainGraph(vx_image frame) {
  main_graph_ = vxCreateGraph(context_);
  NVXIO_CHECK_REFERENCE(main_graph_);

  //
  // Intermediate images. Both images are created as virtual in order to
  // inform the OpenVX framework that the application will never access their
  // content
  //

  vx_image frameGray =
      vxCreateVirtualImage(main_graph_, width_, height_, VX_DF_IMAGE_U8);
  NVXIO_CHECK_REFERENCE(frameGray);

  //
  // Lucas-Kanade optical flow node
  // Note: keypoints of the previous frame are also given as 'new points
  // estimates'
  //

  vx_float32 lk_epsilon = 0.01f;
  vx_scalar s_lk_epsilon =
      vxCreateScalar(context_, VX_TYPE_FLOAT32, &lk_epsilon);
  NVXIO_CHECK_REFERENCE(s_lk_epsilon);

  vx_scalar s_lk_num_iters =
      vxCreateScalar(context_, VX_TYPE_UINT32, &params_.lk_num_iters);
  NVXIO_CHECK_REFERENCE(s_lk_num_iters);

  vx_bool lk_use_init_est = vx_false_e;
  vx_scalar s_lk_use_init_est =
      vxCreateScalar(context_, VX_TYPE_BOOL, &lk_use_init_est);
  NVXIO_CHECK_REFERENCE(s_lk_use_init_est);

  pyr_node_ =
      vxGaussianPyramidNode(main_graph_, frameGray,
                            (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, 0));
  NVXIO_CHECK_REFERENCE(pyr_node_);

  //
  // vxOpticalFlowPyrLKNode accepts input arguements as current pyramid,
  // previous pyramid and points tracked in the previous frame. The output
  // is the set of points tracked in the current frame
  //

  opt_flow_node_forward_ = vxOpticalFlowPyrLKNode(
      main_graph_,
      (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, -1),  // previous pyramid
      (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, 0),   // current pyramid
      kp_prev_list_,  // points to track from previous frame
      kp_prev_list_,
      kp_next_list_,  // points tracked in current frame
      VX_TERM_CRITERIA_BOTH, s_lk_epsilon, s_lk_num_iters, s_lk_use_init_est,
      params_.lk_win_size);
  NVXIO_CHECK_REFERENCE(opt_flow_node_forward_);

  opt_flow_node_backward_ = vxOpticalFlowPyrLKNode(
      main_graph_,
      (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, 0),   // current pyramid
      (vx_pyramid)vxGetReferenceFromDelay(pyr_delay_, -1),  // previous pyramid
      kp_next_list_,  // points to track from current estimate
      kp_next_list_,
      kp_back_list_,  // points tracked in the previous frame
      VX_TERM_CRITERIA_BOTH, s_lk_epsilon, s_lk_num_iters, s_lk_use_init_est,
      params_.lk_win_size);
  NVXIO_CHECK_REFERENCE(opt_flow_node_backward_);

  // Ensure highest graph optimization level
  const char* option = "-O3";
  NVXIO_SAFE_CALL(vxSetGraphAttribute(main_graph_, NVX_GRAPH_VERIFY_OPTIONS,
                                      option, strlen(option)));
  //
  // Graph verification
  // Note: This verification is mandatory prior to graph execution
  //

  NVXIO_SAFE_CALL(vxVerifyGraph(main_graph_));

  vxReleaseScalar(&s_lk_epsilon);
  vxReleaseScalar(&s_lk_num_iters);
  vxReleaseScalar(&s_lk_use_init_est);
  vxReleaseImage(&frameGray);
}

FeatureTracker::Params::Params() {
  // Parameters for optical flow node
  pyr_levels = 6;
  lk_num_iters = 5;
  lk_win_size = 10;

  // Common parameters for corner detector node
  array_capacity = 2000;
  detector_cell_size = 18;
  use_harris_detector = true;

  // Parameters for harris_track node
  harris_k = 0.04f;
  harris_thresh = 100.0f;

  // Parameters for fast_track node
  fast_type = 9;
  fast_thresh = 25;
}

/*****************************************************************************/
/*                                                                           */
/*                        SYNTH TRACKER GRAPH                                */
/*                                                                           */
/*****************************************************************************/
FeatureTrackerSynth::FeatureTrackerSynth(vx_context context,
                                         const Params& params)
    : params_(params) {
  context_ = context;

  format_ = VX_DF_IMAGE_VIRT;
  width_ = 0;
  height_ = 0;

  kp_back_list_ = nullptr;
  kp_next_list_ = nullptr;
  kp_prev_list_ = nullptr;

  main_graph_ = nullptr;
  pyr_node_img_ = nullptr;
  pyr_node_synth_ = nullptr;
  opt_flow_node_forward_ = nullptr;
  opt_flow_node_backward_ = nullptr;

  pyr_rendered = nullptr;
  pyr_image = nullptr;

  dl_time_ = 0;
  valid_time_ = 0;

  is_initialized = false;
}

FeatureTrackerSynth::~FeatureTrackerSynth() { release(); }

void FeatureTrackerSynth::release() {
  format_ = VX_DF_IMAGE_VIRT;
  width_ = 0;
  height_ = 0;

  vxReleaseArray(&kp_back_list_);
  vxReleaseArray(&kp_next_list_);
  vxReleaseArray(&kp_back_list_);

  vxReleaseNode(&pyr_node_img_);
  vxReleaseNode(&pyr_node_synth_);
  vxReleaseNode(&opt_flow_node_forward_);
  vxReleaseNode(&opt_flow_node_backward_);

  vxReleasePyramid(&pyr_rendered);
  vxReleasePyramid(&pyr_image);

  vxReleaseGraph(&main_graph_);
}

void FeatureTrackerSynth::createMainGraph(vx_image frame) {
  main_graph_ = vxCreateGraph(context_);
  NVXIO_CHECK_REFERENCE(main_graph_);

  //
  // Intermediate images. Both images are created as virtual in order to
  // inform the OpenVX framework that the application will never access their
  // content
  //

  vx_image frameGray =
      vxCreateVirtualImage(main_graph_, width_, height_, VX_DF_IMAGE_U8);
  NVXIO_CHECK_REFERENCE(frameGray);

  //
  // Lucas-Kanade optical flow node
  // Note: keypoints of the previous frame are also given as 'new points
  // estimates'
  //

  vx_float32 lk_epsilon = 0.01f;
  vx_scalar s_lk_epsilon =
      vxCreateScalar(context_, VX_TYPE_FLOAT32, &lk_epsilon);
  NVXIO_CHECK_REFERENCE(s_lk_epsilon);

  vx_scalar s_lk_num_iters =
      vxCreateScalar(context_, VX_TYPE_UINT32, &params_.lk_num_iters);
  NVXIO_CHECK_REFERENCE(s_lk_num_iters);

  vx_bool lk_use_init_est = vx_false_e;
  vx_scalar s_lk_use_init_est =
      vxCreateScalar(context_, VX_TYPE_BOOL, &lk_use_init_est);
  NVXIO_CHECK_REFERENCE(s_lk_use_init_est);

  pyr_node_synth_ = vxGaussianPyramidNode(main_graph_, frameGray, pyr_rendered);
  pyr_node_img_ = vxGaussianPyramidNode(main_graph_, frameGray, pyr_image);

  NVXIO_CHECK_REFERENCE(pyr_node_synth_);
  NVXIO_CHECK_REFERENCE(pyr_node_img_);

  vx_image mask = NULL;

  if (params_.use_harris_detector) {
    feature_track_node_ = nvxHarrisTrackNode(
        main_graph_, frameGray, kp_prev_list_, mask, nullptr, params_.harris_k,
        params_.harris_thresh, params_.detector_cell_size, nullptr);
  } else {
    feature_track_node_ = nvxFastTrackNode(
        main_graph_, frameGray, kp_prev_list_, mask, nullptr, params_.fast_type,
        params_.fast_thresh, params_.detector_cell_size, nullptr);
  }
  NVXIO_CHECK_REFERENCE(feature_track_node_);

  //
  // vxOpticalFlowPyrLKNode accepts input arguements as current pyramid,
  // previous pyramid and points tracked in the previous frame. The output
  // is the set of points tracked in the current frame
  //

  opt_flow_node_forward_ = vxOpticalFlowPyrLKNode(
      main_graph_,
      pyr_rendered,   // previous pyramid
      pyr_image,      // current pyramid
      kp_prev_list_,  // points to track from previous frame
      kp_prev_list_,
      kp_next_list_,  // points tracked in current frame
      VX_TERM_CRITERIA_BOTH, s_lk_epsilon, s_lk_num_iters, s_lk_use_init_est,
      params_.lk_win_size);
  NVXIO_CHECK_REFERENCE(opt_flow_node_forward_);

  opt_flow_node_backward_ = vxOpticalFlowPyrLKNode(
      main_graph_,
      pyr_image,      // current pyramid
      pyr_rendered,   // previous pyramid
      kp_next_list_,  // points to track from current estimate
      kp_next_list_,
      kp_back_list_,  // points tracked in the previous frame
      VX_TERM_CRITERIA_BOTH, s_lk_epsilon, s_lk_num_iters, s_lk_use_init_est,
      params_.lk_win_size);
  NVXIO_CHECK_REFERENCE(opt_flow_node_backward_);

  // Ensure highest graph optimization level
  const char* option = "-O3";
  NVXIO_SAFE_CALL(vxSetGraphAttribute(main_graph_, NVX_GRAPH_VERIFY_OPTIONS,
                                      option, strlen(option)));
  //
  // Graph verification
  // Note: This verification is mandatory prior to graph execution
  //

  NVXIO_SAFE_CALL(vxVerifyGraph(main_graph_));

  vxReleaseScalar(&s_lk_epsilon);
  vxReleaseScalar(&s_lk_num_iters);
  vxReleaseScalar(&s_lk_use_init_est);
  vxReleaseImage(&frameGray);
}

void FeatureTrackerSynth::createDataObjects() {
  //
  // Image pyramids for two successive frames are necessary for the computation.
  // A delay object with 2 slots is created for this purpose
  //
  pyr_rendered =
      vxCreatePyramid(context_, params_.pyr_levels, VX_SCALE_PYRAMID_HALF,
                      width_, height_, VX_DF_IMAGE_U8);

  pyr_image =
      vxCreatePyramid(context_, params_.pyr_levels, VX_SCALE_PYRAMID_HALF,
                      width_, height_, VX_DF_IMAGE_U8);

  kp_next_list_ =
      vxCreateArray(context_, NVX_TYPE_KEYPOINTF, params_.array_capacity);
  NVXIO_CHECK_REFERENCE(kp_next_list_);
  kp_prev_list_ =
      vxCreateArray(context_, NVX_TYPE_KEYPOINTF, params_.array_capacity);
  NVXIO_CHECK_REFERENCE(kp_prev_list_);
  kp_back_list_ =
      vxCreateArray(context_, NVX_TYPE_KEYPOINTF, params_.array_capacity);
  NVXIO_CHECK_REFERENCE(kp_back_list_);
}

void FeatureTrackerSynth::init(vx_image sample_image) {
  // Check input format

  vx_df_image format = VX_DF_IMAGE_VIRT;
  vx_uint32 width = 0;
  vx_uint32 height = 0;

  vxQueryImage(sample_image, VX_IMAGE_ATTRIBUTE_FORMAT, &format,
               sizeof(format));
  vxQueryImage(sample_image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width));
  vxQueryImage(sample_image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height,
               sizeof(height));

  NVXIO_ASSERT(format == VX_DF_IMAGE_U8);

  if (width != width_ || height != height_) {
    release();

    format_ = format;
    width_ = width;
    height_ = height;

    createDataObjects();

    createMainGraph(sample_image);
  }
}

void FeatureTrackerSynth::track(vx_image rendered, vx_image curr_frame) {
  vx_df_image format = VX_DF_IMAGE_VIRT;
  vx_uint32 width = 0;
  vx_uint32 height = 0;

  NVXIO_SAFE_CALL(vxQueryImage(rendered, VX_IMAGE_ATTRIBUTE_FORMAT, &format,
                               sizeof(format)));
  NVXIO_SAFE_CALL(
      vxQueryImage(rendered, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
  NVXIO_SAFE_CALL(vxQueryImage(rendered, VX_IMAGE_ATTRIBUTE_HEIGHT, &height,
                               sizeof(height)));

  NVXIO_ASSERT(format == format_);
  NVXIO_ASSERT(width == width_);
  NVXIO_ASSERT(height == height_);

  NVXIO_SAFE_CALL(
      vxSetParameterByIndex(pyr_node_synth_, 0, (vx_reference)rendered));
  NVXIO_SAFE_CALL(
      vxSetParameterByIndex(pyr_node_img_, 0, (vx_reference)curr_frame));
  NVXIO_SAFE_CALL(
      vxSetParameterByIndex(feature_track_node_, 0, (vx_reference)rendered));

  // Process graph
  NVXIO_SAFE_CALL(vxProcessGraph(main_graph_));
}

void FeatureTrackerSynth::printPerfs() const {
  vx_size num_items = 0;
  NVXIO_SAFE_CALL(vxQueryArray(kp_next_list_, VX_ARRAY_ATTRIBUTE_NUMITEMS,
                               &num_items, sizeof(num_items)));
  std::cout << "Found " << num_items << " Features" << std::endl;

  vx_perf_t perf;

  NVXIO_SAFE_CALL(vxQueryGraph(main_graph_, VX_GRAPH_ATTRIBUTE_PERFORMANCE,
                               &perf, sizeof(perf)));
  std::cout << "Graph Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

  NVXIO_SAFE_CALL(vxQueryNode(pyr_node_synth_, VX_NODE_ATTRIBUTE_PERFORMANCE,
                              &perf, sizeof(perf)));
  std::cout << "\t Pyramid Synth Time : " << perf.tmp / 1000000.0 << " ms"
            << std::endl;

  NVXIO_SAFE_CALL(vxQueryNode(pyr_node_img_, VX_NODE_ATTRIBUTE_PERFORMANCE,
                              &perf, sizeof(perf)));
  std::cout << "\t Pyramid Img Time : " << perf.tmp / 1000000.0 << " ms"
            << std::endl;

  NVXIO_SAFE_CALL(vxQueryNode(
      feature_track_node_, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)));
  std::cout << "\t Feature Time : " << perf.tmp / 1000000.0 << " ms"
            << std::endl;

  NVXIO_SAFE_CALL(vxQueryNode(opt_flow_node_forward_,
                              VX_NODE_ATTRIBUTE_PERFORMANCE, &perf,
                              sizeof(perf)));
  std::cout << "\t Optical Flow Time : " << perf.tmp / 1000000.0 << " ms"
            << std::endl;

  NVXIO_SAFE_CALL(vxQueryNode(opt_flow_node_backward_,
                              VX_NODE_ATTRIBUTE_PERFORMANCE, &perf,
                              sizeof(perf)));
  std::cout << "\t Optical Flow Time : " << perf.tmp / 1000000.0 << " ms"
            << std::endl;

  std::cout << "\t Download Time: " << dl_time_ / 1000000.0 << " ms"
            << std::endl;
  std::cout << "\t Valid Time: " << valid_time_ / 1000000.0 << " ms"
            << std::endl;
}

}  // end namespace
}  // end namespace fato
