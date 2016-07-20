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
#ifndef SYNTHETIC_TRACK_HPP
#define SYNTHETIC_TRACK_HPP

#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

#include <string>

#include "../../fato_rendering/include/multiple_rigid_models_ogre.h"
#include "../../fato_rendering/include/windowless_gl_context.h"
#include "../include/pose_estimation.h"

namespace fato {

class SyntheticTrack {
 public:
  SyntheticTrack();

  void init(double nodal_x, double nodal_y, double focal_x,
            double focal_y, int img_w, int img_h, pose::MultipleRigidModelsOgre* rendering_engine);

  std::pair<int, std::vector<double> > poseFromSynth(Pose prev_pose, cv::Mat &curr_img);

  void renderObject(Pose prev_pose, cv::Mat &rendered_image,
                    std::vector<float> &z_buffer);

 private:

  void blendImage();

  void downloadRenderedImage(std::vector<uchar4> &h_texture);

  void downloadZBuffer(std::vector<float> &buffer);

  void trackCorners(cv::Mat &rendered_image, cv::Mat &next_img,
                    std::vector<cv::Point2f> &prev_pts,
                    std::vector<cv::Point2f> &next_pts);

  void debug(cv::Mat& rendered_img, cv::Mat& next_img, std::vector<cv::Point2f>& prev_pts,
             std::vector<cv::Point2f>& next_pts, Pose& prev_pose, Pose& next_pose);

  pose::MultipleRigidModelsOgre* rendering_engine_;

  int img_w_, img_h_;
  double nodal_x_, nodal_y_, focal_x_, focal_y_;
};

}  // end namespace

#endif  // SYNTHETIC_TRACK_HPP
