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
#ifndef TARGET_HPP
#define TARGET_HPP

#include <vector>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>

#include "constants.h"

namespace fato {

class Target {
 public:
  Target() {}

  ~Target() {}

  void init(std::vector<cv::Point3f>& points, cv::Mat& descriptors);

  void removeInvalidPoints(const std::vector<int>& ids);

  bool isConsistent();

  void projectVectors(cv::Mat& camera, cv::Mat& out);

  std::vector<cv::Point3f> model_points_;
  std::vector<cv::Point3f> rel_distances_;
  std::vector<cv::Point2f> active_points;
  std::vector<cv::Point2f> prev_points_;
  std::vector<float> projected_depth_;
  std::vector<KpStatus> point_status_;

  cv::Mat descriptors_;

  cv::Point3f centroid_;

  std::vector<int> active_to_model_;

  cv::Mat rotation, rotation_custom;
  cv::Mat translation, translation_custom;
  cv::Mat rotation_vec;

  // position of points in previous frame, used for structure from motion pose
  // estimation
  std::vector<cv::Point3f> last_frame_points_;

  Eigen::MatrixXd pose_;
  bool target_found_;

};
}

#endif  // TARGET_HPP
