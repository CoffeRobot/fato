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

#ifndef BOUNDINGCUBE_H
#define BOUNDINGCUBE_H

#include <opencv2/core/core.hpp>
#include <vector>
#include <iostream>

namespace pinot_tracker {

class BoundingCube {
 public:
  BoundingCube();

  void initCube(const cv::Mat& points, const cv::Point2f& top_left,
                const cv::Point2f& bottom_right);

  void setPerspective(float fx, float fy, float cx, float cy);

  void setVects(std::vector<cv::Point3f> front, std::vector<cv::Point3f> back) {
    front_vectors_ = front;
    back_vectors_ = back;
  }

  void rotate(cv::Point3f center, const cv::Mat& rotation,
              std::vector<cv::Point3f>& front_rot,
              std::vector<cv::Point3f>& back_rot);

  void estimateDepth(const cv::Mat& points, cv::Point3f center,
                     const cv::Mat& rotation,
                     std::vector<float>& visibility_ratio, cv::Mat& out);

  void linearCC(const cv::Point2f& begin, const cv::Point2f& end,
                const cv::Mat& points, cv::Point3f& max_depth);

  std::vector<cv::Point3f> getFrontPoints() { return front_points_; }
  std::vector<cv::Point3f> getBackPoints() { return back_points_; }

  cv::Point3f getCentroid() { return centroid_; }

  std::string str();

 private:
  void getValues(const cv::Mat& points, const cv::Mat1b& mask,
                 cv::Point3f& min_val, cv::Point3f& max_val,
                 float& median_depth);

  void getFace(const std::vector<float>& visibility,
               const std::vector<cv::Point3f>& front,
               const std::vector<cv::Point3f>& back,
               const cv::Point2f& center_image, cv::Point2f& top_front,
               cv::Point2f& down_front, cv::Point2f& top_back,
               cv::Point2f& down_back);

  void createLinearTracks(int num_tracks, const cv::Point2f& top_front,
                          const cv::Point2f& down_front,
                          const cv::Point2d& top_back,
                          const cv::Point2f& down_back,
                          std::vector<cv::Point2f>& track_start,
                          std::vector<cv::Point2f>& track_end);

  void spawnLinearCC(const cv::Mat& points, float line_jump,
                     const std::vector<cv::Point2f>& track_start,
                     const std::vector<cv::Point2f>& track_end,
                     std::vector<cv::Point2f>& depth_found);

  void getDepth(const cv::Mat& points,
                const std::vector<cv::Point2f>& track_start,
                const std::vector<cv::Point2f>& depth_found,
                float& average_distance, float& median_distance);

  void drawEstimatedCube(cv::Point3f& center, const cv::Mat& rotation,
                         float estimated_depth,
                         cv::Mat& out);

  std::vector<cv::Point3f> front_points_;
  std::vector<cv::Point3f> back_points_;
  std::vector<cv::Point3f> front_vectors_;
  std::vector<cv::Point3f> back_vectors_;

  cv::Point3f centroid_;

  float fx_, fy_, cx_, cy_;
  float max_depth;
};

}  // end namespace

#endif  // CUBEESTIMATOR_H
