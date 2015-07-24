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

namespace pinot_tracker{

class BoundingCube {
 public:
  BoundingCube();

  void initCube(const cv::Mat& points, const cv::Point2f& top_left,
                const cv::Point2f& bottom_right);

  void setPerspective(float focal, float cx, float cy);

  void rotate(cv::Point3f center, const cv::Mat& rotation, std::vector<cv::Point3f>& front_rot,
              std::vector<cv::Point3f>& back_rot);

  std::vector<cv::Point3f> getFrontPoints(){return front_points_;}
  std::vector<cv::Point3f> getBackPoints(){return back_points_;}


 private:

  std::vector<cv::Point3f> front_points_;
  std::vector<cv::Point3f> back_points_;
  std::vector<cv::Point3f> front_vectors_;
  std::vector<cv::Point3f> back_vectors_;

  float focal_, cx_, cy_;
  float max_depth;
};

} // end namespace

#endif  // CUBEESTIMATOR_H
