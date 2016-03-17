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

#ifndef DRAW_FUNCTIONS_H
#define DRAW_FUNCTIONS_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "constants.h"

namespace fato {

void drawBoundingBox(const std::vector<cv::Point2f>& box, cv::Scalar color,
                     int line_width, cv::Mat& out);

void drawBoundingCube(const cv::Point3f& center,
                      const std::vector<cv::Point3f>& front_box,
                      const std::vector<cv::Point3f>& back_box,
                      const float focal, const cv::Point2f& imgCenter,
                      cv::Mat& out);

void drawBoundingCube(const std::vector<cv::Point3f>& front_box,
                      const std::vector<cv::Point3f>& back_box,
                      const float focal, const cv::Point2f& imgCenter,
                      int line_width, cv::Mat& out);

void drawBoundingCube(const std::vector<cv::Point3f>& front_box,
                      const std::vector<cv::Point3f>& back_box,
                      const float focal, const cv::Point2f& imgCenter,
                      const cv::Scalar& color, int line_width, cv::Mat& out);

void applyColorMap(const cv::Mat& in, cv::Mat& out);

void drawObjectLocation(const std::vector<cv::Point3f>& back_box,
                        const std::vector<cv::Point3f>& front_box,
                        const cv::Point3f& center,
                        const std::vector<bool>& visibleFaces,
                        const float focal, const cv::Point2f& imgCenter,
                        cv::Mat& out);

void drawCentroidVotes(const std::vector<cv::Point3f*>& points,
                       const std::vector<cv::Point3f*>& votes,
                       const cv::Point2f& center, bool drawLines,
                       const float focal, cv::Mat& out);

void drawObjectPose(const cv::Point3f& centroid, const float focal,
                    const cv::Point2f& img_center, const cv::Mat& rotation,
                    cv::Mat& out);

void arrowedLine(cv::Mat& img, cv::Point2f pt1, cv::Point2f pt2,
                 const cv::Scalar& color, int thickness = 1, int line_type = 8,
                 int shift = 0, double tipLength = 0.1);

void cross(cv::Mat& img, cv::Point2f center, const cv::Scalar& color,
           int thickness = 1, int line_offset = 1, int line_type = 8, int shift = 0,
           double tipLength = 0.1);

}  // end namespace

#endif  // DRAW_FUNCTIONS_H
