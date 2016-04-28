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


#ifndef HELPERFUNCTIONS_H
#define HELPERFUNCTIONS_H

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/core/core.hpp>
#include <vector>

using namespace cv;
using namespace cv::gpu;
using namespace std;

namespace fato {

namespace gpu {

static void download(const GpuMat& d_mat, vector<Point2f>& vec) {
  vec.resize(d_mat.cols);
  Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
  d_mat.download(mat);
}

static void download(const GpuMat& d_mat, vector<uchar>& vec) {
  vec.resize(d_mat.cols);
  Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
  d_mat.download(mat);
}

static void upload(const vector<Point2f>& points, GpuMat& d_points) {
  Mat mat(1, points.size(), CV_32FC2, (void*)&points[0]);
  d_points.upload(mat);
}

/*inline float getDistance(const cv::Point2f& a, const cv::Point2f& b)
            {
                    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}*/


inline cv::Mat1b getMask(int rows, int cols, const cv::Point2d& begin,
                         const cv::Point2d& end) {
  cv::Mat1b mask(rows, cols, static_cast<uchar>(0));
  rectangle(mask, begin, end, static_cast<uchar>(255), -1);
  return mask;
}
}
}
#endif
