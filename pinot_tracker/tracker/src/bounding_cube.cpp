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
#include <vector>
#include "../include/bounding_cube.h"
#include "../../utilities/include/utilities.h"
#include "../include/pose_estimation.h"
#include <iostream>

using namespace cv;
using namespace std;

namespace pinot_tracker {

BoundingCube::BoundingCube()
    : front_points_(4, Point3f(0, 0, 0)),
      back_points_(4, Point3f(0, 0, 0)),
      front_vectors_(4, Point3f(0, 0, 0)),
      back_vectors_(4, Point3f(0, 0, 0)),
      focal_(0),
      cx_(0),
      cy_(0),
      max_depth(4.0f) {}

void BoundingCube::initCube(const cv::Mat &points, const cv::Point2f &top_left,
                            const cv::Point2f &bottom_right) {
  Mat1b mask(points.rows, points.cols, uchar(0));
  rectangle(mask, top_left, bottom_right, uchar(255));

  vector<float> depth_x, depth_y, depth_z;
  Point3f min_depth(numeric_limits<float>::max(), numeric_limits<float>::max(),
                    numeric_limits<float>::max());
  Point3f max_depth(-numeric_limits<float>::max(),
                    -numeric_limits<float>::max(),
                    -numeric_limits<float>::max());

  float average = 0;
  float counter = 0;
  for (int i = 0; i < points.rows; ++i) {
    for (int j = 0; j < points.cols; ++j) {
      if (mask.at<uchar>(i, j) == 255 && points.at<Vec3f>(i, j)[2] != 0 &&
          is_valid<float>(points.at<Vec3f>(i, j)[2])) {
        float x = points.at<Vec3f>(i, j)[0];
        float y = points.at<Vec3f>(i, j)[1];
        float z = points.at<Vec3f>(i, j)[2];

        depth_z.push_back(z);
        depth_x.push_back(x);
        depth_y.push_back(y);
        average += z;

        min_depth.x = std::min(min_depth.x, x);
        min_depth.y = std::min(min_depth.y, y);
        min_depth.z = std::min(min_depth.z, z);

        max_depth.x = std::max(max_depth.x, x);
        max_depth.y = std::max(max_depth.y, y);
        max_depth.z = std::max(max_depth.z, z);

        counter++;
      }
    }
  }

  sort(depth_x.begin(), depth_x.end());
  sort(depth_y.begin(), depth_y.end());
  sort(depth_z.begin(), depth_z.end());

  auto size = depth_x.size();

  if (size == 0) {
    cout << "No point to calculate median \n";
    return;
  }
  float median_x, median_y, median_z;

  if (size % 2 == 0) {
    median_x = (depth_x.at(size / 2) + depth_x.at(size / 2 + 1)) / 2.0f;
    median_y = (depth_y.at(size / 2) + depth_y.at(size / 2 + 1)) / 2.0f;
    median_z = (depth_z.at(size / 2) + depth_z.at(size / 2 + 1)) / 2.0f;
  } else {
    median_x = depth_x.at(size / 2);
    median_y = depth_y.at(size / 2);
    median_z = depth_z.at(size / 2);
  }

  front_points_.at(0) = Point3f(min_depth.x, min_depth.y, median_z);
  front_points_.at(1) = Point3f(max_depth.x, min_depth.y, median_z);
  front_points_.at(2) = Point3f(max_depth.x, max_depth.y, median_z);
  front_points_.at(3) = Point3f(min_depth.x, max_depth.y, median_z);

  float width = max_depth.x - min_depth.x;
  float height = max_depth.y - min_depth.y;
  float depth = median_z + std::min(width, height);

  back_points_.at(0) = Point3f(min_depth.x, min_depth.y, depth);
  back_points_.at(1) = Point3f(max_depth.x, min_depth.y, depth);
  back_points_.at(2) = Point3f(max_depth.x, max_depth.y, depth);
  back_points_.at(3) = Point3f(min_depth.x, max_depth.y, depth);

  Point3f center(median_x, median_y,
                 median_z + (std::min(width, height) / 2.0f));

  for(auto i = 0; i < 4; ++i)
  {
    front_vectors_.at(i) = front_points_.at(i) - center;
    back_vectors_.at(i) = back_points_.at(i) - center;
  }
}

void BoundingCube::setPerspective(float focal, float cx, float cy) {
  focal_ = focal;
  cx_ = cx;
  cy_ = cy;
}

void BoundingCube::rotate(cv::Point3f center, const Mat &rotation, std::vector<Point3f> &front_rot,
                          std::vector<Point3f> &back_rot) {
  rotateBBox(front_vectors_, rotation, front_rot);
  rotateBBox(back_vectors_, rotation, back_rot);

  for(auto i = 0; i < 4; ++i)
  {
    front_rot.at(i) += center;
    back_rot.at(i) += center;
  }
}

}  // end namespace
