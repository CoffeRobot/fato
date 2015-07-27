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

// TODO: remove all drawing and writing stuff here once the method to compute
// depth works
#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../include/bounding_cube.h"
#include "../../utilities/include/utilities.h"
#include "../../utilities/include/DebugFunctions.h"
#include "../include/pose_estimation.h"

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

  initCube(points, mask);
}

void BoundingCube::initCube(const Mat &points, const Mat &mask) {
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

  centroid_ =
      Point3f(median_x, median_y, median_z + (std::min(width, height) / 2.0f));

  for (auto i = 0; i < 4; ++i) {
    front_vectors_.at(i) = front_points_.at(i) - centroid_;
    back_vectors_.at(i) = back_points_.at(i) - centroid_;
  }
}

void BoundingCube::setPerspective(float focal, float cx, float cy) {
  focal_ = focal;
  cx_ = cx;
  cy_ = cy;
}

void BoundingCube::rotate(cv::Point3f center, const Mat &rotation,
                          std::vector<Point3f> &front_rot,
                          std::vector<Point3f> &back_rot) {
  rotateBBox(front_vectors_, rotation, front_rot);
  rotateBBox(back_vectors_, rotation, back_rot);

  for (auto i = 0; i < 4; ++i) {
    front_rot.at(i) += center;
    back_rot.at(i) += center;
  }
}

void BoundingCube::estimateDepth(const cv::Mat &points, Point3f center,
                                 const Mat &rotation, Mat &out) {
  vector<Point3f> front, back;



  vector<Point3f> back_vect = back_vectors_;

  for(auto& pt : back_vect)
      pt.z += 1.5f;


  rotateBBox(front_vectors_, rotation, front);
  rotateBBox(back_vect, rotation, back);

  for (auto i = 0; i < 4; ++i) {
    front.at(i) += center;
    back.at(i) += center;
  }

  Point2f center_image(cx_, cy_);

  // project to image the points of the cube
  Point2f top_front = projectPoint(focal_, center_image, front.at(1));
  Point2f down_front = projectPoint(focal_, center_image, front.at(2));
  Point2f top_back = projectPoint(focal_, center_image, back.at(1));
  Point2f down_back = projectPoint(focal_, center_image, back.at(2));

  float step = 3.0f;
  int istep = static_cast<int>(step);

  // define front and back step
  float step_front_x = (top_front.x - down_front.x) / step;
  float step_front_y = (top_front.y - down_front.y) / step;

  float step_back_x = (top_back.x - down_back.x) / step;
  float step_back_y = (top_back.y - down_back.y) / step;

  vector<Point2f> front_pts(istep, Point2f());
  vector<Point2f> back_pts(istep, Point2f());

  ofstream file("/home/alessandro/Debug/depth_estimation.txt");

  file << "top front " << top_front << "\n";
  file << "down front " << down_front << "\n";
  file << "top back " << top_back << "\n";
  file << "down back " << down_back << "\n";

  file << "step front " << step_front_x << " " << step_front_y << "\n";
  file << "step back " << step_back_x << " " << step_back_y << "\n";

  line(out, down_front, down_back, Scalar(0, 255, 0), 1);
  line(out, top_front, top_back, Scalar(0, 255, 0), 1);
  line(out, down_back, top_back, Scalar(0, 255, 0), 1);

//  float alpha_top = (top_back.y - top_front.y) / (top_back.x - top_front.x);
//  float alpha_down = (down_back.y - down_front.y) / (down_back.x - down_front.x);

//  float y_top = alpha_top * (points.cols - top_front.x) + top_front.y;
//  float y_down = alpha_down * (points.cols - down_front.x) + down_front.y;

//  if(alpha_top != alpha_down)
//  {
//    float x = alpha_top * top_front.x - alpha_down * down_front.x - top_front.y +
//             down_front.y;
//    x = x / (alpha_top - alpha_down);
//    float y = alpha_top * (x - top_front.x) + top_front.y ;

//    Mat1b mask(points.rows+2, points.cols+2, uchar(0));
//    drawTriangleMask(top_front, down_front, Point2f(x,y), mask);

//    Mat planes[3];
//    split(points,planes);

//    floodFill(planes[2], mask, top_front, Scalar(125), NULL, Scalar(0.02f), Scalar(0.02f),
//            cv::FLOODFILL_MASK_ONLY);

//    imwrite("/home/alessandro/Debug/triangle_mask.png", mask);

//    circle(out, Point2f(x,y), 3, Scalar(255,255,255), -1);
//  }


//  line(out, top_front, Point2f(points.cols-1, y_top), Scalar(0, 255, 0), 1);
//  line(out, down_front, Point2f(points.cols-1, y_down), Scalar(0, 255, 0), 1);

//  for (auto i = 0; i < istep; ++i) {
//    front_pts.at(i).x = down_front.x + i * step_front_x;
//    front_pts.at(i).y = down_front.y + i * step_front_y;
//    back_pts.at(i).x = down_back.x + i * step_back_x;
//    back_pts.at(i).y = down_back.y + i * step_back_y;

//    Point3f max_depth;
//    linearCC(front_pts.at(i), back_pts.at(i), points, max_depth, file);

//    file << front_pts.at(i) << " " << back_pts.at(i) << " max depth "
//         << max_depth << "\n";

//    line(out, front_pts.at(i), Point2f(max_depth.x, max_depth.y),
//         Scalar(0, 0, 255), 1);
//    line(out, front_pts.at(i), back_pts.at(i), Scalar(0, 255, 0), 1);
//  }

  file.close();
}

void BoundingCube::linearCC(const Point2f &begin, const Point2f &end,
                            const Mat &points, Point3f &max_depth, ofstream& file) {
  float alpha = (end.y - begin.y) / (end.x - begin.x);

  int curr_y = begin.y;
  int curr_x = begin.x;
  float last_d = points.at<Vec3f>(curr_x, curr_y)[2];
  float curr_d = last_d;

  while (1) {
    if (curr_y < 0) {
      curr_y = 0;
      break;
    }
    if (curr_y > points.rows) {
      curr_y = points.rows - 1;
    }
    if (curr_x < 0) {
      curr_x = 0;
      break;
    }
    if (curr_x > points.cols) {
      curr_x = points.cols - 1;
      break;
    }

    curr_d = points.at<Vec3f>(curr_x, curr_y)[2];

    if (abs(curr_d - last_d) > 0.02f) break;

    file << curr_x << " " << curr_y << " " << last_d << " "  << curr_d << endl;

    last_d = curr_d;

    curr_x++;
    curr_y = static_cast<int>(alpha * (curr_x - begin.x) + begin.y);
  }

  max_depth.x = curr_x;
  max_depth.y = curr_y;
  max_depth.z = last_d;
}

}  // end namespace
