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
      fx_(0),
      cx_(0),
      cy_(0),
      max_depth(4.0f) {}

void BoundingCube::initCube(const cv::Mat &points, const cv::Point2f &top_left,
                            const cv::Point2f &bottom_right) {
  // creating bounding box mask
  Mat1b mask(points.rows, points.cols, uchar(0));
  rectangle(mask, top_left, bottom_right, uchar(255));

  Point3f min_val, max_val;
  float median_z;
  getValues(points, mask, min_val, max_val, median_z);

  auto projectPoint = [&](const cv::Point2f &pt, float depth,
                                            cv::Point3f &out)
  {
    float xp = pt.x - cx_;
    float yp = -(pt.y - cy_);

    out.x = xp * depth / fx_;
    out.y = yp * depth / fx_;
    out.z = depth;
  };

  Point3f tl, br;
  projectPoint(top_left, median_z, tl);
  projectPoint(bottom_right, median_z, br);


//  front_points_.at(0) = Point3f(min_val.x, min_val.y, median_z);
//  front_points_.at(1) = Point3f(max_val.x, min_val.y, median_z);
//  front_points_.at(2) = Point3f(max_val.x, max_val.y, median_z);
//  front_points_.at(3) = Point3f(min_val.x, max_val.y, median_z);

  front_points_.at(0) = Point3f(tl.x, tl.y, median_z);
  front_points_.at(1) = Point3f(br.x, tl.y, median_z);
  front_points_.at(2) = Point3f(br.x, br.y, median_z);
  front_points_.at(3) = Point3f(tl.x, br.y, median_z);

  cout << "Computed Box..\n";
  cout << front_points_.at(0) << " " << front_points_.at(2) << endl;
  cout << tl << " " << br << endl;


  float width = max_val.x - min_val.x;
  float height = max_val.y - min_val.y;
  float depth = median_z + std::min(width, height);

  back_points_.at(0) = Point3f(min_val.x, min_val.y, depth);
  back_points_.at(1) = Point3f(max_val.x, min_val.y, depth);
  back_points_.at(2) = Point3f(max_val.x, max_val.y, depth);
  back_points_.at(3) = Point3f(min_val.x, max_val.y, depth);

  float median_x = min_val.x + (max_val.x - min_val.x)/2.0f;
  float median_y = min_val.y + (max_val.y - min_val.y)/2.0f;

  centroid_ =
      Point3f(median_x, median_y, median_z + (std::min(width, height) / 2.0f));

  for (auto i = 0; i < 4; ++i) {
    front_vectors_.at(i) = front_points_.at(i) - centroid_;
    back_vectors_.at(i) = back_points_.at(i) - centroid_;
  }
}

void BoundingCube::setPerspective(float focal, float cx, float cy) {
  fx_ = focal;
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

  for (auto &pt : back_vect) pt.z += 1.5f;

  rotateBBox(front_vectors_, rotation, front);
  rotateBBox(back_vect, rotation, back);

  for (auto i = 0; i < 4; ++i) {
    front.at(i) += center;
    back.at(i) += center;
  }

  Point2f center_image(cx_, cy_);

  // project to image the points of the cube
  Point2f top_front = projectPoint(fx_, center_image, front.at(1));
  Point2f down_front = projectPoint(fx_, center_image, front.at(2));
  Point2f top_back = projectPoint(fx_, center_image, back.at(1));
  Point2f down_back = projectPoint(fx_, center_image, back.at(2));

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

  Mat1b mask(points.rows, points.cols, uchar(0));
  drawTriangleMask(down_front, down_back, top_front, mask);
  drawTriangleMask(top_front, down_back, top_back, mask);

  imwrite("/home/alessandro/Debug/depth_mask.png", mask);

  columnwiseStats(points, top_front, top_back, down_front, down_back, file);

  //  float alpha_top = (top_back.y - top_front.y) / (top_back.x - top_front.x);
  //  float alpha_down = (down_back.y - down_front.y) / (down_back.x -
  // down_front.x);

  //  float y_top = alpha_top * (points.cols - top_front.x) + top_front.y;
  //  float y_down = alpha_down * (points.cols - down_front.x) + down_front.y;

  //  if(alpha_top != alpha_down)
  //  {
  //    float x = alpha_top * top_front.x - alpha_down * down_front.x -
  // top_front.y +
  //             down_front.y;
  //    x = x / (alpha_top - alpha_down);
  //    float y = alpha_top * (x - top_front.x) + top_front.y ;

  //    Mat1b mask(points.rows+2, points.cols+2, uchar(0));
  //    drawTriangleMask(top_front, down_front, Point2f(x,y), mask);

  //    Mat planes[3];
  //    split(points,planes);

  //    floodFill(planes[2], mask, top_front, Scalar(125), NULL, Scalar(0.02f),
  // Scalar(0.02f),
  //            cv::FLOODFILL_MASK_ONLY);

  //    imwrite("/home/alessandro/Debug/triangle_mask.png", mask);

  //    circle(out, Point2f(x,y), 3, Scalar(255,255,255), -1);
  //  }

  //  line(out, top_front, Point2f(points.cols-1, y_top), Scalar(0, 255, 0), 1);
  //  line(out, down_front, Point2f(points.cols-1, y_down), Scalar(0, 255, 0),
  // 1);

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
                            const Mat &points, Point3f &max_depth,
                            ofstream &file) {
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

    file << curr_x << " " << curr_y << " " << last_d << " " << curr_d << endl;

    last_d = curr_d;

    curr_x++;
    curr_y = static_cast<int>(alpha * (curr_x - begin.x) + begin.y);
  }

  max_depth.x = curr_x;
  max_depth.y = curr_y;
  max_depth.z = last_d;
}

void BoundingCube::columnwiseStats(const Mat &points, const Point2f &top_front,
                                   const Point2f &top_back,
                                   const Point2f &down_front,
                                   const Point2f &down_back, ofstream &file) {

  auto interpolate = [](const cv::Point2f &fst, const cv::Point2f &scd,
                        std::vector<float> &ys) {
    auto size = static_cast<int>(scd.x - fst.x);
    auto step = (fst.y - scd.y) / static_cast<float>(size);
    ys.resize(size, 0);
    ys.at(0) = fst.y;
    for (auto i = 1; i < size; ++i) ys.at(i) = ys.at(i - 1) + step;
  };

  auto minmax = [](const cv::Point2f &pt, cv::Point2f &min_pt,
                   cv::Point2f &max_pt) {
    if (pt.x > max_pt.x) max_pt.x = pt.x;
    if (pt.y > max_pt.y) max_pt.y = pt.y;
    if (pt.x < min_pt.x) min_pt.x = pt.x;
    if (pt.y < min_pt.y) min_pt.y = pt.y;
  };

  vector<float> top_ys, down_ys;

  interpolate(top_front, top_back, top_ys);
  interpolate(down_front, down_back, down_ys);

  Point2f min_pt, max_pt;

  min_pt.x = std::numeric_limits<int>::max();
  min_pt.y = std::numeric_limits<int>::max();
  max_pt.x = 0;
  max_pt.y = 0;

  minmax(top_front, min_pt, max_pt);
  minmax(top_back, min_pt, max_pt);
  minmax(down_front, min_pt, max_pt);
  minmax(down_back, min_pt, max_pt);

  file << "INTERPOLATION: \n";
  file << "minpt " << min_pt << " maxpt " << max_pt << "\n";

  vector<float> avg_depths;

  for (auto j = min_pt.x; j < max_pt.x; ++j) {
    int counter = 0;
    float average_depth = 0;
    for (auto i = min_pt.y; i < max_pt.y; ++i) {
      if (points.at<Vec3f>(j, i)[2] != 0) {
        counter++;
        average_depth += points.at<Vec3f>(j, i)[2];
      }
    }
    if (counter > 0)
      avg_depths.push_back(average_depth / static_cast<float>(counter));
    else
      avg_depths.push_back(0);
  }

  for (auto i = 0; i < top_ys.size(); ++i) {
    file << top_ys.at(i) << " " << down_ys.at(i) << "\n";
  }

  file << "AVERAGES \n";
  for (auto d : avg_depths) file << d << " ";

  file << "\n";
}

std::string BoundingCube::str() {
  stringstream ss;
  ss << "CUBE VALUES:\n";
  ss << "center: " << centroid_;
  ss << "top_left: " << front_points_.at(0);
  ss << "bottom_left: " << front_points_.at(2) << "\n";

  return ss.str();
}

void BoundingCube::getValues(const Mat &points, const Mat1b &mask,
                             Point3f &min_val, Point3f &max_val,
                             float &median_depth) {
  vector<float> depth_x, depth_y, depth_z;
  min_val = Point3f(numeric_limits<float>::max(), numeric_limits<float>::max(),
                    numeric_limits<float>::max());
  max_val =
      Point3f(-numeric_limits<float>::max(), -numeric_limits<float>::max(),
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

        min_val.x = std::min(min_val.x, x);
        min_val.y = std::min(min_val.y, y);
        min_val.z = std::min(min_val.z, z);

        max_val.x = std::max(max_val.x, x);
        max_val.y = std::max(max_val.y, y);
        max_val.z = std::max(max_val.z, z);

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

  auto half = size / 2;
  if (size % 2 == 0) {
    median_x = (depth_x.at(half) + depth_x.at(half + 1)) / 2.0f;
    median_y = (depth_y.at(half) + depth_y.at(half + 1)) / 2.0f;
    median_z = (depth_z.at(half) + depth_z.at(half + 1)) / 2.0f;
  } else {
    median_x = depth_x.at(half);
    median_y = depth_y.at(half);
    median_z = depth_z.at(half);
  }

  median_depth = median_z;
}

}  // end namespace
