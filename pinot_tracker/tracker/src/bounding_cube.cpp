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
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../include/bounding_cube.h"
#include "../../utilities/include/utilities.h"
#include "../../utilities/include/DebugFunctions.h"
#include "../../utilities/include/draw_functions.h"
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
      max_depth(4.0f),
      accumulated_mean_(0.0f),
      accumulated_counter_(0),
      means_() {}

void BoundingCube::initCube(const cv::Mat &points, const cv::Point2f &top_left,
                            const cv::Point2f &bottom_right) {
  // creating bounding box mask
  Mat1b mask(points.rows, points.cols, uchar(0));
  rectangle(mask, top_left, bottom_right, uchar(255));

  Point3f min_val, max_val;
  float median_z;
  getValues(points, mask, min_val, max_val, median_z);

  auto projectPoint =
      [&](const cv::Point2f &pt, float depth, cv::Point3f &out) {
        float xp = pt.x - cx_;
        float yp = -(pt.y - cy_);

        out.x = xp * depth / fx_;
        out.y = yp * depth / fy_;
        out.z = depth;
      };

  Point3f tl, br;
  projectPoint(top_left, median_z, tl);
  projectPoint(bottom_right, median_z, br);

  front_points_.at(0) = Point3f(tl.x, tl.y, median_z);
  front_points_.at(1) = Point3f(br.x, tl.y, median_z);
  front_points_.at(2) = Point3f(br.x, br.y, median_z);
  front_points_.at(3) = Point3f(tl.x, br.y, median_z);

  float width = max_val.x - min_val.x;
  float height = max_val.y - min_val.y;
  float depth = median_z + std::min(width, height);

  back_points_.at(0) = Point3f(tl.x, tl.y, depth);
  back_points_.at(1) = Point3f(br.x, tl.y, depth);
  back_points_.at(2) = Point3f(br.x, br.y, depth);
  back_points_.at(3) = Point3f(tl.x, br.y, depth);

  float median_x = min_val.x + (max_val.x - min_val.x) / 2.0f;
  float median_y = min_val.y + (max_val.y - min_val.y) / 2.0f;

  centroid_ = Point3f(median_x, median_y, median_z);

  for (auto i = 0; i < 4; ++i) {
    front_vectors_.at(i) = front_points_.at(i) - centroid_;
    back_vectors_.at(i) = back_points_.at(i) - centroid_;
  }
}

void BoundingCube::setPerspective(float fx, float fy, float cx, float cy) {
  fx_ = fx;
  fy_ = fy;
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
                                 const Mat &rotation,
                                 std::vector<float> &visibility_ratio,
                                 Mat &out) {
  cout << "bb -2 ";
  // create virtual infinite bounding box
  vector<Point3f> front, back, spawn_front;
  vector<Point3f> back_vect = back_vectors_;
  vector<Point3f> front_spawn_vect = front_vectors_;
  cout << "-1 ";
  for (auto &pt : back_vect) pt.z += 1.5f;
  for (auto &pt : front_spawn_vect) pt.z += 0.01f;

  try {
    rotateBBox(front_vectors_, rotation, front);
    rotateBBox(back_vect, rotation, back);
    rotateBBox(front_spawn_vect, rotation, spawn_front);
  } catch (cv::Exception &e) {
    ROS_ERROR("BOUNDING_CUBE: error in rotation");
    return;
  }
  cout << "bb 0 ";
  for (const auto &pt : front) {
    if (!is_valid(pt.x) || !is_valid(pt.y)) {
      return;
    }
  }

  for (const auto &pt : back) {
    if (!is_valid(pt.x) || !is_valid(pt.y)) {
      return;
    }
  }

  for (auto i = 0; i < front.size(); ++i) {
    front.at(i) += center;
    back.at(i) += center;
    spawn_front.at(i) += center;
  }

  cout << " 1 ";
  // select the side of the box that is visible

  // create the estension of that face
  Point2f center_image(cx_, cy_);
  int selected_face = -1;
  float face_visibility = 0.0f;

  Point2f top_front(0, 0), down_front(0, 0), top_back(0, 0), down_back(0, 0);
  getFace(visibility_ratio, spawn_front, back, center_image, top_front,
          down_front, top_back, down_back, selected_face, face_visibility);

  // no face selected
  if (selected_face == -1) return;
  // face not visible enough
  if (face_visibility < 0.3f) return;

  cout << " 2 ";
  Scalar color(0, 255, 255);
  //  drawBoundingCube(center, front, back, fx_, center_image, color, 2, out);
  drawBoundingCube(center, front, back, fx_, center_image, out);
  cout << " 2-1 ";
  Mat1b mask(out.rows, out.cols, uchar(0));
  drawTriangleMask(top_front, down_front, top_back, mask);
  drawTriangleMask(top_back, down_front, down_back, mask);
  cout << " 2-2 ";
  drawTriangle(top_front, down_front, top_back, Scalar(255, 0, 0), 0.3, out);
  drawTriangle(top_back, down_front, down_back, Scalar(255, 0, 0), 0.3, out);

  // spawn linear connected components to find the depth of the object

  int num_tracks = 10;
  cout << " 3 ";
  vector<Point2f> track_start(num_tracks, Point2f(0, 0));
  vector<Point2f> track_end(num_tracks, Point2f(0, 0));
  vector<Point2f> depth_found(num_tracks, Point2f(0, 0));

  createLinearTracks(num_tracks, top_front, down_front, top_back, down_back,
                     track_start, track_end);
  cout << " 4 ";
  int line_jump = 2;
  spawnLinearCC(points, line_jump, track_start, track_end, depth_found);

  for (auto i = 1; i < track_start.size(); ++i) {
    if (depth_found.at(i).x != 0 && depth_found.at(i).y != 0) {
      circle(out, depth_found.at(i), 3, Scalar(0, 0, 255), -1);
    }
    line(out, track_start.at(i), track_end.at(i), Scalar(0, 255, 0), 1);
  }
  cout << " 5 ";
  float avg_depth, median_depth;
  getDepth(points, track_start, depth_found, avg_depth, median_depth);
  cout << " 6 \n";
  drawEstimatedCube(center, rotation, median_depth, Scalar(0, 255, 255), out);

  averages_.push_back(median_depth);
  if (averages_.size() > 90) averages_.pop_front();

  auto sliding_average = accumulate(averages_.begin(), averages_.end(), 0.0f) /
                         static_cast<float>(averages_.size());

  // current visibility is inferior to the one already estimated
  if (stored_estimations_.size() > 0) {
    if (face_visibility > stored_estimations_.back().first)
      stored_estimations_.push_back(
          pair<float, float>(face_visibility, sliding_average));

    cout << "Estimated depth, side view " << stored_estimations_.back().first
         << " depth  " << stored_estimations_.back().second
         << " estimation stored " << stored_estimations_.size() << endl;
  }
  else
  {
      stored_estimations_.push_back(
          pair<float, float>(face_visibility, sliding_average));
  }

  //  sort(stored_estimations_.begin(), stored_estimations_.end(), []
  //       (const pair<float,float>& a, const pair<float,float>& b)
  //  {return a.first > b.first;});




  // drawEstimatedCube(center, rotation, average_median, Scalar(255,0,255),
  // out);
  // drawEstimatedCube(center, rotation, median, Scalar(255,255,0), out);
  drawEstimatedCube(center, rotation, sliding_average, Scalar(0, 255, 125),
                    out);
}

void BoundingCube::linearCC(const Point2f &begin, const Point2f &end,
                            const Mat &points, Point3f &max_depth) {
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

    // if (abs(curr_d - last_d) > 1.0f) break;
    if (curr_d == 0) break;

    // file << curr_x << " " << curr_y << " " << last_d << " " << curr_d <<
    // endl;

    last_d = curr_d;

    curr_x++;
    curr_y = static_cast<int>(alpha * (curr_x - begin.x) + begin.y);
  }

  max_depth.x = curr_x;
  max_depth.y = curr_y;
  max_depth.z = last_d;
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

void BoundingCube::getFace(const std::vector<float> &visibility,
                           const std::vector<Point3f> &front,
                           const std::vector<Point3f> &back,
                           const Point2f &center_image, Point2f &top_front,
                           Point2f &down_front, Point2f &top_back,
                           Point2f &down_back, int &face, float &face_vis) {
  if (visibility.at(FACE::FRONT) > 0) {
    int max_face = -1;
    float max_val = 0.1;

    for (auto i = 0; i < 6; ++i) {
      if (i == FACE::FRONT || i == FACE::BACK) continue;

      if (visibility.at(i) > max_val) {
        max_face = i;
        max_val = visibility.at(i);
      }
    }

    face = max_face;
    face_vis = max_val;

    if (max_face == -1) {
      top_front = down_front = top_back = down_back = Point2f(0, 0);
    } else if (max_face == FACE::RIGHT) {
      top_front = projectPoint(fx_, center_image, front.at(0));
      down_front = projectPoint(fx_, center_image, front.at(3));
      top_back = projectPoint(fx_, center_image, back.at(0));
      down_back = projectPoint(fx_, center_image, back.at(3));
    } else if (max_face == FACE::LEFT) {
      top_front = projectPoint(fx_, center_image, front.at(1));
      down_front = projectPoint(fx_, center_image, front.at(2));
      top_back = projectPoint(fx_, center_image, back.at(1));
      down_back = projectPoint(fx_, center_image, back.at(2));
    } else if (max_face == FACE::TOP) {
      top_front = projectPoint(fx_, center_image, front.at(0));
      down_front = projectPoint(fx_, center_image, front.at(1));
      top_back = projectPoint(fx_, center_image, back.at(0));
      down_back = projectPoint(fx_, center_image, back.at(1));
    } else if (max_face == FACE::DOWN) {
      top_front = projectPoint(fx_, center_image, front.at(2));
      down_front = projectPoint(fx_, center_image, front.at(3));
      top_back = projectPoint(fx_, center_image, back.at(2));
      down_back = projectPoint(fx_, center_image, back.at(3));
    }
  } else {
    top_front = down_front = top_back = down_back = Point2f(0, 0);
  }
}

void BoundingCube::createLinearTracks(int num_tracks, const Point2f &top_front,
                                      const Point2f &down_front,
                                      const Point2d &top_back,
                                      const Point2f &down_back,
                                      std::vector<Point2f> &track_start,
                                      std::vector<Point2f> &track_end) {
  float step = static_cast<int>(num_tracks + 1);
  track_start.resize(num_tracks, Point2f(0, 0));
  track_end.resize(num_tracks, Point2f(0, 0));

  // define front and back step
  float step_front_x = (top_front.x - down_front.x) / step;
  float step_front_y = (top_front.y - down_front.y) / step;

  float step_back_x = (top_back.x - down_back.x) / step;
  float step_back_y = (top_back.y - down_back.y) / step;

  for (auto i = 1; i < num_tracks; ++i) {
    auto inc = i + 1;
    track_start.at(i).x = down_front.x + inc * step_front_x;
    track_start.at(i).y = down_front.y + inc * step_front_y;
    track_end.at(i).x = down_back.x + inc * step_back_x;
    track_end.at(i).y = down_back.y + inc * step_back_y;
  }
}

void BoundingCube::spawnLinearCC(const Mat &points, float line_jump,
                                 const std::vector<Point2f> &track_start,
                                 const std::vector<Point2f> &track_end,
                                 std::vector<Point2f> &depth_found) {
  depth_found.resize(track_start.size(), Point2f(0, 0));

  for (auto i = 0; i < track_start.size(); ++i) {
    // set to jump every line_jump pixels on the x
    int x_jumps = abs(track_end.at(i).x - track_start.at(i).x) / line_jump;
    float x_step =
        (track_end.at(i).x - track_start.at(i).x) / static_cast<float>(x_jumps);
    float y_step =
        (track_end.at(i).y - track_start.at(i).y) / static_cast<float>(x_jumps);

    if (x_jumps <= 0) continue;

    Point2f begin_pt = track_start.at(i);
    Vec3f last_pos = points.at<Vec3f>(begin_pt);
    for (int j = 0; j < x_jumps; ++j) {
      auto diff = abs(points.at<Vec3f>(begin_pt)[2] - last_pos[2]);

      if (diff > 0.02f || points.at<Vec3f>(begin_pt)[2] == 0) {
        begin_pt.x -= x_step;
        begin_pt.y -= y_step;
        depth_found.at(i) = begin_pt;
        //        depth_found.at(i) = last_pos;
        break;
      }
      begin_pt.x += x_step;
      begin_pt.y += y_step;
      last_pos = points.at<Vec3f>(begin_pt);
    }
  }
}

void BoundingCube::getDepth(const Mat &points,
                            const std::vector<Point2f> &track_start,
                            const std::vector<Point2f> &depth_found,
                            float &average_distance, float &median_distance) {
  vector<float> distances;
  float avg = 0;
  int valid_distances = 0;
  for (auto i = 0; i < track_start.size(); ++i) {
    const auto &fst =
        points.at<Vec3f>(track_start.at(i).y, track_start.at(i).x);
    const auto &scd =
        points.at<Vec3f>(depth_found.at(i).y, depth_found.at(i).x);
    auto dis = getDistance(fst, scd);

    if (dis > 0) {
      distances.push_back(dis);
      avg += dis;
      valid_distances++;
    }
  }

  if (distances.size() == 0) {
    average_distance = 0;
    median_distance = 0;
    return;
  }

  average_distance = avg / static_cast<float>(valid_distances);

  sort(distances.begin(), distances.end());
  auto median_point = distances.size() / 2;

  if (distances.size() % 2 == 0)
    median_distance =
        (distances.at(median_point - 1) + distances.at(median_point)) / 2.0f;
  else
    median_distance = distances.at(median_point);
}

void BoundingCube::drawEstimatedCube(Point3f &center, const Mat &rotation,
                                     float estimated_depth, Scalar color,
                                     Mat &out) {
  if (estimated_depth <= 0) return;

  // drawing estimation results
  vector<Point3f> estimated_vect = front_vectors_;
  for (auto &pt : estimated_vect) pt.z += estimated_depth;

  vector<Point3f> front, back;
  try {
    rotateBBox(front_vectors_, rotation, front);
    rotateBBox(estimated_vect, rotation, back);
  } catch (cv::Exception &e) {
    ROS_ERROR("BOUNDING_CUBE: error in rotation");
    return;
  }
  for (auto i = 0; i < 4; ++i) {
    front.at(i) += center;
    back.at(i) += center;
  }

  Point2f center_image(cx_, cy_);
  drawBoundingCube(front, back, fx_, center_image, color, 2, out);
}

float BoundingCube::getEstimatedDepth() {
  if (stored_estimations_.size() == 0)
    return -1;
  else
    return stored_estimations_.back().second;
}

}  // end namespace
