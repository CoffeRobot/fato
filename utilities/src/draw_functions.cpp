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

#include "../include/draw_functions.h"
#include "../include/utilities.h"
#include <opencv2/contrib/contrib.hpp>
#include "../include/constants.h"
#include <iostream>
#include <limits>


using namespace std;
using namespace cv;

namespace fato {

void drawBoundingBox(const vector<Point2f>& box, Scalar color, int line_width,
                     Mat& out) {
  line(out, box[0], box[1], color, line_width, 1);
  line(out, box[1], box[2], color, line_width, 1);
  line(out, box[2], box[3], color, line_width, 1);
  line(out, box[3], box[0], color, line_width, 1);
}

void drawBoundingCube(const Point3f& center, const vector<Point3f>& front_box,
                      const vector<Point3f>& back_box, const float focal,
                      const Point2f& imgCenter, Mat& out) {
  Point2f tmp;
  tmp = projectPoint(focal, imgCenter, center);
  circle(out, tmp, 7, Scalar(255, 0, 0), -1);

  drawBoundingCube(front_box, back_box, focal, imgCenter, 3, out);
}

void drawBoundingCube(const vector<Point3f>& front_box,
                      const vector<Point3f>& back_box, const float focal,
                      const Point2f& imgCenter, int line_width, Mat& out) {
  int cols = out.cols;

  if (back_box.size() < 4) {
    cout << "DRAW_FUNCTIONS: error back_box size is " << back_box.size();
    return;
  }
  if (front_box.size() < 4) {
    cout << "DRAW_FUNCTIONS: error front_box size is " << front_box.size();
    return;
  }

  /*********************************************************************************************/
  /*                  Draw back face */
  /*********************************************************************************************/
  line(out, projectPoint(focal, imgCenter, back_box.at(0)),
       projectPoint(focal, imgCenter, back_box.at(1)), Scalar(0, 0, 255),
       line_width);
  line(out, projectPoint(focal, imgCenter, back_box.at(1)),
       projectPoint(focal, imgCenter, back_box.at(2)), Scalar(0, 0, 255),
       line_width);
  line(out, projectPoint(focal, imgCenter, back_box.at(2)),
       projectPoint(focal, imgCenter, back_box.at(3)), Scalar(0, 0, 255),
       line_width);
  line(out, projectPoint(focal, imgCenter, back_box.at(3)),
       projectPoint(focal, imgCenter, back_box.at(0)), Scalar(0, 0, 255),
       line_width);

  /*********************************************************************************************/
  /*                  Draw connecting lines */
  /*********************************************************************************************/
  line(out, projectPoint(focal, imgCenter, front_box.at(0)),
       projectPoint(focal, imgCenter, back_box.at(0)), Scalar(0, 255, 0),
       line_width);
  line(out, projectPoint(focal, imgCenter, front_box.at(1)),
       projectPoint(focal, imgCenter, back_box.at(1)), Scalar(0, 255, 255),
       line_width);
  line(out, projectPoint(focal, imgCenter, front_box.at(2)),
       projectPoint(focal, imgCenter, back_box.at(2)), Scalar(255, 255, 0),
       line_width);
  line(out, projectPoint(focal, imgCenter, front_box.at(3)),
       projectPoint(focal, imgCenter, back_box.at(3)), Scalar(255, 0, 255),
       line_width);

  /*********************************************************************************************/
  /*                  Draw front face */
  /*********************************************************************************************/

  line(out, projectPoint(focal, imgCenter, front_box.at(0)),
       projectPoint(focal, imgCenter, front_box.at(1)), Scalar(255, 0, 0),
       line_width);
  line(out, projectPoint(focal, imgCenter, front_box.at(1)),
       projectPoint(focal, imgCenter, front_box.at(2)), Scalar(255, 0, 0),
       line_width);
  line(out, projectPoint(focal, imgCenter, front_box.at(2)),
       projectPoint(focal, imgCenter, front_box.at(3)), Scalar(255, 0, 0),
       line_width);
  line(out, projectPoint(focal, imgCenter, front_box.at(3)),
       projectPoint(focal, imgCenter, front_box.at(0)), Scalar(255, 0, 0),
       line_width);
}

void drawBoundingCube(const std::vector<Point3f>& front_box,
                      const std::vector<Point3f>& back_box, const float focal,
                      const Point2f& imgCenter, const Scalar& color,
                      int line_width, Mat& out) {
  int cols = out.cols;

  if (back_box.size() < 4) {
    cout << "DRAW_FUNCTIONS: error back_box size is " << back_box.size();
    return;
  }
  if (front_box.size() < 4) {
    cout << "DRAW_FUNCTIONS: error front_box size is " << front_box.size();
    return;
  }

  /****************************************************************************/
  /*                  Draw back face                                          */
  /****************************************************************************/
  line(out, projectPoint(focal, imgCenter, back_box.at(0)),
       projectPoint(focal, imgCenter, back_box.at(1)), color, line_width);
  line(out, projectPoint(focal, imgCenter, back_box.at(1)),
       projectPoint(focal, imgCenter, back_box.at(2)), color, line_width);
  line(out, projectPoint(focal, imgCenter, back_box.at(2)),
       projectPoint(focal, imgCenter, back_box.at(3)), color, line_width);
  line(out, projectPoint(focal, imgCenter, back_box.at(3)),
       projectPoint(focal, imgCenter, back_box.at(0)), color, line_width);

  /****************************************************************************/
  /*                  Draw connecting lines                                   */
  /****************************************************************************/
  line(out, projectPoint(focal, imgCenter, front_box.at(0)),
       projectPoint(focal, imgCenter, back_box.at(0)), color, line_width);
  line(out, projectPoint(focal, imgCenter, front_box.at(1)),
       projectPoint(focal, imgCenter, back_box.at(1)), color, line_width);
  line(out, projectPoint(focal, imgCenter, front_box.at(2)),
       projectPoint(focal, imgCenter, back_box.at(2)), color, line_width);
  line(out, projectPoint(focal, imgCenter, front_box.at(3)),
       projectPoint(focal, imgCenter, back_box.at(3)), color, line_width);

  /****************************************************************************/
  /*                  Draw front face                                         */
  /****************************************************************************/

  line(out, projectPoint(focal, imgCenter, front_box.at(0)),
       projectPoint(focal, imgCenter, front_box.at(1)), color, line_width);
  line(out, projectPoint(focal, imgCenter, front_box.at(1)),
       projectPoint(focal, imgCenter, front_box.at(2)), color, line_width);
  line(out, projectPoint(focal, imgCenter, front_box.at(2)),
       projectPoint(focal, imgCenter, front_box.at(3)), color, line_width);
  line(out, projectPoint(focal, imgCenter, front_box.at(3)),
       projectPoint(focal, imgCenter, front_box.at(0)), color, line_width);
}

void applyColorMap(const Mat& src, Mat& dst) {
  cv::Mat adjMap;
  float min_distance = 400;
  float max_distance = 8000;
  double scale = 255.0 / double(max_distance - min_distance);
  src.convertTo(adjMap, CV_8UC1, scale, min_distance / 100);
  cv::applyColorMap(adjMap, dst, cv::COLORMAP_JET);
}

void drawObjectLocation(const vector<Point3f>& back_box,
                        const vector<Point3f>& front_box, const Point3f& center,
                        const vector<bool>& visibleFaces, const float focal,
                        const Point2f& imgCenter, Mat& out) {
  int cols = out.cols / 2;

  int lineWidth = 1;
  Scalar color(255, 255, 255);

  if (visibleFaces.size() < 6) {
    cout << "DRAW_FUNCTIONS: error visible faces size";
    return;
  }
  if (back_box.size() < 4) {
    cout << "DRAW_FUNCTIONS: error back_box size";
    return;
  }
  if (front_box.size() < 4) {
    cout << "DRAW_FUNCTIONS: error front_box size";
    return;
  }
  /*********************************************************************************************/
  /*                  Draw back face */
  /*********************************************************************************************/
  if (visibleFaces.at(FACE::BACK)) {
    color = Scalar(0, 0, 255);
    line(out, projectPoint(focal, imgCenter, back_box[0]),
         projectPoint(focal, imgCenter, back_box[2]), color, lineWidth);
  } else {
    color = Scalar(255, 255, 255);
  }

  line(out, projectPoint(focal, imgCenter, back_box[0]),
       projectPoint(focal, imgCenter, back_box[1]), color, lineWidth);
  line(out, projectPoint(focal, imgCenter, back_box[1]),
       projectPoint(focal, imgCenter, back_box[2]), color, lineWidth);
  line(out, projectPoint(focal, imgCenter, back_box[2]),
       projectPoint(focal, imgCenter, back_box[3]), color, lineWidth);
  line(out, projectPoint(focal, imgCenter, back_box[3]),
       projectPoint(focal, imgCenter, back_box[0]), color, lineWidth);

  /*********************************************************************************************/
  /*                  Draw right face */
  /*********************************************************************************************/
  if (visibleFaces[FACE::RIGHT]) {
    color = Scalar(0, 255, 255);
    line(out, projectPoint(focal, imgCenter, front_box[0]),
         projectPoint(focal, imgCenter, back_box[3]), color, lineWidth);
  } else {
    color = Scalar(255, 255, 255);
  }

  line(out, projectPoint(focal, imgCenter, front_box[0]),
       projectPoint(focal, imgCenter, back_box[0]), color, lineWidth);
  line(out, projectPoint(focal, imgCenter, back_box[0]),
       projectPoint(focal, imgCenter, back_box[3]), color, lineWidth);
  line(out, projectPoint(focal, imgCenter, back_box[3]),
       projectPoint(focal, imgCenter, front_box[3]), color, lineWidth);
  line(out, projectPoint(focal, imgCenter, front_box[3]),
       projectPoint(focal, imgCenter, front_box[0]), color, lineWidth);

  /*********************************************************************************************/
  /*                  Draw left face */
  /*********************************************************************************************/
  if (visibleFaces[FACE::LEFT]) {
    color = Scalar(0, 255, 0);
    line(out, projectPoint(focal, imgCenter, front_box[1]),
         projectPoint(focal, imgCenter, back_box[2]), color, lineWidth);
  } else {
    color = Scalar(255, 255, 255);
  }

  line(out, projectPoint(focal, imgCenter, front_box[1]),
       projectPoint(focal, imgCenter, back_box[1]), color, lineWidth);
  line(out, projectPoint(focal, imgCenter, back_box[1]),
       projectPoint(focal, imgCenter, back_box[2]), color, lineWidth);
  line(out, projectPoint(focal, imgCenter, back_box[2]),
       projectPoint(focal, imgCenter, front_box[2]), color, lineWidth);
  line(out, projectPoint(focal, imgCenter, front_box[2]),
       projectPoint(focal, imgCenter, front_box[1]), color, lineWidth);

  /*********************************************************************************************/
  /*                  Draw top face */
  /*********************************************************************************************/
  if (visibleFaces[FACE::DOWN]) {
    color = Scalar(255, 255, 0);
    line(out, projectPoint(focal, imgCenter, front_box[0]),
         projectPoint(focal, imgCenter, back_box[1]), color, lineWidth);
  } else {
    color = Scalar(255, 255, 255);
  }

  line(out, projectPoint(focal, imgCenter, front_box[0]),
       projectPoint(focal, imgCenter, back_box[0]), color, lineWidth);
  line(out, projectPoint(focal, imgCenter, back_box[0]),
       projectPoint(focal, imgCenter, back_box[1]), color, lineWidth);
  line(out, projectPoint(focal, imgCenter, back_box[1]),
       projectPoint(focal, imgCenter, front_box[1]), color, lineWidth);
  line(out, projectPoint(focal, imgCenter, front_box[1]),
       projectPoint(focal, imgCenter, front_box[0]), color, lineWidth);

  /*********************************************************************************************/
  /*                  Draw bottom face */
  /*********************************************************************************************/
  if (visibleFaces[FACE::TOP]) {
    color = Scalar(255, 0, 255);
    line(out, projectPoint(focal, imgCenter, front_box[3]),
         projectPoint(focal, imgCenter, back_box[2]), color, lineWidth);
  } else {
    color = Scalar(0, 255, 255);
  }

  line(out, projectPoint(focal, imgCenter, front_box[3]),
       projectPoint(focal, imgCenter, back_box[3]), color, lineWidth);
  line(out, projectPoint(focal, imgCenter, back_box[3]),
       projectPoint(focal, imgCenter, back_box[2]), color, lineWidth);
  line(out, projectPoint(focal, imgCenter, back_box[2]),
       projectPoint(focal, imgCenter, front_box[2]), color, lineWidth);
  line(out, projectPoint(focal, imgCenter, front_box[2]),
       projectPoint(focal, imgCenter, front_box[3]), color, lineWidth);

  /*********************************************************************************************/
  /*                  Draw front face */
  /*********************************************************************************************/
  if (visibleFaces[FACE::FRONT]) {
    color = Scalar(255, 0, 0);
    line(out, projectPoint(focal, imgCenter, front_box[0]),
         projectPoint(focal, imgCenter, front_box[2]), color, lineWidth);
  } else {
    color = Scalar(255, 255, 255);
  }

  Point2f tmp = projectPoint(focal, imgCenter, center);
  circle(out, tmp, 7, Scalar(255, 0, 0), -1);
  line(out, projectPoint(focal, imgCenter, front_box[0]),
       projectPoint(focal, imgCenter, front_box[1]), color, lineWidth);
  line(out, projectPoint(focal, imgCenter, front_box[1]),
       projectPoint(focal, imgCenter, front_box[2]), color, lineWidth);
  line(out, projectPoint(focal, imgCenter, front_box[2]),
       projectPoint(focal, imgCenter, front_box[3]), color, lineWidth);
  line(out, projectPoint(focal, imgCenter, front_box[3]),
       projectPoint(focal, imgCenter, front_box[0]), color, lineWidth);
}

void drawCentroidVotes(const vector<Point3f*>& points,
                       const vector<Point3f*>& votes, const cv::Point2f& center,
                       bool drawLines, const float focal, Mat& out) {
  for (size_t i = 0; i < points.size(); i++) {
    Point2f vote_prj, point_prj;
    projectPoint(focal, center, votes.at(i), vote_prj);
    projectPoint(focal, center, points.at(i), point_prj);
    Scalar color(0, 255, 0);
    Scalar color2(0, 102, 255);
    circle(out, vote_prj, 2, color, -1);
    circle(out, point_prj, 3, color, 1);
    if (drawLines) line(out, vote_prj, point_prj, color2, 1);
  }
}

void drawObjectPose(const Point3f& centroid, const float focal,
                    const Point2f& img_center, const Mat& rotation, Mat& out) {
  auto rotatePoint = [](const Mat& rotation, cv::Point3f& pt) {
    Mat a(1, 3, CV_32FC1);
    a.at<float>(0) = pt.x;
    a.at<float>(1) = pt.y;
    a.at<float>(2) = pt.z;
    Mat b, b_T, a_T;

    transpose(a, a_T);
    b_T = rotation * a_T;
    transpose(b_T, b);

    pt.x = b.at<float>(0);
    pt.y = b.at<float>(1);
    pt.z = b.at<float>(2);
  };

  Point3f x_axis(0.06, 0, 0);
  Point3f y_axis(0, 0.06, 0);
  Point3f z_axis(0, 0, 0.06);
  // x_axis = y_axis = z_axis = centroid;

  // x_axis.x = 0.02 - x_axis.x;
  // y_axis.y += 0.02;
  // z_axis.z -= 0.02;

  rotatePoint(rotation, x_axis);
  x_axis = centroid + x_axis;
  rotatePoint(rotation, y_axis);
  y_axis = centroid + y_axis;
  rotatePoint(rotation, z_axis);
  z_axis = centroid + z_axis;

  Point2f center = projectPoint(focal, img_center, centroid);

  Point2f xa = projectPoint(focal, img_center, x_axis);
  Point2f ya = projectPoint(focal, img_center, y_axis);
  Point2f za = projectPoint(focal, img_center, z_axis);

  arrowedLine(out, center, xa, Scalar(0, 0, 255), 3);
  arrowedLine(out, center, ya, Scalar(0, 255, 0), 3);
  arrowedLine(out, center, za, Scalar(255, 0, 0), 3);
}

//BUG: cannot conver matrix to double if it is const!!!
void drawObjectPose(const Point3f& centroid, Mat& camera_matrix,
                    const Mat& rotation, const Mat& translation, Mat& out) {
  Point3f c = centroid;
  vector<Point3f> points;
  Point3f x_axis(0.06, 0, 0);
  Point3f y_axis(0, 0.06, 0);
  Point3f z_axis(0, 0, 0.06);
  points.push_back(c);
  points.push_back(c + x_axis);
  points.push_back(c + y_axis);
  points.push_back(c + z_axis);

  rotation.convertTo(rotation, CV_64F);

  vector<Point2f> projected_points;
  projectPoints(points, rotation, translation, camera_matrix, Mat(),
                projected_points);

  arrowedLine(out, projected_points[0], projected_points[1], Scalar(0, 0, 255),
              3);
  arrowedLine(out, projected_points[0], projected_points[2], Scalar(0, 255, 0),
              3);
  arrowedLine(out, projected_points[0], projected_points[3], Scalar(255, 0, 0),
              3);
    }

void drawObjectPose(const Point3f& centroid, Mat& camera_matrix,
                    const Mat& rotation, const Mat& translation,
                    std::vector<cv::Scalar>& axis_colors, Mat& out) {
  Point3f c = centroid;
  vector<Point3f> points;
  Point3f x_axis(0.06, 0, 0);
  Point3f y_axis(0, 0.06, 0);
  Point3f z_axis(0, 0, 0.06);
  points.push_back(c);
  points.push_back(c + x_axis);
  points.push_back(c + y_axis);
  points.push_back(c + z_axis);

  rotation.convertTo(rotation, CV_64F);

  vector<Point2f> projected_points;
  projectPoints(points, rotation, translation, camera_matrix, Mat(),
                projected_points);

  arrowedLine(out, projected_points[0], projected_points[1], axis_colors.at(0),
              3);
  arrowedLine(out, projected_points[0], projected_points[2], axis_colors.at(1),
              3);
  arrowedLine(out, projected_points[0], projected_points[3], axis_colors.at(2),
              3);
}


void arrowedLine(Mat& img, Point2f pt1, Point2f pt2, const Scalar& color,
                 int thickness, int line_type, int shift, double tipLength) {
  const double tipSize =
      norm(pt1 - pt2) * tipLength;  // Factor to normalize the size of the tip
                                    // depending on the length of the arrow
  line(img, pt1, pt2, color, thickness, line_type, shift);
  const double angle = atan2((double)pt1.y - pt2.y, (double)pt1.x - pt2.x);
  Point p(cvRound(pt2.x + tipSize * cos(angle + CV_PI / 4)),
          cvRound(pt2.y + tipSize * sin(angle + CV_PI / 4)));
  line(img, p, pt2, color, thickness, line_type, shift);
  p.x = cvRound(pt2.x + tipSize * cos(angle - CV_PI / 4));
  p.y = cvRound(pt2.y + tipSize * sin(angle - CV_PI / 4));
  line(img, p, pt2, color, thickness, line_type, shift);
}

void cross(Mat& img, Point2f center, const Scalar& color, int thickness,
           int line_offset, int line_type, int shift, double tipLength) {
  Point2f tp, bp, lp, rp;
  tp = bp = lp = rp = center;

  tp.y -= line_offset;
  (tp.y < 0) ? tp.y = 0 : tp.y;

  bp.y += line_offset;
  (bp.y > img.rows) ? bp.y = img.rows - 1 : bp.y;

  lp.x -= line_offset;
  (lp.x < 0) ? lp.x = 0 : lp.x;

  rp.x += line_offset;
  (rp.x > img.cols) ? rp.x = img.cols - 1 : rp.y;

  line(img, tp, bp, color, thickness, line_type, shift);
  line(img, lp, rp, color, thickness, line_type, shift);
}

void drawKeypoints(const std::vector<KeyPoint> &points, Mat &out)
{

    float max_response = 0;
    float min_response = numeric_limits<float>::max();

    for(auto pt : points)
    {
        cv::circle(out, pt.pt, pt.size, cv::Scalar(0,255,0), 1);
    }

}

}  // end namespace
