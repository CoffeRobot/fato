#include "../include/draw_functions.h"
#include "../include/utilities.h"
#include <opencv2/contrib/contrib.hpp>
#include "../include/constants.h"
#include <iostream>

using namespace std;
using namespace cv;

namespace pinot_tracker {

void drawBoundingBox(const vector<Point2f>& box, Scalar color, int line_width,
                     Mat& out) {
  line(out, box[0], box[1], color, line_width, 1);
  line(out, box[1], box[2], color, line_width, 1);
  line(out, box[2], box[3], color, line_width, 1);
  line(out, box[3], box[0], color, line_width, 1);
}


void drawBoundingCube(const Point3f& center, const vector<Point3f>& front_box,
                      const vector<Point3f>& back_box, const float focal,
                      const Point2f& imgCenter, Mat& out)
{
    Point2f tmp;
    tmp = projectPoint(focal, imgCenter, center);
    circle(out, tmp, 7, Scalar(255, 0, 0), -1);

    drawBoundingCube(front_box, back_box, focal, imgCenter, out);
}

void drawBoundingCube(const vector<Point3f>& front_box,
                      const vector<Point3f>& back_box, const float focal,
                      const Point2f& imgCenter, Mat& out) {
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
       projectPoint(focal, imgCenter, back_box.at(1)), Scalar(0, 0, 255), 3);
  line(out, projectPoint(focal, imgCenter, back_box.at(1)),
       projectPoint(focal, imgCenter, back_box.at(2)), Scalar(0, 0, 255), 3);
  line(out, projectPoint(focal, imgCenter, back_box.at(2)),
       projectPoint(focal, imgCenter, back_box.at(3)), Scalar(0, 0, 255), 3);
  line(out, projectPoint(focal, imgCenter, back_box.at(3)),
       projectPoint(focal, imgCenter, back_box.at(0)), Scalar(0, 0, 255), 3);

  /*********************************************************************************************/
  /*                  Draw connecting lines */
  /*********************************************************************************************/
  line(out, projectPoint(focal, imgCenter, front_box.at(0)),
       projectPoint(focal, imgCenter, back_box.at(0)), Scalar(0, 255, 0), 3);
  line(out, projectPoint(focal, imgCenter, front_box.at(1)),
       projectPoint(focal, imgCenter, back_box.at(1)), Scalar(0, 255, 255), 3);
  line(out, projectPoint(focal, imgCenter, front_box.at(2)),
       projectPoint(focal, imgCenter, back_box.at(2)), Scalar(255, 255, 0), 3);
  line(out, projectPoint(focal, imgCenter, front_box.at(3)),
       projectPoint(focal, imgCenter, back_box.at(3)), Scalar(255, 0, 255), 3);

  /*********************************************************************************************/
  /*                  Draw front face */
  /*********************************************************************************************/


  line(out, projectPoint(focal, imgCenter, front_box.at(0)),
       projectPoint(focal, imgCenter, front_box.at(1)), Scalar(255, 0, 0), 3);
  line(out, projectPoint(focal, imgCenter, front_box.at(1)),
       projectPoint(focal, imgCenter, front_box.at(2)), Scalar(255, 0, 0), 3);
  line(out, projectPoint(focal, imgCenter, front_box.at(2)),
       projectPoint(focal, imgCenter, front_box.at(3)), Scalar(255, 0, 0), 3);
  line(out, projectPoint(focal, imgCenter, front_box.at(3)),
       projectPoint(focal, imgCenter, front_box.at(0)), Scalar(255, 0, 0), 3);
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
    Scalar color(0,255,0);
    circle(out, vote_prj, 2, color, -1);
    circle(out, point_prj, 3, color, 1);
    if (drawLines)
         line(out, vote_prj, point_prj, color, 1);
  }
}

}  // end namespace
