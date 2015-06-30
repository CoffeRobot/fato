#include "DebugFunctions.h"
#include "Constants.h"
#include <sstream>

void cross(const cv::Point2f& p, const Scalar& c, int width, cv::Mat& out) {
  line(out, Point2f(p.x, p.y - 2), Point2f(p.x, p.y + 2), c, width);
  line(out, Point2f(p.x - 2, p.y), Point2f(p.x + 2, p.y), c, width);
}

void drawCentroidVotes(const vector<KeyPoint>& keypoints,
                       vector<Point2f>& points, const vector<bool>& clustered,
                       const vector<bool>& border, const vector<Status>& status,
                       bool drawLines, bool drawFalse, Mat& out) {
  // std::cout << "\n";
  // draw each cluster with a separate color but the good one in green color
  random_device rd;
  default_random_engine engine(rd());
  uniform_int_distribution<unsigned int> uniform_dist(0, 255);

  for (size_t i = 0; i < clustered.size(); i++) {
    Scalar color;

    bool valid = (status[i] == Status::BOTH || status[i] == Status::TRACK ||
                  status[i] == Status::MATCH);

    if (clustered[i] && valid) {
      if (border[i])
        color = Scalar(0, 255, 255);
      else
        color = Scalar(0, 255, 0);

      circle(out, points[i], 3, color, 1);

      if (drawLines) {
        line(out, points[i], keypoints[i].pt, color, 1);
      }
    }

    if (drawFalse && status[i] == Status::LOST) {
      if (clustered[i] && !valid) {
        color = Scalar(255, 0, 255);
        circle(out, points[i], 3, color, 1);

        if (drawLines) {
          line(out, points[i], keypoints[i].pt, color, 1);
        }
      } else {
        color = Scalar(0, 0, 255);
        circle(out, points[i], 3, color, 1);

        if (drawLines) {
          line(out, points[i], keypoints[i].pt, color, 1);
        }
      }
    }
  }
}

void drawCentroidVotes(const vector<Point3f>& keypoints,
                       vector<Point3f>& points, const vector<bool>& clustered,
                       const vector<bool>& border, const vector<Status>& status,
                       bool drawLines, bool drawFalse, const float focal,
                       const Point2f& center, ofstream& file, Mat& out) {
  for (int i = 0; i < clustered.size(); i++) {
    Scalar color;

    bool valid = (status[i] == Status::BOTH || status[i] == Status::TRACK ||
                  status[i] == Status::MATCH);

    if (clustered[i] && valid) {
      Point2f tmp = reprojectPoint(focal, center, points[i]);
      Point2f kp = reprojectPoint(focal, center, keypoints[i]);

      if (border[i])
        color = Scalar(0, 255, 255);
      else
        color = Scalar(0, 255, 0);

      circle(out, tmp, 3, color, 1);

      if (drawLines) {
        line(out, tmp, kp, color, 1);
      }

      file << i << " : " << toString(keypoints[i]) << " " << toString(points[i])
           << " " << toString(kp) << " " << toString(tmp) << "\n";
    }
    if (!clustered[i] && valid && points[i].z != 0) {
      Point2f tmp = reprojectPoint(focal, center, points[i]);
      Point2f kp = reprojectPoint(focal, center, keypoints[i]);

      color = Scalar(0, 0, 255);

      circle(out, tmp, 3, color, 1);

      if (drawLines) {
        line(out, tmp, kp, color, 1);
      }

      file << i << " : !! " << toString(keypoints[i]) << " "
           << toString(points[i]) << " " << toString(kp) << " " << toString(tmp)
           << "\n";
    }
  }
}

void drawCentroidVotes(const vector<Point3f>& keypoints,
                       vector<Point3f>& points, const vector<bool>& clustered,
                       const vector<bool>& border, const vector<Status>& status,
                       bool drawLines, bool drawFalse, const float focal,
                       const Point2f& center, Mat& out) {
  for (int i = 0; i < clustered.size(); i++) {
    Scalar color;

    bool valid = (status[i] == Status::BOTH || status[i] == Status::TRACK ||
                  status[i] == Status::MATCH);

    if (clustered[i] && valid) {
      Point2f tmp = reprojectPoint(focal, center, points[i]);
      Point2f kp = reprojectPoint(focal, center, keypoints[i]);

      if (border[i])
        color = Scalar(0, 255, 255);
      else
        color = Scalar(0, 255, 0);

      circle(out, tmp, 3, color, 1);

      if (drawLines) {
        line(out, tmp, kp, color, 1);
      }
    }
    if (!clustered[i] && valid && points[i].z != 0) {
      Point2f tmp = reprojectPoint(focal, center, points[i]);
      Point2f kp = reprojectPoint(focal, center, keypoints[i]);

      color = Scalar(0, 0, 255);

      circle(out, tmp, 3, color, 1);

      if (drawLines) {
        line(out, tmp, kp, color, 1);
      }
    }
  }
}

void drawCentroidVotes(const vector<Point3f*>& keypoints,
                       const vector<Point3f>& votes,
                       const vector<bool>& clustered,
                       const vector<bool>& border,
                       const vector<Status*>& status, bool drawLines,
                       bool drawFalse, const float focal, const Point2f& center,
                       Mat& out) {
  for (size_t i = 0; i < clustered.size(); i++) {
    Scalar color;

    bool valid = (*status[i] == Status::BOTH || *status[i] == Status::TRACK ||
                  *status[i] == Status::MATCH);

    if (clustered[i] && valid) {

      Point2f tmp = reprojectPoint(focal, center, votes[i]);
      Point2f kp = reprojectPoint(focal, center, *keypoints[i]);

      if (border[i])
        color = Scalar(0, 255, 255);
      else
        color = Scalar(0, 255, 0);

      circle(out, tmp, 3, color, 1);

      if (drawLines) {
        line(out, tmp, kp, color, 1);
      }
    }
    if (!clustered[i] && valid && votes[i].z != 0) {
      Point2f tmp = reprojectPoint(focal, center, votes[i]);
      Point2f kp = reprojectPoint(focal, center, *keypoints[i]);

      color = Scalar(0, 0, 255);

      circle(out, tmp, 3, color, 1);

      if (drawLines) {
        line(out, tmp, kp, color, 1);
      }
    }
    if (*status[i] == Status::NOCLUSTER) {
      Point2f tmp = reprojectPoint(focal, center, votes[i]);
      Point2f kp = reprojectPoint(focal, center, *keypoints[i]);

      color = Scalar(255, 0, 255);

      circle(out, tmp, 3, color, 1);

      if (drawLines) {
        line(out, tmp, kp, color, 1);
      }
    }
  }
}

void drawCentroidVotes(const vector<Point3f*>& keypoints,
                       const vector<Point3f>& votes,
                       const vector<bool>& clustered,
                       const vector<bool>& border,
                       const vector<Status*>& status, bool drawLines,
                       bool drawFalse, const float focal, const Point2f& center,
                       ofstream& file, Mat& out) {
  for (size_t i = 0; i < clustered.size(); i++) {
    Scalar color;

    bool valid = (*status[i] == Status::BOTH || *status[i] == Status::TRACK ||
                  *status[i] == Status::MATCH);

    if (clustered[i] && valid && keypoints[i]->z != 0) {

      Point2f tmp = reprojectPoint(focal, center, votes[i]);
      Point2f kp = reprojectPoint(focal, center, *keypoints[i]);

      if (border[i])
        color = Scalar(0, 255, 255);
      else
        color = Scalar(0, 255, 0);

      circle(out, tmp, 3, color, 1);

      if (drawLines) {
        line(out, tmp, kp, color, 1);
      }

      file << i << " :  " << toString(*keypoints[i]) << " "
           << toString(votes[i]) << " " << toString(kp) << " " << toString(tmp)
           << "\n";
    }
    if (!clustered[i] && valid && keypoints[i]->z != 0) {
      Point2f tmp = reprojectPoint(focal, center, votes[i]);
      Point2f kp = reprojectPoint(focal, center, *keypoints[i]);

      color = Scalar(0, 0, 255);

      circle(out, tmp, 3, color, 1);

      if (drawLines) {
        line(out, tmp, kp, color, 1);
      }

      file << i << " : !! " << toString(*keypoints[i]) << " "
           << toString(votes[i]) << " " << toString(kp) << " " << toString(tmp)
           << "\n";
    }
    if (*status[i] == Status::NOCLUSTER && keypoints[i]->z != 0) {
      Point2f tmp = reprojectPoint(focal, center, votes[i]);
      Point2f kp = reprojectPoint(focal, center, *keypoints[i]);

      color = Scalar(255, 0, 255);

      circle(out, tmp, 3, color, 1);

      if (drawLines) {
        line(out, tmp, kp, color, 1);
      }

      file << i << " : !!!! " << toString(*keypoints[i]) << " "
           << toString(votes[i]) << " " << toString(kp) << " " << toString(tmp)
           << "\n";
    }
  }
}

void buildCompositeImg(const Mat& fst, const Mat& scd, Mat& out) {

  unsigned int cols = scd.cols;
  unsigned int rows = scd.rows;

  Size size(cols * 2, rows);

  Mat tmp1, tmp2;
  fst.copyTo(tmp1);
  scd.copyTo(tmp2);

  if (fst.channels() == 1) cvtColor(fst, tmp1, CV_GRAY2BGR);
  if (scd.channels() == 1) cvtColor(scd, tmp2, CV_GRAY2BGR);

  out.create(size, tmp2.type());

  tmp1.copyTo(out(Rect(0, 0, cols, rows)));
  tmp2.copyTo(out(Rect(cols, 0, cols, rows)));

  if (out.channels() == 1) cvtColor(out, out, CV_GRAY2BGR);
}

void drawObjectLocation(const Point2f& fstC, const vector<Point2f>& fstBBox,
                        const Point2f& scdC, const vector<Point2f>& scdBBox,
                        Mat& out) {

  int cols = out.cols / 2;

  Point2f tmp = fstC;
  tmp.x += cols;

  Point2f offset(cols, 0);

  circle(out, tmp, 5, Scalar(255, 0, 0), -1);
  line(out, fstBBox[0] + offset, fstBBox[1] + offset, Scalar(255, 0, 0), 3);
  line(out, fstBBox[1] + offset, fstBBox[2] + offset, Scalar(255, 0, 0), 3);
  line(out, fstBBox[2] + offset, fstBBox[3] + offset, Scalar(255, 0, 0), 3);
  line(out, fstBBox[3] + offset, fstBBox[0] + offset, Scalar(255, 0, 0), 3);

  circle(out, scdC, 5, Scalar(255, 0, 0), -1);
  line(out, scdBBox[0], scdBBox[1], Scalar(255, 0, 0), 3);
  line(out, scdBBox[1], scdBBox[2], Scalar(255, 0, 0), 3);
  line(out, scdBBox[2], scdBBox[3], Scalar(255, 0, 0), 3);
  line(out, scdBBox[3], scdBBox[0], Scalar(255, 0, 0), 3);
}

void drawObjectLocation(const Point3f& fstC, const vector<Point3f>& fstBBox,
                        const Point3f& scdC, const vector<Point3f>& scdBBox,
                        const float focal, const Point2f& center, Mat& out) {

  int cols = out.cols / 2;

  Point2f tmp = reprojectPoint(focal, center, fstC);
  tmp.x += cols;

  Point2f offset(cols, 0);

  circle(out, tmp, 7, Scalar(255, 0, 0), -1);
  line(out, reprojectPoint(focal, center, fstBBox[0]) + offset,
       reprojectPoint(focal, center, fstBBox[1]) + offset, Scalar(255, 0, 0),
       3);
  line(out, reprojectPoint(focal, center, fstBBox[1]) + offset,
       reprojectPoint(focal, center, fstBBox[2]) + offset, Scalar(255, 0, 0),
       3);
  line(out, reprojectPoint(focal, center, fstBBox[2]) + offset,
       reprojectPoint(focal, center, fstBBox[3]) + offset, Scalar(255, 0, 0),
       3);
  line(out, reprojectPoint(focal, center, fstBBox[3]) + offset,
       reprojectPoint(focal, center, fstBBox[0]) + offset, Scalar(255, 0, 0),
       3);

  tmp = reprojectPoint(focal, center, scdC);
  circle(out, tmp, 7, Scalar(255, 0, 0), -1);
  line(out, reprojectPoint(focal, center, scdBBox[0]),
       reprojectPoint(focal, center, scdBBox[1]), Scalar(255, 0, 0), 3);
  line(out, reprojectPoint(focal, center, scdBBox[1]),
       reprojectPoint(focal, center, scdBBox[2]), Scalar(255, 0, 0), 3);
  line(out, reprojectPoint(focal, center, scdBBox[2]),
       reprojectPoint(focal, center, scdBBox[3]), Scalar(255, 0, 0), 3);
  line(out, reprojectPoint(focal, center, scdBBox[3]),
       reprojectPoint(focal, center, scdBBox[0]), Scalar(255, 0, 0), 3);
}

void drawObjectLocation(const BorgCube& fstCube, const BorgCube& updCube,
                        const vector<bool>& visibleFaces, const float focal,
                        const Point2f& imgCenter, Mat& out) {
  int cols = out.cols / 2;

  /*********************************************************************************************/
  /*                  Draw original bounding box */
  /*********************************************************************************************/
  Point2f tmp = reprojectPoint(focal, imgCenter, fstCube.m_center);
  tmp.x += cols;
  Point2f offset(cols, 0);

  const vector<Point3f>& fstBBox = fstCube.m_pointsFront;

  circle(out, tmp, 7, Scalar(255, 0, 0), -1);
  line(out, reprojectPoint(focal, imgCenter, fstBBox[0]) + offset,
       reprojectPoint(focal, imgCenter, fstBBox[1]) + offset, Scalar(255, 0, 0),
       1);
  line(out, reprojectPoint(focal, imgCenter, fstBBox[1]) + offset,
       reprojectPoint(focal, imgCenter, fstBBox[2]) + offset, Scalar(255, 0, 0),
       1);
  line(out, reprojectPoint(focal, imgCenter, fstBBox[2]) + offset,
       reprojectPoint(focal, imgCenter, fstBBox[3]) + offset, Scalar(255, 0, 0),
       1);
  line(out, reprojectPoint(focal, imgCenter, fstBBox[3]) + offset,
       reprojectPoint(focal, imgCenter, fstBBox[0]) + offset, Scalar(255, 0, 0),
       1);

  const vector<Point3f>& scdBackBox = updCube.m_pointsBack;
  const vector<Point3f>& scdFrontBox = updCube.m_pointsFront;
  int lineWidth = 1;
  Scalar color(255, 255, 255);
  /*********************************************************************************************/
  /*                  Draw back face */
  /*********************************************************************************************/
  if (visibleFaces[FACE::BACK]) {
    lineWidth = 1;
    color = Scalar(0, 0, 255);
    line(out, reprojectPoint(focal, imgCenter, scdBackBox[0]),
         reprojectPoint(focal, imgCenter, scdBackBox[2]), color, lineWidth);
  } else {
    lineWidth = 1;
    color = Scalar(255, 255, 255);
  }

  line(out, reprojectPoint(focal, imgCenter, scdBackBox[0]),
       reprojectPoint(focal, imgCenter, scdBackBox[1]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[1]),
       reprojectPoint(focal, imgCenter, scdBackBox[2]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[2]),
       reprojectPoint(focal, imgCenter, scdBackBox[3]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[3]),
       reprojectPoint(focal, imgCenter, scdBackBox[0]), color, lineWidth);

  /*********************************************************************************************/
  /*                  Draw right face */
  /*********************************************************************************************/
  if (visibleFaces[FACE::RIGHT]) {
    lineWidth = 1;
    color = Scalar(255, 0, 255);
    line(out, reprojectPoint(focal, imgCenter, scdFrontBox[0]),
         reprojectPoint(focal, imgCenter, scdBackBox[3]), color, lineWidth);
  } else {
    lineWidth = 1;
    color = Scalar(255, 255, 255);
  }

  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[0]),
       reprojectPoint(focal, imgCenter, scdBackBox[0]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[0]),
       reprojectPoint(focal, imgCenter, scdBackBox[3]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[3]),
       reprojectPoint(focal, imgCenter, scdFrontBox[3]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[3]),
       reprojectPoint(focal, imgCenter, scdFrontBox[0]), color, lineWidth);

  /*********************************************************************************************/
  /*                  Draw left face */
  /*********************************************************************************************/
  if (visibleFaces[FACE::LEFT]) {
    lineWidth = 1;
    color = Scalar(0, 255, 0);
    line(out, reprojectPoint(focal, imgCenter, scdFrontBox[1]),
         reprojectPoint(focal, imgCenter, scdBackBox[2]), color, lineWidth);
  } else {
    lineWidth = 1;
    color = Scalar(255, 255, 255);
  }

  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[1]),
       reprojectPoint(focal, imgCenter, scdBackBox[1]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[1]),
       reprojectPoint(focal, imgCenter, scdBackBox[2]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[2]),
       reprojectPoint(focal, imgCenter, scdFrontBox[2]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[2]),
       reprojectPoint(focal, imgCenter, scdFrontBox[1]), color, lineWidth);

  /*********************************************************************************************/
  /*                  Draw top face */
  /*********************************************************************************************/
  if (visibleFaces[FACE::DOWN]) {
    lineWidth = 1;
    color = Scalar(255, 255, 0);
    line(out, reprojectPoint(focal, imgCenter, scdFrontBox[0]),
         reprojectPoint(focal, imgCenter, scdBackBox[1]), color, lineWidth);
  } else {
    lineWidth = 1;
    color = Scalar(255, 255, 255);
  }

  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[0]),
       reprojectPoint(focal, imgCenter, scdBackBox[0]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[0]),
       reprojectPoint(focal, imgCenter, scdBackBox[1]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[1]),
       reprojectPoint(focal, imgCenter, scdFrontBox[1]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[1]),
       reprojectPoint(focal, imgCenter, scdFrontBox[0]), color, lineWidth);

  /*********************************************************************************************/
  /*                  Draw bottom face */
  /*********************************************************************************************/
  if (visibleFaces[FACE::TOP]) {
    lineWidth = 1;
    color = Scalar(0, 255, 255);
    line(out, reprojectPoint(focal, imgCenter, scdFrontBox[3]),
         reprojectPoint(focal, imgCenter, scdBackBox[2]), color, lineWidth);
  } else {
    lineWidth = 1;
    color = Scalar(0, 255, 255);
  }

  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[3]),
       reprojectPoint(focal, imgCenter, scdBackBox[3]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[3]),
       reprojectPoint(focal, imgCenter, scdBackBox[2]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[2]),
       reprojectPoint(focal, imgCenter, scdFrontBox[2]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[2]),
       reprojectPoint(focal, imgCenter, scdFrontBox[3]), color, lineWidth);

  /*********************************************************************************************/
  /*                  Draw front face */
  /*********************************************************************************************/
  if (visibleFaces[FACE::FRONT]) {
    lineWidth = 1;
    color = Scalar(255, 0, 0);
    line(out, reprojectPoint(focal, imgCenter, scdFrontBox[0]),
         reprojectPoint(focal, imgCenter, scdFrontBox[2]), color, lineWidth);
  } else {
    lineWidth = 1;
    color = Scalar(255, 255, 255);
  }

  tmp = reprojectPoint(focal, imgCenter, updCube.m_center);
  circle(out, tmp, 7, Scalar(255, 0, 0), -1);
  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[0]),
       reprojectPoint(focal, imgCenter, scdFrontBox[1]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[1]),
       reprojectPoint(focal, imgCenter, scdFrontBox[2]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[2]),
       reprojectPoint(focal, imgCenter, scdFrontBox[3]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[3]),
       reprojectPoint(focal, imgCenter, scdFrontBox[0]), color, lineWidth);
}

void drawObjectLocation(const BorgCube& updCube,
                        const vector<bool>& visibleFaces, const float focal,
                        const Point2f& imgCenter, Mat& out) {
  int cols = out.cols / 2;

  const vector<Point3f>& scdBackBox = updCube.m_pointsBack;
  const vector<Point3f>& scdFrontBox = updCube.m_pointsFront;
  int lineWidth = 1;
  Scalar color(255, 255, 255);
  /*********************************************************************************************/
  /*                  Draw back face */
  /*********************************************************************************************/
  if (visibleFaces[FACE::BACK]) {
    color = Scalar(0, 0, 255);
    line(out, reprojectPoint(focal, imgCenter, scdBackBox[0]),
         reprojectPoint(focal, imgCenter, scdBackBox[2]), color, lineWidth);
  } else {
    color = Scalar(255, 255, 255);
  }

  line(out, reprojectPoint(focal, imgCenter, scdBackBox[0]),
       reprojectPoint(focal, imgCenter, scdBackBox[1]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[1]),
       reprojectPoint(focal, imgCenter, scdBackBox[2]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[2]),
       reprojectPoint(focal, imgCenter, scdBackBox[3]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[3]),
       reprojectPoint(focal, imgCenter, scdBackBox[0]), color, lineWidth);

  /*********************************************************************************************/
  /*                  Draw right face */
  /*********************************************************************************************/
  if (visibleFaces[FACE::RIGHT]) {
    color = Scalar(0, 255, 255);
    line(out, reprojectPoint(focal, imgCenter, scdFrontBox[0]),
         reprojectPoint(focal, imgCenter, scdBackBox[3]), color, lineWidth);
  } else {
    color = Scalar(255, 255, 255);
  }

  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[0]),
       reprojectPoint(focal, imgCenter, scdBackBox[0]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[0]),
       reprojectPoint(focal, imgCenter, scdBackBox[3]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[3]),
       reprojectPoint(focal, imgCenter, scdFrontBox[3]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[3]),
       reprojectPoint(focal, imgCenter, scdFrontBox[0]), color, lineWidth);

  /*********************************************************************************************/
  /*                  Draw left face */
  /*********************************************************************************************/
  if (visibleFaces[FACE::LEFT]) {
    color = Scalar(0, 255, 0);
    line(out, reprojectPoint(focal, imgCenter, scdFrontBox[1]),
         reprojectPoint(focal, imgCenter, scdBackBox[2]), color, lineWidth);
  } else {
    color = Scalar(255, 255, 255);
  }

  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[1]),
       reprojectPoint(focal, imgCenter, scdBackBox[1]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[1]),
       reprojectPoint(focal, imgCenter, scdBackBox[2]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[2]),
       reprojectPoint(focal, imgCenter, scdFrontBox[2]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[2]),
       reprojectPoint(focal, imgCenter, scdFrontBox[1]), color, lineWidth);

  /*********************************************************************************************/
  /*                  Draw top face */
  /*********************************************************************************************/
  if (visibleFaces[FACE::DOWN]) {
    color = Scalar(255, 255, 0);
    line(out, reprojectPoint(focal, imgCenter, scdFrontBox[0]),
         reprojectPoint(focal, imgCenter, scdBackBox[1]), color, lineWidth);
  } else {
    color = Scalar(255, 255, 255);
  }

  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[0]),
       reprojectPoint(focal, imgCenter, scdBackBox[0]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[0]),
       reprojectPoint(focal, imgCenter, scdBackBox[1]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[1]),
       reprojectPoint(focal, imgCenter, scdFrontBox[1]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[1]),
       reprojectPoint(focal, imgCenter, scdFrontBox[0]), color, lineWidth);

  /*********************************************************************************************/
  /*                  Draw bottom face */
  /*********************************************************************************************/
  if (visibleFaces[FACE::TOP]) {
    color = Scalar(255, 0, 255);
    line(out, reprojectPoint(focal, imgCenter, scdFrontBox[3]),
         reprojectPoint(focal, imgCenter, scdBackBox[2]), color, lineWidth);
  } else {
    color = Scalar(0, 255, 255);
  }

  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[3]),
       reprojectPoint(focal, imgCenter, scdBackBox[3]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[3]),
       reprojectPoint(focal, imgCenter, scdBackBox[2]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[2]),
       reprojectPoint(focal, imgCenter, scdFrontBox[2]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[2]),
       reprojectPoint(focal, imgCenter, scdFrontBox[3]), color, lineWidth);

  /*********************************************************************************************/
  /*                  Draw front face */
  /*********************************************************************************************/
  if (visibleFaces[FACE::FRONT]) {
    color = Scalar(255, 0, 0);
    line(out, reprojectPoint(focal, imgCenter, scdFrontBox[0]),
         reprojectPoint(focal, imgCenter, scdFrontBox[2]), color, lineWidth);
  } else {
    color = Scalar(255, 255, 255);
  }

  Point2f tmp = reprojectPoint(focal, imgCenter, updCube.m_center);
  circle(out, tmp, 7, Scalar(255, 0, 0), -1);
  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[0]),
       reprojectPoint(focal, imgCenter, scdFrontBox[1]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[1]),
       reprojectPoint(focal, imgCenter, scdFrontBox[2]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[2]),
       reprojectPoint(focal, imgCenter, scdFrontBox[3]), color, lineWidth);
  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[3]),
       reprojectPoint(focal, imgCenter, scdFrontBox[0]), color, lineWidth);
}

void drawBoundingCube(const Point3f& scdC, const vector<Point3f>& scdFrontBox,
                      const vector<Point3f>& scdBackBox, const float focal,
                      const Point2f& imgCenter, Mat& out) {
  int cols = out.cols;
  Point2f tmp;
  /*********************************************************************************************/
  /*                  Draw back face */
  /*********************************************************************************************/
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[0]),
       reprojectPoint(focal, imgCenter, scdBackBox[1]), Scalar(0, 0, 255), 3);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[1]),
       reprojectPoint(focal, imgCenter, scdBackBox[2]), Scalar(0, 0, 255), 3);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[2]),
       reprojectPoint(focal, imgCenter, scdBackBox[3]), Scalar(0, 0, 255), 3);
  line(out, reprojectPoint(focal, imgCenter, scdBackBox[3]),
       reprojectPoint(focal, imgCenter, scdBackBox[0]), Scalar(0, 0, 255), 3);

  /*********************************************************************************************/
  /*                  Draw connecting lines */
  /*********************************************************************************************/
  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[0]),
       reprojectPoint(focal, imgCenter, scdBackBox[0]), Scalar(0, 255, 0), 3);
  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[1]),
       reprojectPoint(focal, imgCenter, scdBackBox[1]), Scalar(0, 255, 255), 3);
  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[2]),
       reprojectPoint(focal, imgCenter, scdBackBox[2]), Scalar(255, 255, 0), 3);
  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[3]),
       reprojectPoint(focal, imgCenter, scdBackBox[3]), Scalar(255, 0, 255), 3);

  /*********************************************************************************************/
  /*                  Draw front face */
  /*********************************************************************************************/

  tmp = reprojectPoint(focal, imgCenter, scdC);
  circle(out, tmp, 7, Scalar(255, 0, 0), -1);
  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[0]),
       reprojectPoint(focal, imgCenter, scdFrontBox[1]), Scalar(255, 0, 0), 3);
  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[1]),
       reprojectPoint(focal, imgCenter, scdFrontBox[2]), Scalar(255, 0, 0), 3);
  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[2]),
       reprojectPoint(focal, imgCenter, scdFrontBox[3]), Scalar(255, 0, 0), 3);
  line(out, reprojectPoint(focal, imgCenter, scdFrontBox[3]),
       reprojectPoint(focal, imgCenter, scdFrontBox[0]), Scalar(255, 0, 0), 3);
}

void drawKeypointsMatching(const vector<KeyPoint>& fstPoint,
                           const vector<KeyPoint>& scdPoints,
                           const vector<Status>& pointStatus,
                           const vector<Scalar>& colors, int& numMatch,
                           int& numTrack, int& numBoth, bool drawLines,
                           Mat& out) {

  int cols = out.cols / 2;

  for (size_t i = 0; i < scdPoints.size(); i++) {
    const Scalar& color = colors[i];

    const Status& s = pointStatus[i];

    const Point2f& fst = fstPoint[i].pt;
    const Point2f& scd = scdPoints[i].pt;

    Point2f fstOffset = fst;
    fstOffset.x += cols;

    if (s == Status::MATCH) {
      circle(out, fstOffset, 3, color, 1);
      circle(out, scd, 3, color, 1);
      if (drawLines) line(out, scd, fstOffset, color, 1);
      numMatch++;
    } else if (s == Status::TRACK) {

      Rect prev(fst.x - 2 + cols, fst.y - 2, 5, 5);
      Rect next(scd.x - 2, scd.y - 2, 5, 5);

      rectangle(out, prev, color, 1);
      rectangle(out, next, color, 1);
      if (drawLines) line(out, scd, fstOffset, color, 1);
      numTrack++;
    } else if (s == Status::BACKGROUND) {
      // circle(out, fstOffset, 3, color, 1);
      // circle(out, scd, 3, color, 1);
    } else if (s == Status::BOTH) {
      cross(fstOffset, color, 1, out);
      cross(scd, color, 1, out);
      if (drawLines) line(out, scd, fstOffset, color, 1);
      numBoth++;
    }
  }
}

void drawPointsMatching(const vector<Point3f>& fstPoints,
                        const vector<Point3f>& scdPoints,
                        const vector<Status>& pointStatus,
                        const vector<Scalar>& colors, int& numMatch,
                        int& numTrack, int& numBoth, bool drawLines,
                        const float focal, const Point2f& center, Mat& out) {
  int cols = out.cols / 2;

  for (size_t i = 0; i < scdPoints.size(); i++) {
    const Scalar& color = colors[i];

    const Status& s = pointStatus[i];

    const Point2f& fst = reprojectPoint(focal, center, fstPoints[i]);
    const Point2f& scd = reprojectPoint(focal, center, scdPoints[i]);

    Point2f fstOffset = fst;
    fstOffset.x += cols;

    if (s == Status::MATCH) {
      circle(out, fstOffset, 3, color, 1);
      circle(out, scd, 3, color, 1);
      if (drawLines) line(out, scd, fstOffset, color, 1);
      numMatch++;
    } else if (s == Status::TRACK) {

      Rect prev(fst.x - 2 + cols, fst.y - 2, 5, 5);
      Rect next(scd.x - 2, scd.y - 2, 5, 5);

      rectangle(out, prev, color, 1);
      rectangle(out, next, color, 1);
      if (drawLines) line(out, scd, fstOffset, color, 1);
      numTrack++;
    } else if (s == Status::BACKGROUND) {
      // circle(out, fstOffset, 3, color, 1);
      // circle(out, scd, 3, color, 1);
    } else if (s == Status::BOTH) {
      cross(fstOffset, color, 1, out);
      cross(scd, color, 1, out);
      if (drawLines) line(out, scd, fstOffset, color, 1);
      numBoth++;
    }
  }
}

void drawPointsMatching(const vector<Point3f*>& fstPoints,
                        const vector<Point3f*>& scdPoints,
                        const vector<Status*>& pointStatus,
                        const vector<Scalar*>& colors, int& numMatch,
                        int& numTrack, int& numBoth, bool drawLines,
                        const float focal, const Point2f& center, Mat& out) {
  int cols = out.cols / 2;
  for (size_t i = 0; i < scdPoints.size(); i++) {
    const Scalar& color = *colors[i];

    const Status& s = *pointStatus[i];

    Point2f fst(0, 0);
    Point2f scd(0, 0);

    if (fstPoints[i]->z == 0 || scdPoints[i]->z == 0) continue;

    bool fstVal = reprojectPoint(focal, center, *fstPoints[i], fst);
    bool scdVal = reprojectPoint(focal, center, *scdPoints[i], scd);

    Point2f fstOffset = fst;
    fstOffset.x += cols;

    if (s == Status::MATCH) {

      circle(out, fstOffset, 3, color, 1);
      circle(out, scd, 3, color, 1);
      if (drawLines) line(out, scd, fstOffset, color, 1);
      numMatch++;
    } else if (s == Status::TRACK) {
      // cout << "Track crash!";
      Rect prev(fst.x - 2 + cols, fst.y - 2, 5, 5);
      Rect next(scd.x - 2, scd.y - 2, 5, 5);

      rectangle(out, prev, color, 1);
      rectangle(out, next, color, 1);
      if (drawLines) line(out, scd, fstOffset, color, 1);
      numTrack++;
      // cout << "Track crash!";
    } else if (s == Status::BOTH) {
      // cout << "Both crash!";
      cross(fstOffset, color, 1, out);
      cross(scd, color, 1, out);
      if (drawLines) line(out, scd, fstOffset, color, 1);
      numBoth++;
      // cout << "Both crash!";
    }
    // cout << "\n";
  }
}

void drawPointsMatchingICRA(const vector<Point3f*>& fstPoints,
                            const vector<Point3f*>& scdPoints,
                            const vector<Status*>& pointStatus,
                            const vector<Scalar*>& colors, int& numMatch,
                            int& numTrack, int& numBoth, bool drawLines,
                            const float focal, const Point2f& center,
                            Mat& out) {
  int cols = out.cols / 2;
  for (size_t i = 0; i < scdPoints.size(); i++) {
    const Scalar& color = *colors[i];

    const Status& s = *pointStatus[i];

    Point2f fst(0, 0);
    Point2f scd(0, 0);

    if (fstPoints[i]->z == 0 || scdPoints[i]->z == 0) continue;

    bool fstVal = reprojectPoint(focal, center, *fstPoints[i], fst);
    bool scdVal = reprojectPoint(focal, center, *scdPoints[i], scd);

    if (s == Status::MATCH) {

      circle(out, scd, 3, color, 1);
      numMatch++;
    } else if (s == Status::TRACK) {
      // cout << "Track crash!";

      Rect next(scd.x - 2, scd.y - 2, 5, 5);

      rectangle(out, next, color, 1);

      numTrack++;
      // cout << "Track crash!";
    } else if (s == Status::BOTH) {
      // cout << "Both crash!";
      cross(scd, color, 1, out);
      numBoth++;
      // cout << "Both crash!";
    }
    // cout << "\n";
  }
}

void countKeypointsMatching(const vector<KeyPoint>& fstPoint,
                            const vector<KeyPoint>& scdPoints,
                            const vector<Status>& pointStatus, int& numMatch,
                            int& numTrack, int& numBoth) {
  for (size_t i = 0; i < scdPoints.size(); i++) {
    const Status& s = pointStatus[i];

    const Point2f& fst = fstPoint[i].pt;
    const Point2f& scd = scdPoints[i].pt;

    Point2f fstOffset = fst;

    if (s == Status::MATCH)
      numMatch++;
    else if (s == Status::TRACK)
      numTrack++;
    else if (s == Status::BOTH)
      numBoth++;
  }
}

void countKeypointsMatching(const vector<Status*>& pointStatus, int& numMatch,
                            int& numTrack, int& numBoth) {
  for (size_t i = 0; i < pointStatus.size(); i++) {
    const Status* s = pointStatus[i];

    if (*s == Status::MATCH)
      numMatch++;
    else if (*s == Status::TRACK)
      numTrack++;
    else if (*s == Status::BOTH)
      numBoth++;
  }
}

void drawKeipointsStats(const int init, const int matched, const int tracked,
                        const int both, Mat& out) {

  int rows = out.rows;
  int cols = out.cols / 2;

  int histWidth = 30;
  int windowHeight = 50;
  int windowY = rows - windowHeight;
  // int windowX = 2 * cols - histWidth * 3;
  int windowX = cols;

  Rect histR(windowX, windowY, 3 * histWidth, windowHeight);

  int nelems = 0, nMatch = 0, nTrack = 0, nboth = 0;
  nelems = both + matched + tracked;

  rectangle(out, histR, Scalar(0, 0, 0), -1);

  if (init > 0) {
    nMatch = (windowHeight * (matched + both)) / init;
    nTrack = (windowHeight * (tracked + both)) / init;
    nboth = (windowHeight * both) / init;

    Rect mR(windowX, windowY + (windowHeight - nMatch), histWidth, nMatch);
    Rect tR(windowX + histWidth, windowY + (windowHeight - nTrack), histWidth,
            nTrack);
    Rect bR(windowX + 2 * histWidth, windowY + (windowHeight - nboth),
            histWidth, nboth);

    rectangle(out, mR, Scalar(255, 0, 0), -1);
    rectangle(out, tR, Scalar(0, 255, 0), -1);
    rectangle(out, bR, Scalar(0, 0, 255), -1);
  }
}

void drawInformationHeader(const int numFrames, const float scale,
                           const float angle, int clusterSize, int matched,
                           int tracked, Mat& out) {
  stringstream ss;
  ss.precision(2);
  ss << "Frame: " << numFrames << " S: " << scale << " A: " << angle
     << " V: " << clusterSize << " M: " << matched << " T: " << tracked;

  rectangle(out, Rect(0, 0, (out.cols / 2), 30), Scalar(0, 0, 0), -1);

  putText(out, ss.str(), Point2f(10, 10), FONT_HERSHEY_PLAIN, 1,
          Scalar(255, 255, 255), 1);

  // imshow("Debug Window", out);
}

void drawInformationHeader(const Point2f& top, const string information,
                           float alpha, int width, int height, Mat& out) {

  for (int i = top.y; i < top.y + height; ++i) {
    for (int j = 0; j < width; j++) {
        out.at<Vec3b>(i, j) = (1 - alpha) * out.at<Vec3b>(i, j);

    }
  }
  // rectangle(out, Rect(0, 0, (out.cols * 0.75), 30), Scalar(0, 0, 0), -1);

  putText(out, information, Point2f(top.x, top.y + 10), FONT_HERSHEY_PLAIN, 1,
          Scalar(255, 255, 255), 1);
}

void drawInformationHeaderICRA(Point2f& top, const string frame,
                               const string angle, const string visibility,
                               float alpha, int width, int height, Mat& out) {
  for (int i = top.x; i < out.rows; ++i) {
    for (int j = top.y; j < out.cols; j++) {
      if (i < height && j < width) {
        out.at<Vec3b>(i, j) = (1 - alpha) * Vec3b(255, 255, 255);
      }
    }
  }
  // rectangle(out, Rect(0, 0, (out.cols * 0.75), 30), Scalar(0, 0, 0), -1);

  putText(out, frame, Point2f(top.x + 10, top.y + 20), FONT_HERSHEY_PLAIN, 1,
          Scalar(255, 255, 255), 1);
  putText(out, angle, Point2f(top.x + 10, top.y + 40), FONT_HERSHEY_PLAIN, 1,
          Scalar(255, 255, 255), 1);
}

void scanLine(const Point2f& a, const Point2f& b, const int minY,
              vector<int>& mins, vector<int>& maxs) {
  int deltaY = abs(a.y - b.y);

  int offset = min(a.y, b.y) - minY;

  const Point2f& minPoint = (a.y < b.y) ? a : b;
  const Point2f& maxPoint = (a.y >= b.y) ? a : b;

  float step = (maxPoint.x - minPoint.x) / static_cast<float>(deltaY);
  float minX = minPoint.x;

  for (size_t i = 0; i < deltaY; i++) {
    int xVal = static_cast<int>(minX + i * step);
    mins[i + offset] = min(mins[i + offset], xVal);
    maxs[i + offset] = max(maxs[i + offset], xVal);
  }
}

void drawTriangle(const Point2f& a, const Point2f& b, const Point2f& c,
                  Scalar color, float alpha, Mat& out) {
  int minX = min(a.x, min(b.x, c.x));
  int maxX = max(a.x, max(b.x, c.x));

  int minY = min(a.y, min(b.y, c.y));
  int maxY = max(a.y, max(b.y, c.y));

  int deltaY = maxY - minY;

  vector<int> minXs(deltaY, numeric_limits<int>::max());
  vector<int> maxXs(deltaY, 0);

  scanLine(a, b, minY, minXs, maxXs);
  scanLine(a, c, minY, minXs, maxXs);
  scanLine(b, c, minY, minXs, maxXs);

  for (size_t i = 0; i < minXs.size(); i++) {
    int y = i + minY;

    for (size_t j = minXs[i]; j <= maxXs[i]; j++) {
      out.at<Vec3b>(y, j) = (1 - alpha) * out.at<Vec3b>(y, j) +
                            alpha * Vec3b(color[0], color[1], color[2]);
    }
  }
}

void drawTriangleMask(const Point2f& a, const Point2f& b, const Point2f& c,
                      Mat1b& out) {
  int minX = min(a.x, min(b.x, c.x));
  int maxX = max(a.x, max(b.x, c.x));

  int minY = min(a.y, min(b.y, c.y));
  int maxY = max(a.y, max(b.y, c.y));

  int deltaY = maxY - minY;

  vector<int> minXs(deltaY, numeric_limits<int>::max());
  vector<int> maxXs(deltaY, 0);

  scanLine(a, b, minY, minXs, maxXs);
  scanLine(a, c, minY, minXs, maxXs);
  scanLine(b, c, minY, minXs, maxXs);

  for (size_t i = 0; i < minXs.size(); i++) {
    int y = i + minY;

    for (size_t j = minXs[i]; j <= maxXs[i]; j++) {
      out.at<uchar>(y, j) = 255;
    }
  }
}

cv::Point2f reprojectPoint(const float focal, const Point2f& center,
                           const cv::Point3f& src) {
  Point2f dst;

  dst.x = (focal * src.x / src.z) + center.x;
  dst.y = (center.y - (focal * src.y / src.z));

  return dst;
}

bool reprojectPoint(const float focal, const Point2f& center,
                    const Point3f& src, Point2f& dst) {

  if (src.z == 0) return false;

  dst.x = (focal * src.x / src.z) + center.x;
  dst.y = (center.y - (focal * src.y / src.z));

  if (dst.x < 0 || dst.x > center.x * 2) return false;
  if (dst.y < 0 || dst.y > center.y * 2) return false;

  if (isnan(dst.x) || isnan(dst.y)) return false;

  return true;
}

std::string toString(const Matrix3d& rotation) {
  stringstream ss;
  ss << "[";
  for (int i = 0; i < rotation.rows(); ++i) {
    ss << "[";
    for (int j = 0; j < rotation.cols(); j++) {
      ss << rotation(i, j);
      if (j < rotation.cols() - 1) ss << ",";
    }
    ss << "]";
    if (i < rotation.rows() - 1) ss << ",";
  }
  ss << "]";

  return ss.str();
}

std::string toString(const Quaterniond& quaternion) {
  stringstream ss;

  ss << "[" << quaternion.w() << "," << quaternion.x() << "," << quaternion.y()
     << "," << quaternion.z() << "]";

  return ss.str();
}

string toPythonString(const Mat& rotation) {
  if (rotation.cols != 3 || rotation.rows != 3) return "";

  stringstream ss;
  ss << "[";
  for (size_t i = 0; i < 3; i++) {
    ss << "[";
    for (size_t j = 0; j < 3; j++) {
      ss << rotation.at<float>(i, j);
      if (j < 2) ss << ",";
    }
    ss << " ]";
    if (i < 2) ss << ",";
  }
  ss << "]";

  return ss.str();
}

string toPythonArray(const Mat& rotation) {
  if (rotation.cols != 3 || rotation.rows != 3) return "";

  stringstream ss;
  ss << "[";
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      ss << rotation.at<float>(i, j);
      if (i != 2 || j != 2) ss << ",";
    }
  }
  ss << "]";

  return ss.str();
}

std::string faceToString(int face) {
  switch (face) {
    case FACE::BACK:
      return "back";
    case FACE::DOWN:
      return "down";
    case FACE::FRONT:
      return "front";
    case FACE::LEFT:
      return "left";
    case FACE::RIGHT:
      return "right";
    case FACE::TOP:
      return "top";
    default:
      return "";
  }
}

std::string toPythonString(const std::vector<cv::Point3f>& cloud) {
  stringstream ss;
  ss << "[";
  for (size_t j = 0; j < cloud.size(); j++) {
    if (cloud[j].z != 0) ss << toString(cloud[j]);
    if (cloud[j].z != 0 && j < cloud.size() - 1) ss << ",";
  }
  ss << "]";

  return ss.str();
}

void drawBoundingBox(const std::vector<cv::Point2f>& box, cv::Scalar& color,
                     Mat& out) {
  line(out, box[0], box[1], color, 1, 1);
  line(out, box[1], box[2], color, 1, 1);
  line(out, box[2], box[3], color, 1, 1);
  line(out, box[3], box[0], color, 1, 1);
}

void drawVotes(const std::vector<Point2f>& votes, cv::Scalar& color, Mat& out) {
  for (size_t i = 0; i < votes.size(); i++) {
    circle(out, votes[i], 2, color, -1);
  }
}
