#include "../include/draw_functions.h"
#include "../include/utilities.h"

using namespace std;
using namespace cv;

namespace pinot_tracker
{

void drawBoundingBox(const vector<Point2f>& box, Scalar color, int line_width,
                     Mat& out) {
  line(out, box[0], box[1], color, line_width, 1);
  line(out, box[1], box[2], color, line_width, 1);
  line(out, box[2], box[3], color, line_width, 1);
  line(out, box[3], box[0], color, line_width, 1);
}

void drawBoundingCube(const Point3f& scdC, const vector<Point3f>& scdFrontBox,
                      const vector<Point3f>& scdBackBox, const float focal,
                      const Point2f& imgCenter, Mat& out) {
  int cols = out.cols;
  Point2f tmp;
  /*********************************************************************************************/
  /*                  Draw back face */
  /*********************************************************************************************/
  line(out, projectPoint(focal, imgCenter, scdBackBox[0]),
       projectPoint(focal, imgCenter, scdBackBox[1]), Scalar(0, 0, 255), 3);
  line(out, projectPoint(focal, imgCenter, scdBackBox[1]),
       projectPoint(focal, imgCenter, scdBackBox[2]), Scalar(0, 0, 255), 3);
  line(out, projectPoint(focal, imgCenter, scdBackBox[2]),
       projectPoint(focal, imgCenter, scdBackBox[3]), Scalar(0, 0, 255), 3);
  line(out, projectPoint(focal, imgCenter, scdBackBox[3]),
       projectPoint(focal, imgCenter, scdBackBox[0]), Scalar(0, 0, 255), 3);

  /*********************************************************************************************/
  /*                  Draw connecting lines */
  /*********************************************************************************************/
  line(out, projectPoint(focal, imgCenter, scdFrontBox[0]),
       projectPoint(focal, imgCenter, scdBackBox[0]), Scalar(0, 255, 0), 3);
  line(out, projectPoint(focal, imgCenter, scdFrontBox[1]),
       projectPoint(focal, imgCenter, scdBackBox[1]), Scalar(0, 255, 255), 3);
  line(out, projectPoint(focal, imgCenter, scdFrontBox[2]),
       projectPoint(focal, imgCenter, scdBackBox[2]), Scalar(255, 255, 0), 3);
  line(out, projectPoint(focal, imgCenter, scdFrontBox[3]),
       projectPoint(focal, imgCenter, scdBackBox[3]), Scalar(255, 0, 255), 3);

  /*********************************************************************************************/
  /*                  Draw front face */
  /*********************************************************************************************/

  tmp = projectPoint(focal, imgCenter, scdC);
  circle(out, tmp, 7, Scalar(255, 0, 0), -1);
  line(out, projectPoint(focal, imgCenter, scdFrontBox[0]),
       projectPoint(focal, imgCenter, scdFrontBox[1]), Scalar(255, 0, 0), 3);
  line(out, projectPoint(focal, imgCenter, scdFrontBox[1]),
       projectPoint(focal, imgCenter, scdFrontBox[2]), Scalar(255, 0, 0), 3);
  line(out, projectPoint(focal, imgCenter, scdFrontBox[2]),
       projectPoint(focal, imgCenter, scdFrontBox[3]), Scalar(255, 0, 0), 3);
  line(out, projectPoint(focal, imgCenter, scdFrontBox[3]),
       projectPoint(focal, imgCenter, scdFrontBox[0]), Scalar(255, 0, 0), 3);
}

}
