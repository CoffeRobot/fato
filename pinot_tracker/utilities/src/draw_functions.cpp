#include "../include/draw_functions.h"

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

}
