#include "../include/utilities.h"

namespace pinot_tracker{

cv::Mat1b getMask(int rows, int cols, const cv::Point2d& begin,
                         const cv::Point2d& end) {
  cv::Mat1b mask(rows, cols, static_cast<uchar>(0));
  rectangle(mask, begin, end, static_cast<uchar>(255), -1);
  return mask;
}

} // end namespace

