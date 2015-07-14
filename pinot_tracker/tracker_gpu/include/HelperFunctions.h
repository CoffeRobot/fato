#ifndef HELPERFUNCTIONS_H
#define HELPERFUNCTIONS_H

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/core/core.hpp>
#include <vector>

using namespace cv;
using namespace cv::gpu;
using namespace std;

namespace pinot {

namespace gpu {

static void download(const GpuMat& d_mat, vector<Point2f>& vec) {
  vec.resize(d_mat.cols);
  Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
  d_mat.download(mat);
}

static void download(const GpuMat& d_mat, vector<uchar>& vec) {
  vec.resize(d_mat.cols);
  Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
  d_mat.download(mat);
}

static void upload(const vector<Point2f>& points, GpuMat& d_points) {
  Mat mat(1, points.size(), CV_32FC2, (void*)&points[0]);
  d_points.upload(mat);
}

/*inline float getDistance(const cv::Point2f& a, const cv::Point2f& b)
            {
                    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}*/


inline cv::Mat1b getMask(int rows, int cols, const cv::Point2d& begin,
                         const cv::Point2d& end) {
  cv::Mat1b mask(rows, cols, static_cast<uchar>(0));
  rectangle(mask, begin, end, static_cast<uchar>(255), -1);
  return mask;
}
}
}
#endif
