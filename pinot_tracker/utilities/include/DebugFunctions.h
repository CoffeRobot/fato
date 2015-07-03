#pragma once
#include <opencv/cv.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <random>
#ifdef __unix__
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#else
#include <Eigen/Dense>
#include <Eigen/Geometry>
#endif
#include <fstream>

#include "Constants.h"
#include "ToString.h"

using namespace std;
using namespace cv;
using namespace Eigen;

void cross(const cv::Point2f& p, const Scalar& c, int width, cv::Mat& out);

void drawCentroidVotes(const vector<KeyPoint>& keypoints,
                       vector<Point2f>& points, const vector<bool>& clustered,
                       const vector<bool>& border, const vector<Status>& status,
                       bool drawLines, bool drawFalse, Mat& out);

void drawCentroidVotes(const vector<Point3f>& keypoints,
                       vector<Point3f>& points, const vector<bool>& clustered,
                       const vector<bool>& border, const vector<Status>& status,
                       bool drawLines, bool drawFalse, const float focal,
                       const Point2f& center, ofstream& file, Mat& out);

void drawCentroidVotes(const vector<Point3f>& keypoints,
                       vector<Point3f>& points, const vector<bool>& clustered,
                       const vector<bool>& border, const vector<Status>& status,
                       bool drawLines, bool drawFalse, const float focal,
                       const Point2f& center, Mat& out);

void drawCentroidVotes(const vector<Point3f*>& keypoints,
                       const vector<Point3f>& votes,
                       const vector<bool>& clustered,
                       const vector<bool>& border,
                       const vector<Status*>& status, bool drawLines,
                       bool drawFalse, const float focal, const Point2f& center,
                       Mat& out);

void drawCentroidVotes(const vector<Point3f*>& keypoints,
                       const vector<Point3f>& votes,
                       const vector<bool>& clustered,
                       const vector<bool>& border,
                       const vector<Status*>& status, bool drawLines,
                       bool drawFalse, const float focal, const Point2f& center,
                       ofstream& file, Mat& out);

void buildCompositeImg(const Mat& fst, const Mat& scd, Mat& out);

void drawObjectLocation(const Point2f& fstC, const vector<Point2f>& fstBBox,
                        const Point2f& scdC, const vector<Point2f>& scdBBox,
                        Mat& out);

void drawObjectLocation(const Point3f& fstC, const vector<Point3f>& fstBBox,
                        const Point3f& scdC, const vector<Point3f>& scdBBox,
                        const float focal, const Point2f& center, Mat& out);

/*void drawObjectLocation(const BorgCube& fstCube, const BorgCube& updCube,
                        const vector<bool>& visibleFaces, const float focal,
                        const Point2f& imgCenter, Mat& out);

void drawObjectLocation(const BorgCube& updCube,
                        const vector<bool>& visibleFaces, const float focal,
                        const Point2f& imgCenter, Mat& out);
*/

void drawBoundingCube(const cv::Point3f& scdC,
                      const std::vector<cv::Point3f>& scdFrontBox,
                      const std::vector<cv::Point3f>& scdBackBox,
                      const float focal, const cv::Point2f& imgCenter,
                      cv::Mat& out);

void drawKeypointsMatching(const vector<KeyPoint>& fstPoint,
                           const vector<KeyPoint>& scdPoints,
                           const vector<Status>& pointStatus,
                           const vector<Scalar>& colors, int& numMatch,
                           int& numTrack, int& numBoth, bool drawLines,
                           Mat& out);

void drawPointsMatching(const vector<Point3f>& fstPoints,
                        const vector<Point3f>& scdPoints,
                        const vector<Status>& pointStatus,
                        const vector<Scalar>& colors, int& numMatch,
                        int& numTrack, int& numBoth, bool drawLines,
                        const float focal, const Point2f& center, Mat& out);

void drawPointsMatching(const vector<Point3f*>& fstPoints,
                        const vector<Point3f*>& scdPoints,
                        const vector<Status*>& pointStatus,
                        const vector<Scalar*>& colors, int& numMatch,
                        int& numTrack, int& numBoth, bool drawLines,
                        const float focal, const Point2f& center, Mat& out);

void drawPointsMatchingICRA(const vector<Point3f*>& fstPoints,
                            const vector<Point3f*>& scdPoints,
                            const vector<Status*>& pointStatus,
                            const vector<Scalar*>& colors, int& numMatch,
                            int& numTrack, int& numBoth, bool drawLines,
                            const float focal, const Point2f& center, Mat& out);

void countKeypointsMatching(const vector<KeyPoint>& fstPoint,
                            const vector<KeyPoint>& scdPoints,
                            const vector<Status>& pointStatus, int& numMatch,
                            int& numTrack, int& numBoth);

void countKeypointsMatching(const vector<Status*>& pointStatus, int& numMatch,
                            int& numTrack, int& numBoth);

void drawKeipointsStats(const int init, const int matched, const int tracked,
                        const int both, Mat& out);

void drawInformationHeader(const int numFrames, const float scale,
                           const float angle, int clusterSize, int matched,
                           int tracked, Mat& out);

void drawInformationHeader(const Point2f& top, const string information,
                           float alpha, int width, int height, Mat& out);

void drawInformationHeaderICRA(Point2f& top, const string frame,
                               const string angle, const string visibility,
                               float alpha, int width, int height, Mat& out);

void drawTriangle(const Point2f& a, const Point2f& b, const Point2f& c,
                  Scalar color, float alpha, Mat& out);

void drawTriangleMask(const Point2f& a, const Point2f& b, const Point2f& c,
                      Mat1b& out);

void debugCalculations(std::string path,
                       const std::vector<KeyPoint>& currentKps,
                       const std::vector<KeyPoint>& fstKps);

std::string faceToString(int face);

cv::Point2f reprojectPoint(const float focal, const cv::Point2f& center,
                           const cv::Point3f& src);

bool reprojectPoint(const float focal, const Point2f& center,
                    const Point3f& src, Point2f& dst);

std::string toString(const Matrix3d& rotation);

std::string toString(const Quaterniond& quaternion);

void drawBoundingBox(const std::vector<cv::Point2f>& box, cv::Scalar& color,
                     Mat& out);

void drawVotes(const std::vector<Point2f>& votes, cv::Scalar& color, Mat& out);

template <typename T>
string toString(const Mat& rotation, int precision) {

  stringstream ss;
  ss.precision(precision);
  for (size_t i = 0; i < rotation.rows; i++) {
    for (size_t j = 0; j < rotation.cols; j++) {
      ss << " | " << std::fixed << rotation.at<T>(i, j);
    }
    ss << " |\n";
  }

  return ss.str();
}

std::string toPythonString(const cv::Mat& rotation);

std::string toPythonArray(const cv::Mat& rotation);

std::string toPythonString(const std::vector<cv::Point3f>& cloud);
