#pragma once
#include <opencv/cv.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
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

#include "constants.h"
#include "ToString.h"

namespace pinot_tracker{

void cross(const cv::Point2f& p, const cv::Scalar& c, int width, cv::Mat& out);

void drawCentroidVotes(const std::vector<cv::KeyPoint>& keypoints,
                       std::vector<cv::Point2f>& points,
                       const std::vector<bool>& clustered,
                       const std::vector<bool>& border,
                       const std::vector<Status>& status, bool drawLines,
                       bool drawFalse, cv::Mat& out);

void drawCentroidVotes(const std::vector<cv::Point3f>& keypoints,
                       std::vector<cv::Point3f>& points,
                       const std::vector<bool>& clustered,
                       const std::vector<bool>& border,
                       const std::vector<Status>& status, bool drawLines,
                       bool drawFalse, const float focal,
                       const cv::Point2f& center, std::ofstream& file,
                       cv::Mat& out);



void drawCentroidVotes(const std::vector<cv::Point3f*>& keypoints,
                       const std::vector<cv::Point3f>& votes,
                       const std::vector<bool>& clustered,
                       const std::vector<bool>& border,
                       const std::vector<Status*>& status, bool drawLines,
                       bool drawFalse, const float focal,
                       const cv::Point2f& center, cv::Mat& out);

void drawCentroidVotes(const std::vector<cv::Point3f*>& keypoints,
                       const std::vector<cv::Point3f>& votes,
                       const std::vector<bool>& clustered,
                       const std::vector<bool>& border,
                       const std::vector<Status*>& status, bool drawLines,
                       bool drawFalse, const float focal,
                       const cv::Point2f& center, std::ofstream& file,
                       cv::Mat& out);

void buildCompositeImg(const cv::Mat& fst, const cv::Mat& scd, cv::Mat& out);

void drawObjectLocation(const cv::Point2f& fstC,
                        const std::vector<cv::Point2f>& fstBBox,
                        const cv::Point2f& scdC,
                        const std::vector<cv::Point2f>& scdBBox, cv::Mat& out);

void drawObjectLocation(const cv::Point3f& fstC,
                        const std::vector<cv::Point3f>& fstBBox,
                        const cv::Point3f& scdC,
                        const std::vector<cv::Point3f>& scdBBox,
                        const float focal, const cv::Point2f& center,
                        cv::Mat& out);

/*void drawObjectLocation(const BorgCube& fstCube, const BorgCube& updCube,
                        const vector<bool>& visibleFaces, const float focal,
                        const Point2f& imgCenter, Mat& out);

void drawObjectLocation(const BorgCube& updCube,
                        const vector<bool>& visibleFaces, const float focal,
                        const Point2f& imgCenter, Mat& out);
*/

void drawKeypointsMatching(const std::vector<cv::KeyPoint>& fstPoint,
                           const std::vector<cv::KeyPoint>& scdPoints,
                           const std::vector<Status>& pointStatus,
                           const std::vector<cv::Scalar>& colors, int& numMatch,
                           int& numTrack, int& numBoth, bool drawLines,
                           cv::Mat& out);

void drawPointsMatching(const std::vector<cv::Point3f>& fstPoints,
                        const std::vector<cv::Point3f>& scdPoints,
                        const std::vector<Status>& pointStatus,
                        const std::vector<cv::Scalar>& colors, int& numMatch,
                        int& numTrack, int& numBoth, bool drawLines,
                        const float focal, const cv::Point2f& center,
                        cv::Mat& out);

void drawPointsMatching(const std::vector<cv::Point3f*>& fstPoints,
                        const std::vector<cv::Point3f*>& scdPoints,
                        const std::vector<Status*>& pointStatus,
                        const std::vector<cv::Scalar*>& colors, int& numMatch,
                        int& numTrack, int& numBoth, bool drawLines,
                        const float focal, const cv::Point2f& center,
                        cv::Mat& out);

void drawPointsMatchingICRA(const std::vector<cv::Point3f*>& fstPoints,
                            const std::vector<cv::Point3f*>& scdPoints,
                            const std::vector<Status*>& pointStatus,
                            const std::vector<cv::Scalar*>& colors,
                            int& numMatch, int& numTrack, int& numBoth,
                            bool drawLines, const float focal,
                            const cv::Point2f& center, cv::Mat& out);

void countKeypointsMatching(const std::vector<cv::KeyPoint>& fstPoint,
                            const std::vector<cv::KeyPoint>& scdPoints,
                            const std::vector<Status>& pointStatus,
                            int& numMatch, int& numTrack, int& numBoth);

void countKeypointsMatching(const std::vector<Status*>& pointStatus,
                            int& numMatch, int& numTrack, int& numBoth);

void drawKeipointsStats(const int init, const int matched, const int tracked,
                        const int both, cv::Mat& out);

void drawInformationHeader(const int numFrames, const float scale,
                           const float angle, int clusterSize, int matched,
                           int tracked, cv::Mat& out);

void drawInformationHeader(const cv::Point2f& top,
                           const std::string information, float alpha,
                           int width, int height, cv::Mat& out);

void drawInformationHeaderICRA(cv::Point2f& top, const std::string frame,
                               const std::string angle,
                               const std::string visibility, float alpha,
                               int width, int height, cv::Mat& out);

void drawTriangle(const cv::Point2f& a, const cv::Point2f& b,
                  const cv::Point2f& c, cv::Scalar color, float alpha,
                  cv::Mat& out);

void drawTriangleMask(const cv::Point2f& a, const cv::Point2f& b,
                      const cv::Point2f& c, cv::Mat1b& out);

void debugCalculations(std::string path,
                       const std::vector<cv::KeyPoint>& currentKps,
                       const std::vector<cv::KeyPoint>& fstKps);

std::string faceToString(int face);

cv::Point2f reprojectPoint(const float focal, const cv::Point2f& center,
                           const cv::Point3f& src);

bool reprojectPoint(const float focal, const cv::Point2f& center,
                    const cv::Point3f& src, cv::Point2f& dst);


void drawVotes(const std::vector<cv::Point2f>& votes, cv::Scalar& color,
               cv::Mat& out);



} // end namesapce
