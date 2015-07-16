#ifndef PINOT_TRACKER3D_H
#define PINOT_TRACKER3D_H

#include <iostream>
#include <fstream>
#include <string>
#include <opencv/cv.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include <unordered_map>
#include <map>
#include <vector>
#include <memory>
#include <set>
#include <random>
#ifdef __WIN64
#include <Eigen/Dense>
#include <Eigen/Geometry>
#endif
#ifdef __unix__
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#endif

#include "../include/matcher.h"
#include "../include/params.h"
#include "../../clustering/include/DBScanClustering.h"
#include "../include/borg_cube.h"
#include "../../utilities/include/Constants.h"
#include "../../utilities/include/DebugFunctions.h"
#include "../include/params.h"

namespace pinot_tracker {

class Tracker3D {
 public:
  Tracker3D(TrackerParams params)
      : params_(params),
        m_threshold(30),
        m_octave(3),
        m_patternScale(1.0f),
        m_featuresDetector(),
        m_featureMatcher(new cv::BFMatcher(cv::NORM_HAMMING, true)),
        m_prevFrame(),
        m_dbClusterer(),
        m_fstCube(),
        m_updatedCube() {
    m_featuresDetector.create("Feature2D.BRISK");
  };

  virtual ~Tracker3D();

  void init(cv::Mat& rgb, cv::Mat& disaprity, cv::Point2d& top_left,
            cv::Point2d& bottom_right);

  void init(cv::Mat& rgb, cv::Mat& disparity, cv::Mat& mask);

  void computeNext(const Mat& rgb, const Mat& disparity, Mat& out);

  void drawResult(cv::Mat& out);

  void close();

  Point3f getCurrentCentroid() { return m_updatedCentroid; }

  std::vector<Point3f> getCurrentBB() { return m_updatedCube.m_pointsFront; }


 private:
  void getCurrentPoints(const vector<int>& currentFaces,
                        vector<Status*>& pointsStatus,
                        vector<Point3f*>& fstPoints,
                        vector<Point3f*>& updPoints,
                        vector<KeyPoint*>& fstKeypoints,
                        vector<KeyPoint*>& updKeypoints,
                        vector<Point3f*>& relPointPos, Mat& fstDescriptors,
                        vector<Scalar*>& colors, vector<bool*>& isClustered);

  // find matches using custom match class
  int matchFeaturesCustom(const Mat& fstDescriptors,
                          const vector<KeyPoint*>& fstKeypoints,
                          const Mat& nextDescriptors,
                          const vector<KeyPoint>& extractedKeypoints,
                          vector<KeyPoint*>& updKeypoints,
                          vector<Status*>& pointsStatus);
  // uses lucas kanade to track keypoints, faster implemetation
  int trackFeatures(const Mat& grayImg, const Mat& cloud,
                    vector<Status*>& keypointStatus,
                    vector<Point3f*>& updPoints,
                    vector<KeyPoint*>& updKeypoints, int& trackedCount,
                    int& bothCount);
  // check if image is grayscale
  void checkImage(const cv::Mat& src, cv::Mat& gray);
  // get rotation matrix using mean of points
  cv::Mat getRotationMatrix(const vector<Point3f*>& fstPoints,
                            const vector<Point3f*>& updPoints,
                            const vector<Status*>& pointsStatus);

  cv::Mat getRotationMatrixDebug(const Mat& rgbImg,
                                 const vector<Point3f*>& fstPoints,
                                 const vector<Point3f*>& updPoints,
                                 const vector<Status*>& pointsStatus);
  // given rotation and scale cast votes for the new centroid
  void getVotes(const cv::Mat& cloud, vector<Status*> pointsStatus,
                vector<KeyPoint*>& fstKeypoints,
                vector<KeyPoint*>& updKeypoints, vector<Point3f*>& relPointPos,
                const Mat& rotation);
  // cluster the votes
  void clusterVotes(vector<Status>& keypointStatus);

  void clusterVotesBorder(vector<Status*>& keypointStatus,
                          vector<Point3f>& centroidVotes, vector<int>& indices,
                          vector<vector<int>>& clusters);

  // calcualte the centroid from the initial keypoints
  void initCentroid(const std::vector<cv::Point3f>& points, BoundingCube& cube);
  // calculate the relative position of the initial keypoints
  void initRelativePosition(BoundingCube& cube);

  void updateCentroid(const vector<Status*>& keypointStatus,
                      const Mat& rotation);

  void updatePointsStatus(vector<Status*>& pointsStatus,
                          vector<bool>& isClustered);

  void initBBox(const Mat& cloud);

  bool isAppearanceNew(const Mat& rotation);

  bool isCurrentApperanceToLearn(const vector<float>& visibilityRatio,
                                 const double& medianAngle, int& faceToLearn);

  bool learnFrame(const Mat& rgb, const Mat& cloud,
                  const vector<bool>& isFaceVisible,
                  const vector<float>& visibilityRatio, const Mat& rotation);

  void learnFrame(const Mat& rgb, const Mat& cloud, const int& faceToLearn,
                  const Mat& rotation);

  void learnFace(const Mat1b& mask, const Mat& rgb, const Mat& cloud,
                 const Mat& rotation, const int& face, BoundingCube& fstCube,
                 BoundingCube& updatedCube);

  int learnFaceDebug(const Mat1b& mask, const Mat& rgb, const Mat& cloud,
                     const Mat& rotation, const int& face, BoundingCube& fstCube,
                     BoundingCube& updatedCube, Mat& out);

  void debugTrackingStep(
      const cv::Mat& fstFrame, const cv::Mat& scdFrame,
      const std::vector<int>& indices, std::vector<vector<int>>& clusters,
      const bool isLost, const bool isLearning,
      const vector<Status*>& pointsStatus, vector<KeyPoint*>& fstKeypoints,
      const vector<Point3f*>& fstPoints, const vector<Point3f*>& updPoints,
      const vector<KeyPoint*>& updKeypoints, const vector<Scalar*>& colors,
      std::vector<bool> visibleFaces, std::vector<float>& visibilityRatio,
      double angularDistance, Mat& out);

  void debugTrackingStepICRA(
      const cv::Mat& fstFrame, const cv::Mat& scdFrame,
      const std::vector<int>& indices, std::vector<vector<int>>& clusters,
      const bool isLost, const bool isLearning,
      const vector<Status*>& pointsStatus, vector<KeyPoint*>& fstKeypoints,
      const vector<Point3f*>& fstPoints, const vector<Point3f*>& updPoints,
      const vector<KeyPoint*>& updKeypoints, const vector<Scalar*>& colors,
      std::vector<bool> visibleFaces, std::vector<float>& visibilityRatio,
      double angularDistance, Mat& out);

  void debugTrackingStepPar(cv::Mat fstFrame, cv::Mat scdFrame);

  inline float getDistance(const cv::KeyPoint& a, const cv::KeyPoint& b) {
    return sqrt(pow(a.pt.x - b.pt.x, 2) + pow(a.pt.y - b.pt.y, 2));
  }

  inline float getDistance(const cv::Point2f& a, const cv::Point2f& b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
  }

  inline float getDistance(const cv::Point3f& a, const cv::Point3f& b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
  }

  inline bool isKeypointValid(const Status& s) {
    return (s == Status::MATCH || s == Status::TRACK || s == Status::INIT ||
            s == Status::BOTH);
  }

  inline bool isKeypointTracked(const Status& s) {
    return (s == Status::MATCH || s == Status::TRACK || s == Status::BOTH);
  }

  inline cv::Point2f mult(cv::Mat2f& rot, cv::Point2f& p) {
    return cv::Point2f(rot.at<float>(0, 0) * p.x + rot.at<float>(1, 0) * p.y,
                       rot.at<float>(0, 1) * p.x + rot.at<float>(1, 1) * p.y);
  }

  inline Point3f getMean(const vector<Point3f>& points) {
    Point3f m(0, 0, 0);
    const int size = points.size();
    const float sizeF = static_cast<float>(size);

    for (size_t i = 0; i < size; i++) {
      m.x += points[i].x;
      m.y += points[i].y;
      m.z += points[i].z;
    }

    m.x = m.x / sizeF;
    m.y = m.y / sizeF;
    m.z = m.z / sizeF;

    return m;
  }

  void calculateVisibility(const Mat& rotation, const BoundingCube& fstCube,
                           vector<bool>& isFaceVisible,
                           vector<float>& visibilityRatio);

  void calculateVisibilityEigen(const Mat& rotation, const BoundingCube& fstCube,
                                vector<bool>& isFaceVisible,
                                vector<float>& visibilityRatio);

  std::string getResultName(string path);

  int getFilesCount(string path, string substring);

  void debugLearnedModel(const Mat& rgb, int face, const Mat& rotation);

  void drawLearnedFace(const Mat& rgb, int face, Mat& out);

  /*void updateClusteredParams(float scale, float angle);*/

  void debugCalculations();

  double getQuaternionMedianDist(const vector<Eigen::Quaterniond>& history, int window,
                                 const Eigen::Quaterniond& q);

  bool findObject(BoundingCube& fst, BoundingCube& upd, const Mat& extractedDescriptors,
                  const vector<KeyPoint>& extractedKeypoints, int& faceFound,
                  int& matchesNum);

  // object to extract and compare features
  cv::BRISK m_featuresDetector;
  std::unique_ptr<cv::DescriptorMatcher> m_featureMatcher;
  Matcher m_customMatcher;
  /****************************************************************************/
  /*                          Configuration Parameters                        */
  /****************************************************************************/
  TrackerParams params_;
  int m_threshold;
  int m_octave;
  float m_patternScale;
  /*********************************************************************************************/
  /*                          INITIAL MODEL */
  /*********************************************************************************************/
  cv::Mat m_fstFrame, m_firstCloud;
  BoundingCube m_fstCube;
  vector<vector<bool>> m_isPointClustered;

  /*********************************************************************************************/
  /*                          UPDATED MODEL */
  /*********************************************************************************************/
  // std::vector<cv::Point2f> m_matchedPoints;
  // std::vector<cv::Point2f> m_trackedPoints;
  cv::Mat m_currFrame, m_currCloud;
  BoundingCube m_updatedCube;
  vector<int> m_currentFaces;

  /*********************************************************************************************/
  /*                          VOTING VARIABLES */
  /*********************************************************************************************/
  std::vector<cv::Point3f> m_centroidVotes;
  std::vector<bool> m_clusteredCentroidVotes;
  std::vector<bool> m_clusteredBorderVotes;
  DBScanClustering<cv::Point3f*> m_dbClusterer;

  // vectors of points used to update after matching and tracking are performed

  std::vector<cv::KeyPoint> m_initKeypoints;
  unsigned int m_initKPCount;
  /*********************************************************************************************/
  /*                          CENTROID E BBOX VARIABLES */
  /*********************************************************************************************/
  // cv::Point3f m_firstCentroid;
  cv::Point3f m_updatedCentroid;
  cv::Rect m_fstBBox;

  std::vector<uchar> m_status;
  std::vector<float> m_errors;

  cv::Mat m_descriptor;
  cv::Mat m_foregroundMask;

  cv::Mat m_prevFrame;

  // variables used to find new learning frames
  std::set<pair<int, int>> m_learnedFrames;

  int m_validKeypoints;
  int m_clusterVoteSize;
  /*********************************************************************************************/
  /*                        REPROJECTION */
  /*********************************************************************************************/
  float m_focal;
  Point2f m_imageCenter;

  /*********************************************************************************************/
  /*                        TIMING */
  /*********************************************************************************************/
  int m_numFrames;

  vector<float> m_partialTimes;
  float m_frameTime;

  /*********************************************************************************************/
  /*                   RESULTS AND DEBUG */
  /*********************************************************************************************/
  std::ofstream m_debugFile, m_timeFile, m_matrixFile, m_resultFile;
  cv::VideoWriter m_debugWriter;
  std::vector<std::vector<cv::Scalar>> m_pointsColor;
  std::string m_resultName;
  std::vector<cv::Scalar> m_faceColors;
  cv::Mat m_prevRotation;
  cv::Mat m_initRotation;
  stringstream m_quaterionStream;

  /*********************************************************************************************/
  /*                        LEARNING */
  /*********************************************************************************************/
  std::vector<cv::Mat> m_pointsOfView;
  std::vector<cv::Mat> m_learnedFaces;
  vector<Eigen::Quaterniond> m_quaternionHistory;
  vector<float> m_learnedFaceVisibility;
  vector<double> m_learnedFaceMedianAngle;
  /*********************************************************************************************/
  /*                        LEARNING */
  /*********************************************************************************************/
  std::vector<bool> m_visibleFaces;

  bool hasAppearanceToChange;

  std::string toPythonString2(const vector<Point3f>& firstFrameCloud,
                              const vector<Point3f>& updatedFrameCloud,
                              const vector<Status>& keypointStatus);
};

}  // end namespace

#endif
