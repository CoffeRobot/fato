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
#include "../../utilities/include/constants.h"
#include "../../utilities/include/DebugFunctions.h"
#include "../include/params.h"

namespace pinot_tracker {

class Tracker3D {
 public:
  Tracker3D()
      : params_(),
        m_threshold(30),
        m_octave(3),
        m_patternScale(1.0f),
        m_featuresDetector(),
        m_featureMatcher(new cv::BFMatcher(cv::NORM_HAMMING, true)),
        m_prevFrame(),
        m_dbClusterer(),
        m_fstCube(),
        m_updatedCube(),
        log_header("TRACKER: ") {
    m_featuresDetector.create("Feature2D.BRISK");
  };

  ~Tracker3D();

  int init(TrackerParams params, cv::Mat& rgb, cv::Mat& points,
           cv::Point2d& top_left, cv::Point2d& bottom_right);

  int init(cv::Mat& rgb, cv::Mat& points, cv::Mat& mask);

  void clear();

  void computeNext(const cv::Mat& rgb, const cv::Mat& points, cv::Mat& out);

  void drawResult(cv::Mat& out);

  void close();

  cv::Point3f getCurrentCentroid() { return m_updatedCentroid; }

  std::vector<cv::Point3f> getFrontBB() { return m_updatedCube.m_pointsFront; }
  std::vector<cv::Point3f> getBackBB() { return m_updatedCube.m_pointsBack; }

  void getActivePoints(std::vector<cv::Point3f *> &points,
                        std::vector<cv::Point3f *> &votes);

  void drawObjectLocation(cv::Mat& out);

 private:
  void getCurrentPoints(const std::vector<int>& currentFaces,
                        std::vector<Status*>& pointsStatus,
                        std::vector<cv::Point3f*>& fstPoints,
                        std::vector<cv::Point3f*>& updPoints,
                        std::vector<cv::KeyPoint*>& fstKeypoints,
                        std::vector<cv::KeyPoint*>& updKeypoints,
                        std::vector<cv::Point3f*>& relPointPos,
                        cv::Mat& fstDescriptors,
                        std::vector<cv::Scalar*>& colors,
                        std::vector<bool*>& isClustered);

  // find matches using custom match class
  int matchFeaturesCustom(const cv::Mat& fstDescriptors,
                          const std::vector<cv::KeyPoint*>& fstKeypoints,
                          const cv::Mat& nextDescriptors,
                          const std::vector<cv::KeyPoint>& extractedKeypoints,
                          std::vector<cv::KeyPoint*>& updKeypoints,
                          std::vector<Status*>& pointsStatus);
  // uses lucas kanade to track keypoints, faster implemetation
  int trackFeatures(const cv::Mat& grayImg, const cv::Mat& cloud,
                    std::vector<Status*>& keypointStatus,
                    std::vector<cv::Point3f*>& updPoints,
                    std::vector<cv::KeyPoint*>& updKeypoints, int& trackedCount,
                    int& bothCount);
  // check if image is grayscale
  void checkImage(const cv::Mat& src, cv::Mat& gray);
  // get rotation matrix using mean of points
  cv::Mat getRotationMatrix(const std::vector<cv::Point3f*>& fstPoints,
                            const std::vector<cv::Point3f*>& updPoints,
                            const std::vector<Status*>& pointsStatus);

  cv::Mat getRotationMatrixDebug(const cv::Mat& rgbImg,
                                 const std::vector<cv::Point3f*>& fstPoints,
                                 const std::vector<cv::Point3f*>& updPoints,
                                 const std::vector<Status*>& pointsStatus);
  // given rotation and scale cast votes for the new centroid
  void getVotes(const cv::Mat& cloud, std::vector<Status*> pointsStatus,
                std::vector<cv::KeyPoint*>& fstKeypoints,
                std::vector<cv::KeyPoint*>& updKeypoints,
                std::vector<cv::Point3f*>& relPointPos,
                const cv::Mat& rotation);
  // cluster the votes
  void clusterVotes(std::vector<Status>& keypointStatus);

  void clusterVotesBorder(std::vector<Status*>& keypointStatus,
                          std::vector<cv::Point3f>& centroidVotes,
                          std::vector<int>& indices,
                          std::vector<std::vector<int>>& clusters);

  // calcualte the centroid from the initial keypoints
  bool initCentroid(const std::vector<cv::Point3f>& points, BoundingCube& cube);
  // calculate the relative position of the initial keypoints
  void initRelativePosition(BoundingCube& cube);

  void updateCentroid(const std::vector<Status*>& keypointStatus,
                      const cv::Mat& rotation);

  void updatePointsStatus(std::vector<Status*>& pointsStatus,
                          std::vector<bool>& isClustered);

  void initBBox(const cv::Mat& cloud);

  bool isAppearanceNew(const cv::Mat& rotation);

  bool isCurrentApperanceToLearn(const std::vector<float>& visibilityRatio,
                                 const double& medianAngle, int& faceToLearn);

  bool learnFrame(const cv::Mat& rgb, const cv::Mat& cloud,
                  const std::vector<bool>& isFaceVisible,
                  const std::vector<float>& visibilityRatio,
                  const cv::Mat& rotation);

  void learnFrame(const cv::Mat& rgb, const cv::Mat& cloud,
                  const int& faceToLearn, const cv::Mat& rotation);

  void learnFace(const cv::Mat1b& mask, const cv::Mat& rgb,
                 const cv::Mat& cloud, const cv::Mat& rotation, const int& face,
                 BoundingCube& fstCube, BoundingCube& updatedCube);

  int learnFaceDebug(const cv::Mat1b& mask, const cv::Mat& rgb,
                     const cv::Mat& cloud, const cv::Mat& rotation,
                     const int& face, BoundingCube& fstCube,
                     BoundingCube& updatedCube, cv::Mat& out);

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

  inline cv::Point3f getMean(const std::vector<cv::Point3f>& points) {
    cv::Point3f m(0, 0, 0);
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

  void calculateVisibility(const cv::Mat& rotation, const BoundingCube& fstCube,
                           std::vector<bool>& isFaceVisible,
                           std::vector<float>& visibilityRatio);

  void calculateVisibilityEigen(const cv::Mat& rotation,
                                const BoundingCube& fstCube,
                                std::vector<bool>& isFaceVisible,
                                std::vector<float>& visibilityRatio);

  std::string getResultName(std::string path);

  int getFilesCount(std::string path, std::string substring);

  void debugLearnedModel(const cv::Mat& rgb, int face, const cv::Mat& rotation);

  void drawLearnedFace(const cv::Mat& rgb, int face, cv::Mat& out);

  /*void updateClusteredParams(float scale, float angle);*/

  void debugCalculations();

  double getQuaternionMedianDist(const std::vector<Eigen::Quaterniond>& history,
                                 int window, const Eigen::Quaterniond& q);

  bool findObject(BoundingCube& fst, BoundingCube& upd,
                  const cv::Mat& extractedDescriptors,
                  const std::vector<cv::KeyPoint>& extractedKeypoints,
                  int& faceFound, int& matchesNum);

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
  cv::Mat m_fstFrame;
  BoundingCube m_fstCube;
  std::vector<std::vector<bool>> m_isPointClustered;
  cv::Mat3f m_firstCloud;
  /*********************************************************************************************/
  /*                          UPDATED MODEL */
  /*********************************************************************************************/
  // std::vector<cv::Point2f> m_matchedPoints;
  // std::vector<cv::Point2f> m_trackedPoints;
  cv::Mat m_currFrame;
  BoundingCube m_updatedCube;
  std::vector<int> m_currentFaces;
  cv::Mat3f m_currCloud;
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
  std::set<std::pair<int, int>> m_learnedFrames;

  int m_validKeypoints;
  int m_clusterVoteSize;
  /*********************************************************************************************/
  /*                        REPROJECTION */
  /*********************************************************************************************/
  float m_focal;
  cv::Point2f m_imageCenter;

  /*********************************************************************************************/
  /*                        TIMING */
  /*********************************************************************************************/
  int m_numFrames;

  std::vector<float> m_partialTimes;
  float m_frameTime;

  /****************************************************************************/
  /*                   RESULTS AND DEBUG */
  /****************************************************************************/
  std::vector<std::vector<cv::Scalar>> m_pointsColor;
  std::string m_resultName;
  std::vector<cv::Scalar> m_faceColors;
  cv::Mat m_prevRotation;
  cv::Mat m_initRotation;
  std::stringstream m_quaterionStream;

  std::vector<cv::Point3f> m_votingPoints;
  std::vector<cv::Point3f> m_votedPoints;

  /*********************************************************************************************/
  /*                        LEARNING */
  /*********************************************************************************************/
  std::vector<cv::Mat> m_pointsOfView;
  std::vector<cv::Mat> m_learnedFaces;
  std::vector<Eigen::Quaterniond> m_quaternionHistory;
  std::vector<float> m_learnedFaceVisibility;
  std::vector<double> m_learnedFaceMedianAngle;
  /*********************************************************************************************/
  /*                        LEARNING */
  /*********************************************************************************************/
  std::vector<bool> m_visibleFaces;

  bool hasAppearanceToChange;

  std::string toPythonString2(const std::vector<cv::Point3f>& firstFrameCloud,
                              const std::vector<cv::Point3f>& updatedFrameCloud,
                              const std::vector<Status>& keypointStatus);

  std::string log_header;
};

}  // end namespace

#endif
