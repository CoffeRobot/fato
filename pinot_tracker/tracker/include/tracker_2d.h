/*****************************************************************************/
/*  Copyright (c) 2015, Alessandro Pieropan                                  */
/*  All rights reserved.                                                     */
/*                                                                           */
/*  Redistribution and use in source and binary forms, with or without       */
/*  modification, are permitted provided that the following conditions       */
/*  are met:                                                                 */
/*                                                                           */
/*  1. Redistributions of source code must retain the above copyright        */
/*  notice, this list of conditions and the following disclaimer.            */
/*                                                                           */
/*  2. Redistributions in binary form must reproduce the above copyright     */
/*  notice, this list of conditions and the following disclaimer in the      */
/*  documentation and/or other materials provided with the distribution.     */
/*                                                                           */
/*  3. Neither the name of the copyright holder nor the names of its         */
/*  contributors may be used to endorse or promote products derived from     */
/*  this software without specific prior written permission.                 */
/*                                                                           */
/*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      */
/*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        */
/*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    */
/*  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     */
/*  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   */
/*  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         */
/*  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    */
/*  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    */
/*  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      */
/*  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    */
/*  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     */
/*****************************************************************************/

#ifndef PINOT_TRACKER2_H
#define PINOT_TRACKER2_H

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
//TODO: find a nicer way to help qtcreator finds headers
#include "../../clustering/include/DBScanClustering.h"
#include "../../utilities/include/DebugFunctions.h"
#include "../../utilities/include/constants.h"
#include "../../utilities/include/profiler.h"
// #include <profiler.h>

#include "matcher.h"
#include "params.h"

namespace pinot_tracker {

class Tracker2D {
 public:
  Tracker2D(const TrackerParams& params)
      : params_(params),
        m_featuresDetector(30, 3, 1.0f),
        m_featureMatcher(new cv::BFMatcher(cv::NORM_HAMMING, true)),
        m_prevFrame(),
        m_dbClusterer() {
    m_featuresDetector.create("Feature2D.BRISK");
  };

  virtual ~Tracker2D();

  void init(cv::Mat& src, cv::Mat& mask);

  void init(cv::Mat& src, const cv::Point2d& fst, const cv::Point2d& scd);

  void computeNext(cv::Mat& next, cv::Mat& out);

  void drawResult(cv::Mat& out);

  void close();

  cv::Point2f getCentroid() { return m_updatedCentroid; }
  std::vector<cv::Point2f> getBoundingBox() { return m_updatedBBPoints; }

  int m_threshold;
  int m_octave;
  float m_patternScale;

 private:
  void extractSyntheticKeypoints(const cv::Mat& src,
                                 std::vector<cv::KeyPoint>& points,
                                 cv::Mat& descriptors);

  // find matches using brisk features
  void matchFeatures(const cv::Mat& grayImg);
  // find matches using custom match class
  int matchFeaturesCustom(const cv::Mat& grayImg, std::vector<cv::KeyPoint>& nextKeypoint,
                          cv::Mat& nextDescriptors);
  // uses lucas kanade to track keypoints, faster implemetation
  void trackFeatures(const cv::Mat& grayImg, int& trackedCount, int& bothCount);
  // check if image is grayscale
  void checkImage(const cv::Mat& src, cv::Mat& gray);
  // get the median of found points
  float getMedianScale();
  // get the media rotation of found points
  float getMedianRotation();
  // given rotation and scale cast votes for the new centroid
  void getVotes(float scale, float angle);
  // cluster the votes
  void clusterVotes();
  // debug version of the clustering method
  void clusterVotesDebug(std::vector<int>& indices,
                         std::vector<std::vector<int>>& clusters);

  void clusterVotesBorder(std::vector<int>& indices,
                          std::vector<std::vector<int>>& clusers);

  // calcualte the centroid from the initial keypoints
  void initCentroid(std::vector<cv::KeyPoint>& keypoints);
  // calculate the relative position of the initial keypoints
  void initRelativePosition();

  void calculateCentroid(std::vector<cv::KeyPoint>& keypoints, float scale,
                         float angle, cv::Point2f& p);

  void getClusteredEstimations(std::vector<float>& scales,
                               std::vector<float>& angles);

  void updateVotes();
  // debug version of update votes
  void updateVotesDebug(std::vector<int>& indices,
                        std::vector<std::vector<int>>& clusters);

  void updateCentroid();

  void initBBox();

  bool learnFrame(float scale, float angle);

  void addLearnedFrame(float scale, float angle);

  void learnKeypoints(const int numMatches, const int numTracked,
                      const int numBoth, const cv::Mat& grayImg,
                      const cv::Mat& next,
                      const std::vector<cv::KeyPoint>& keypoints);

  void backprojectKeypoins(const std::vector<const cv::KeyPoint*>& keypoints,
                           cv::Mat& out);

  void debugTrackingStep(const cv::Mat& fstFrame, const cv::Mat& scdFrame,
                         const std::vector<int>& indices,
                         std::vector<std::vector<int>>& clusters, cv::Mat& out);

  void parDebugtrackingStep(const cv::Mat& fstFrame, const cv::Mat& scdFrame);

  void debugTrackingStepPar(cv::Mat fstFrame, cv::Mat scdFrame);

  inline float getDistance(const cv::KeyPoint& a, const cv::KeyPoint& b) {
    return sqrt(pow(a.pt.x - b.pt.x, 2) + pow(a.pt.y - b.pt.y, 2));
  }

  inline float getDistance(const cv::Point2f& a, const cv::Point2f& b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
  }

  inline bool isKeypointValid(int id) {
    return (m_keypointStatus[id] == Status::MATCH ||
            m_keypointStatus[id] == Status::TRACK ||
            m_keypointStatus[id] == Status::INIT ||
            m_keypointStatus[id] == Status::BOTH);
  }

  inline bool isKeypointTracked(int id) {
    return (m_keypointStatus[id] == Status::MATCH ||
            m_keypointStatus[id] == Status::TRACK ||
            m_keypointStatus[id] == Status::BOTH);
  }

  inline bool isKeypointValidCluster(int id) {
    return isKeypointValid(id) && m_clusteredCentroidVotes[id];
  }

  inline cv::Point2f mult(cv::Mat2f& rot, cv::Point2f& p) {
    return cv::Point2f(rot.at<float>(0, 0) * p.x + rot.at<float>(1, 0) * p.y,
                       rot.at<float>(0, 1) * p.x + rot.at<float>(1, 1) * p.y);
  }

  void updateKeypoints();

  void updateClusteredParams(float scale, float angle);

  void printKeypointStatus(std::string filename, std::string message);

  void debugCalculations();

  // object to extract and compare features
  cv::BRISK m_featuresDetector;
  std::unique_ptr<cv::DescriptorMatcher> m_featureMatcher;
  Matcher m_customMatcher;

  cv::Mat m_firstFrameDescriptors;
  std::vector<cv::KeyPoint> m_firstFrameKeypoints;
  std::vector<cv::KeyPoint> m_updatedKeypoints;
  std::vector<Status> m_keypointStatus;
  std::vector<cv::Point2f> m_relativePointsPos;
  std::vector<cv::Point2f> m_centroidVotes;
  std::vector<bool> m_clusteredCentroidVotes;
  std::vector<bool> m_clusteredBorderVotes;
  std::vector<cv::Scalar> m_pointsColor;
  DBScanClustering<cv::Point2f*> m_dbClusterer;

  // vectors of points used to update after matching and tracking are performed
  std::vector<cv::Point2f> m_matchedPoints;
  std::vector<cv::Point2f> m_trackedPoints;

  std::vector<cv::KeyPoint> m_initKeypoints;
  unsigned int m_initKPCount;

  cv::Point2f m_firstCentroid;
  cv::Point2f m_updatedCentroid;

  cv::Rect m_fstBBox;
  std::vector<cv::Point2f> m_fstRelativeBBPoints;
  std::vector<cv::Point2f> m_fstBBPoints;
  std::vector<cv::Point2f> m_updatedBBPoints;

  std::vector<uchar> m_status;
  std::vector<float> m_errors;

  cv::Mat m_descriptor;
  cv::Mat m_foregroundMask;

  cv::Mat m_prevFrame;
  cv::Mat m_currFrame;
  cv::Mat m_fstFrame;

  // variables used to find new learning frames
  std::set<std::pair<int, int>> m_learnedFrames;
  float m_angleStep;
  float m_scaleStep;
  float m_lastAngle;
  float m_lastScale;

  // variables used for debugging purposes
  float m_mAngle;
  float m_mScale;
  int m_validKeypoints;
  int m_clusterVoteSize;

  TrackerParams params_;

  /*********************************************************************************************/
  /*                        TIMING */
  /*********************************************************************************************/
  float m_extractT, m_matchT, m_trackT, m_scaleT, m_rotT, m_updateT, m_votingT,
      m_clusterT, m_centroidT, m_updateVoteT, m_drawT;
  int m_numFrames;
};

}  // end namespace

#endif
