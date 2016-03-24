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

#include "../include/tracker_2d.h"

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#define _USE_MATH_DEFINES
#include <math.h>
#include <limits>
#include <algorithm>
#include <utility>
#include <iomanip>

#include "../include/pose_estimation.h"
#include "../include/profiler.h"
#include "../include/DBScanClustering.h"
#include "../include/utilities.h"

using namespace std;
using namespace cv;

namespace fato {

Tracker::Tracker() : matcher_confidence_(0.75), matcher_ratio_(0.75) {}

Tracker::Tracker(Config& params, int descriptor_type,
                     unique_ptr<FeatureMatcher> matcher)
    : matcher_confidence_(0.75), matcher_ratio_(0.75) {
  feature_detector_ = std::move(matcher);
}

Tracker::~Tracker() { taskFinished(); }

bool Tracker::isPointValid(const int& id) {
  return m_pointsStatus[id] == Status::TRACK;
}

void Tracker::init(const cv::Mat& rgb, const cv::Point2d& fst,
                     const cv::Point2d& scd) {
  auto mask = getMask(rgb.rows, rgb.cols, fst, scd);
  init(rgb, mask);
}

void Tracker::init(const Mat& rgb, const Mat& mask) {

  if(!feature_detector_)
  {
      cerr << "Error: feature matcher is not initialized!" << endl;
      return;
  }

  Mat gray;

  m_width = rgb.cols;
  m_height = rgb.rows;

  cvtColor(rgb, gray, CV_BGR2GRAY);

  feature_detector_->setTarget(gray);

  // cout << "After set model" << endl;

  m_initKeypoints = feature_detector_->getTargetPoints();
  m_initDescriptors = feature_detector_->getTargetDescriptors();

  if(m_initKeypoints.size() == 0)
  {
    cerr << "Warning: no keypoint extracted. Nothing is going to be tracked!" << endl;
    return;
  }

  for (int i = 0; i < m_initKeypoints.size(); i++) {
    Point2f pt = m_initKeypoints[i].pt;

    m_points.push_back(pt);
  }

  // m_init_to_upd_ids.resize(m_initKeypoints.size(), -1);
  int valid_points = 0;
  for (int i = 0; i < m_initKeypoints.size(); i++) {
    Point2f& pt = m_points[i];

    if (mask.at<uchar>(pt) == 255) {
      m_pointsStatus.push_back(Status::INIT);
      valid_points++;
    } else {
      m_pointsStatus.push_back(Status::BACKGROUND);
    }
    m_updatedPoints.push_back(m_points[i]);
    m_votes.push_back(Point2f(0, 0));
    m_upd_to_init_ids.push_back(i);
    m_init_to_upd_ids.push_back(i);
  }

  if (valid_points == 0) cout << "No valid point extracted!" << endl;

  m_points_status_debug = m_pointsStatus;
  m_original_model_size = m_points.size();
  /****************************************************************************/
  /*                          INIT CENTROID                                   */
  /****************************************************************************/
  // cout << "Init centroid" << endl;
  m_initCentroid = initCentroid(m_updatedPoints);
  m_updatedCentroid = m_initCentroid;
  // cout << m_updatedCentroid << endl;
  /****************************************************************************/
  /*                          INIT RELATIVE DISTANCES                         */
  /****************************************************************************/
  // cout << "Init relative distance" << endl;
  initRelativeDistance(m_updatedPoints, m_initCentroid, m_relativeDistances);
  /****************************************************************************/
  /*                          INIT BOUNDING BOX                               */
  /****************************************************************************/
  // cout << "Init bounding box" << endl;
  initBoundingBox(mask, m_initCentroid, m_boundingBox, m_boundingBoxRelative,
                  m_boundingBoxUpdated);
  /****************************************************************************/
  /*                          INIT LEARNING                                   */
  /****************************************************************************/
  m_learned_poses.insert(make_pair<float, float>(0, 0));
  m_angle = 0;
  m_scale = 1;
  m_scale_old = m_scale;
  m_angle_old = m_angle;
  m_centroid_old = m_initCentroid;
  /****************************************************************************/
  /*                          INIT THREADS                                    */
  /****************************************************************************/
  m_isRunning = true;
  m_trackerStatus = async(launch::async, &Tracker::runTracker, this);
  m_detectorStatus = async(launch::async, &Tracker::runDetector, this);
  /****************************************************************************/
  /*                          LOADING IMGS ON GPU                             */
  /****************************************************************************/
  // cout << "load images" << endl;
  rgb.copyTo(m_init_rgb_img);
  gray.copyTo(prev_gray_);

  cout << "END" << endl;
}

void Tracker::setMatcerParameters(float confidence, float second_ratio) {
  matcher_confidence_ = confidence;
  matcher_ratio_ = second_ratio;
}

Point2f Tracker::initCentroid(const vector<Point2f>& points) {
  Point2f centroid(0, 0);
  int validPoints = 0;

  for (size_t i = 0; i < points.size(); i++) {
    if (m_pointsStatus[i] == Status::INIT) {
      centroid += points[i];
      validPoints++;
    }
  }

  centroid.x = centroid.x / static_cast<float>(validPoints);
  centroid.y = centroid.y / static_cast<float>(validPoints);

  return centroid;
}

void Tracker::initRelativeDistance(const vector<Point2f>& points,
                                     const Point2f& centroid,
                                     vector<Point2f>& relDistances) {
  relDistances.reserve(points.size());

  for (size_t i = 0; i < points.size(); i++) {
    relDistances.push_back(points[i] - centroid);
  }
}

void Tracker::initBoundingBox(const Mat& mask, const Point2f& centroid,
                                vector<Point2f>& initBox,
                                vector<Point2f>& relativeBox,
                                vector<Point2f>& updBox) {
  int minX = numeric_limits<int>::max();
  int minY = minX;
  int maxX = 0;
  int maxY = 0;

  for (size_t j = 0; j < mask.rows; j++) {
    for (size_t i = 0; i < mask.cols; i++) {
      if (mask.at<uchar>(j, i) != 0) {
        minX = minX < i ? minX : i;
        maxX = maxX > i ? maxX : i;
        minY = minY < j ? minY : j;
        maxY = maxY > j ? maxY : j;
      }
    }
  }

  initBox.push_back(Point2f(minX, minY));
  initBox.push_back(Point2f(maxX, minY));
  initBox.push_back(Point2f(maxX, maxY));
  initBox.push_back(Point2f(minX, maxY));

  for (size_t i = 0; i < initBox.size(); i++) {
    relativeBox.push_back(initBox[i] - centroid);
  }

  updBox = initBox;
}

void Tracker::getOpticalFlow(const Mat& prev, const Mat& next,
                               vector<Point2f>& points, vector<int>& ids,
                               vector<Status>& status) {
  vector<Point2f> next_points, prev_points;
  vector<uchar> next_status, prev_status;
  vector<float> next_errors, prev_errors;

  calcOpticalFlowPyrLK(prev, next, points, next_points, next_status,
                       next_errors);

  calcOpticalFlowPyrLK(next, prev, next_points, prev_points, prev_status,
                       prev_errors);
  m_flow_counter = 0;
  for (int i = 0; i < next_points.size(); ++i) {
    float error = fato::getDistance(points[i], prev_points[i]);

    Status& s = status[ids[i]];

    if (prev_status[i] == 1 && error < 20) {
      // const int& id = ids[i];
      auto id = i;

      if (s == Status::MATCH) {
        status[id] = Status::TRACK;
        m_flow_counter++;
      } else if (s == Status::LOST)
        status[id] = Status::LOST;
      else if (s == Status::NOCLUSTER) {
        status[id] = Status::LOST;
      } else if (s == Status::TRACK || s == Status::INIT) {
        status[id] = Status::TRACK;
        m_flow_counter++;
      }

      points[i] = next_points[i];
    } else {
      // status[ids[i]] = Status::LOST;
      if (status[i] != Status::BACKGROUND) status[i] = Status::LOST;
      // TODO: remove pointer and ids if lost, it will still be in the detector
      // list
    }
  }
}

float Tracker::getMedianRotation(const vector<Point2f>& initPoints,
                                   const vector<Point2f>& updPoints,
                                   const vector<int>& ids) {
  vector<double> angles;
  angles.reserve(updPoints.size());

  for (size_t i = 0; i < updPoints.size(); i++) {
    for (size_t j = 0; j < updPoints.size(); j++) {
      if (isPointValid(ids[i]) && isPointValid(ids[j])) {
        Point2f a = updPoints[i] - updPoints[j];
        Point2f b = initPoints[ids[i]] - initPoints[ids[j]];

        double val = atan2(a.y, a.x) - atan2(b.y, b.x);

        if (abs(val) > M_PI) {
          int sign = (val < 0) ? -1 : 1;
          val = val - sign * 2 * M_PI;
        }
        angles.push_back(val);
      }
    }
  }

  sort(angles.begin(), angles.end());

  double median;
  size_t size = angles.size();

  if (size == 0) return 0;

  if (size % 2 == 0) {
    median = (angles[size / 2 - 1] + angles[size / 2]) / 2;
  } else {
    median = angles[size / 2];
  }

  return static_cast<float>(median);
}

float Tracker::getMedianScale(const vector<Point2f>& initPoints,
                                const vector<Point2f>& updPoints,
                                const vector<int>& ids) {
  vector<float> scales;

  for (size_t i = 0; i < updPoints.size(); ++i) {
    for (size_t j = 0; j < updPoints.size(); j++) {
      if (isPointValid(ids[i]) && isPointValid(ids[j])) {
        float nextDistance = fato::getDistance(updPoints[i], updPoints[j]);
        float currDistance =
            fato::getDistance(initPoints[ids[i]], initPoints[ids[j]]);

        if (currDistance != 0 && i != j) {
          scales.push_back(nextDistance / currDistance);
        }
      }
    }
  }

  sort(scales.begin(), scales.end());

  float median;
  size_t size = scales.size();

  if (size == 0) {
    return 1;
  }

  if (size % 2 == 0) {
    median = (scales[size / 2 - 1] + scales[size / 2]) / 2;
  } else {
    median = scales[size / 2];
  }

  return median;
}

void Tracker::voteForCentroid(const vector<Point2f>& relativeDistances,
                                const vector<Point2f>& updPoints,
                                const float& angle, const float& scale,
                                vector<Point2f>& votes) {
  Mat2f rotMat(2, 2);

  rotMat.at<float>(0, 0) = cosf(angle);
  rotMat.at<float>(0, 1) = sinf(angle);
  rotMat.at<float>(1, 0) = -sinf(angle);
  rotMat.at<float>(1, 1) = cosf(angle);

  // int voteCount = 0;

  // cout << "VOTES: P " << updPoints.size() << " V: " << votes.size() << endl;

  for (size_t i = 0; i < updPoints.size(); i++) {
    if (isPointValid(i)) {
      const Point2f& a = updPoints[i];
      const Point2f& rm = relativeDistances[i];

      votes[i] = a - scale * mult(rotMat, rm);
    }
  }
}

void Tracker::clusterVotes(vector<Point2f>& centroidVotes,
                             vector<bool>& isClustered) {
  DBScanClustering<Point2f*> clusterer;

  vector<Point2f*> votes;
  vector<unsigned int> indices;

  for (unsigned int i = 0; i < centroidVotes.size(); i++) {
    if (isPointValid(i)) {
      votes.push_back(&centroidVotes[i]);
      indices.push_back(i);
    }
  }

  clusterer.clusterPoints(&votes, 15, 5, [](Point2f* a, Point2f* b) {
    return sqrt(pow(a->x - b->x, 2) + pow(a->y - b->y, 2));
  });

  auto res = clusterer.getClusters();
  int maxVal = 0;
  int maxId = -1;

  for (size_t i = 0; i < res.size(); i++) {
    if (res[i].size() > maxVal) {
      maxVal = res[i].size();
      maxId = i;
    }
  }

  isClustered.resize(centroidVotes.size(), false);

  int clusterVoteSize = 0;

  if (maxId > -1) {
    for (size_t i = 0; i < res[maxId].size(); i++) {
      unsigned int& id = indices[res[maxId][i]];
      isClustered[id] = true;
    }

    clusterVoteSize = res[maxId].size();
  }
}

void Tracker::updateCentroid(const float& angle, const float& scale,
                               const vector<Point2f>& votes,
                               const vector<bool>& isClustered,
                               Point2f& updCentroid) {
  updCentroid.x = 0;
  updCentroid.y = 0;

  int validKp = 0;

  for (int i = 0; i < votes.size(); ++i) {
    if (isPointValid(i) && isClustered[i]) {
      updCentroid += votes[i];
      validKp++;
    }
  }

  if (validKp == 0) {
    m_is_object_lost = true;
  } else {
    m_is_object_lost = false;
    updCentroid.x = updCentroid.x / static_cast<float>(validKp);
    updCentroid.y = updCentroid.y / static_cast<float>(validKp);
  }
}

void Tracker::updatePointsStatus(const vector<bool>& isClustered,
                                   vector<Point2f>& points,
                                   vector<Point2f>& votes,
                                   vector<Point2f>& relDistances,
                                   vector<int>& ids,
                                   vector<Status>& pointsStatus) {
  vector<int> toBeRemoved;

  for (int i = 0; i < isClustered.size(); ++i) {
    if (!isClustered[i]) {
      toBeRemoved.push_back(i);
      pointsStatus[ids[i]] = Status::LOST;
    }
  }

  int back = points.size() - 1;

  for (int i = 0; i < toBeRemoved.size(); ++i) {
    const int dest = back - i;
    const int src = toBeRemoved[i];
    std::swap(points[src], points[dest]);
    std::swap(votes[src], votes[dest]);
    std::swap(ids[src], ids[dest]);
    std::swap(relDistances[src], relDistances[dest]);
  }

  int reducedSize = points.size() - toBeRemoved.size();
  points.resize(reducedSize);
  votes.resize(reducedSize);
  ids.resize(reducedSize);
  relDistances.resize(reducedSize);
}

void Tracker::labelNotClusteredPts(const vector<bool>& isClustered,
                                     vector<Point2f>& points,
                                     vector<Point2f>& votes,
                                     vector<Point2f>& relDistances,
                                     vector<int>& ids,
                                     vector<Status>& pointsStatus) {
  for (int i = 0; i < isClustered.size(); ++i) {
    if (!isClustered[i] && m_pointsStatus[ids[i]] != Status::BACKGROUND) {
      pointsStatus[ids[i]] = Status::LOST;
    }
    // else if (isClustered[i] && pointsStatus[ids[i]] == Status::NOCLUSTER)
    //  pointsStatus[ids[i]] = Status::TRACK;
  }
}

void Tracker::discardNotClustered(std::vector<Point2f>& upd_points,
                                    std::vector<Point2f>& init_pts,
                                    cv::Point2f& upd_centroid,
                                    cv::Point2f& init_centroid,
                                    std::vector<int>& ids,
                                    std::vector<Status>& pointsStatus) {
  for (auto i = 0; i < upd_points.size(); ++i) {
    auto id = ids[i];

    if (pointsStatus[id] == Status::NOCLUSTER) {
      float init_dist = fato::getDistance(init_pts[id], init_centroid);
      float upd_dist = fato::getDistance(upd_points[i], upd_centroid) * m_scale;

      float ratio = min(init_dist, upd_dist) / max(init_dist, upd_dist);

      if (ratio < 0.85) pointsStatus[id] = Status::LOST;
    }
  }
}

void Tracker::removeLostPoints(const std::vector<bool>& isClustered,
                                 std::vector<Point2f>& points,
                                 std::vector<Point2f>& votes,
                                 std::vector<Point2f>& relDistances,
                                 std::vector<int>& ids,
                                 std::vector<Status>& pointsStatus) {
  vector<int> toBeRemoved;

  for (int i = 0; i < points.size(); ++i) {
    const int& id = ids[i];
    if (pointsStatus[i] == Status::LOST) {
      toBeRemoved.push_back(i);
    }
  }

  int back = points.size() - 1;

  for (int i = 0; i < toBeRemoved.size(); ++i) {
    const int dest = back - i;
    const int src = toBeRemoved[i];
    std::swap(points[src], points[dest]);
    std::swap(votes[src], votes[dest]);
    std::swap(ids[src], ids[dest]);
    std::swap(relDistances[src], relDistances[dest]);
  }

  int reducedSize = points.size() - toBeRemoved.size();
  points.resize(reducedSize);
  votes.resize(reducedSize);
  ids.resize(reducedSize);
  relDistances.resize(reducedSize);
}

void Tracker::updateBoundingBox(const float& angle, const float& scale,
                                  const vector<Point2f>& boundingBoxRelative,
                                  const Point2f& updCentroid,
                                  vector<Point2f>& updBox) {
  Mat2f rotMat(2, 2);

  rotMat.at<float>(0, 0) = cosf(angle);
  rotMat.at<float>(0, 1) = sinf(angle);
  rotMat.at<float>(1, 0) = -sinf(angle);
  rotMat.at<float>(1, 1) = cosf(angle);

  for (int i = 0; i < updBox.size(); ++i) {
    updBox[i] = updCentroid + scale * mult(rotMat, boundingBoxRelative[i]);
  }
}

void Tracker::computeNext(const Mat& next) {
  // cout << "Computing next! " << endl;

  m_nextRgb = next;
  m_completed = 0;
  m_trackerDone = false;
  m_matcherDone = false;

  m_learn_new_pose = false;

  m_trackerCondition.notify_one();
  m_detectorCondition.notify_one();

  std::chrono::microseconds sleepTime(1);

  while (!m_trackerDone || !m_matcherDone) {
    std::this_thread::sleep_for(sleepTime);
  }
  /****************************************************************************/
  /*                LEARNING PROCEDURE                                        */
  /****************************************************************************/
  // m_learn_new_pose = evaluatePose(m_angle, m_scale);
  /*if (m_learn_new_pose && !m_is_object_lost) {
    learnPose(m_boundingBoxUpdated, dm_prevGray, m_points, m_updatedPoints,
              m_pointsStatus, m_init_to_upd_ids);
  }*/
  /****************************************************************************/
  /*                STORE COMPUTATIONS                                        */
  /****************************************************************************/
  m_scale_old = m_scale;
  m_angle_old = m_angle;
  m_centroid_old = m_updatedCentroid;
}

void Tracker::taskFinished() {
  m_isRunning = false;
  m_trackerCondition.notify_one();
  m_detectorCondition.notify_one();
}

int Tracker::runTracker() {
  // cv::gpu::setDevice(0);

  m_trackerFrameCount = 0;
  m_trackerTime = 0;

  cout << "Tracker process initialized! \n" << endl;

  while (m_isRunning) {
    unique_lock<mutex> lock(m_trackerMutex);
    m_trackerCondition.wait(lock);
    if (!m_isRunning) break;
    if (m_nextRgb.empty()) break;
    /*********************************************************************************/
    /*********************************************************************************/
    Mat rgb = m_nextRgb;
    // cout << "before tracker \n" << endl;
    trackNext(rgb);
    m_completed++;
    m_trackerDone = true;
    // cout << "after tracker \n" << endl;
    /*********************************************************************************/
    m_trackerFrameCount++;
    /*********************************************************************************/
  }

  cout << "Tracker stopped!" << endl;
  return 0;
}

int Tracker::runDetector() {
  m_detectorFrameCount = 0;
  m_detectorTime = 0;

  cout << "Detector porcess initialized! \n" << endl;

  while (m_isRunning) {
    unique_lock<mutex> lock(m_detectorMutex);
    m_detectorCondition.wait(lock);
    if (!m_isRunning) break;
    if (m_nextRgb.empty()) break;
    /*********************************************************************************/

    /*********************************************************************************/
    Mat rgb = m_nextRgb;
    // cout << "before detector \n" << endl;
    detectNext(rgb);
    // cout << "after detector \n" << endl;
    m_matcherDone = true;
    m_completed++;
    /*********************************************************************************/
    m_detectorFrameCount++;
    /*********************************************************************************/
  }

  cout << "Matcher stopped!" << endl;
  return 0;
}

void Tracker::trackNext(Mat next) {
  /*************************************************************************************/
  /*                       LOADING IMAGES */
  /*************************************************************************************/
  Mat next_gray;
  cvtColor(next, next_gray, CV_BGR2GRAY);
  /*************************************************************************************/
  /*                       TRACKING */
  /*************************************************************************************/
  //  cout << "Optical flow " << endl;
  //TOFIX: crushing here because of empty points vector
  getOpticalFlow(prev_gray_, next_gray, m_updatedPoints, m_upd_to_init_ids,
                 m_pointsStatus);
  /*************************************************************************************/
  /*                             POSE ESTIMATION /
  /*************************************************************************************/
  //auto& profiler = Profiler::getInstance();

  vector<Point2f*> model_pts;
  vector<Point2f*> tracked_pts;
  //profiler->start("pose_n");
  getTrackedPoints(m_points, m_updatedPoints, m_upd_to_init_ids, model_pts,
                   tracked_pts);
  float scale, angle;
  getPose2D(model_pts, tracked_pts, scale, angle);
  //profiler->stop("pose_n");
  /*************************************************************************************/
  /*                             VOTING */
  /*************************************************************************************/
  //   cout << "Vote " << endl;
  voteForCentroid(m_relativeDistances, m_updatedPoints, angle, scale, m_votes);
  /*************************************************************************************/
  /*                             CLUSTERING */
  /*************************************************************************************/
  //   cout << "Cluster " << endl;
  vector<bool> isClustered;
  clusterVotes(m_votes, isClustered);
  /*************************************************************************************/
  /*                             UPDATING LIST OF POINTS */
  /*************************************************************************************/
  // TODO: tracker works better without removing points here, WHY?!?
  labelNotClusteredPts(isClustered, m_updatedPoints, m_votes,
                       m_relativeDistances, m_upd_to_init_ids, m_pointsStatus);

  // discardNotClustered(m_updatedPoints, m_points, m_updatedCentroid,
  //                    m_initCentroid, m_upd_to_init_ids, m_pointsStatus);

  // removeLostPoints(isClustered, m_updatedPoints, m_votes,
  //                 m_relativeDistances, m_pointsIds, m_pointsStatus);
  /*************************************************************************************/
  /*                             UPDATING CENTROID */
  /*************************************************************************************/
  //   cout << "Upd centroid " << endl;
  updateCentroid(angle, scale, m_votes, isClustered, m_updatedCentroid);
  /*************************************************************************************/
  /*                             UPDATING BOX */
  /*************************************************************************************/
  //   cout << "Upd box " << endl;
  updateBoundingBox(angle, scale, m_boundingBoxRelative, m_updatedCentroid,
                    m_boundingBoxUpdated);
  /*************************************************************************************/
  /*                       SWAP MEMORY */
  /*************************************************************************************/
  next_gray.copyTo(prev_gray_);
}

void Tracker::detectNext(Mat next) {
  /*************************************************************************************/
  /*                       FEATURE EXTRACTION */
  /*************************************************************************************/
  vector<KeyPoint> keypoints;
  Mat gray, descriptors;
  cvtColor(next, gray, CV_BGR2GRAY);
  // m_featuresDetector->detect(gray, keypoints);
  // m_featuresDetector->compute(gray, keypoints, descriptors);
  /*************************************************************************************/
  /*                       FEATURE MATCHING */
  /*************************************************************************************/
  vector<vector<DMatch>> matches;
  // cout << "Before match" << endl;
  feature_detector_->match(gray, keypoints, descriptors, matches);
  // cout << "After match" << endl;
  // m_customMatcher.matchHamming(descriptors, m_initDescriptors, 2, matches);
  /*************************************************************************************/
  /*                      SYNC WITH TRACKER */
  /*************************************************************************************/
  std::chrono::microseconds sleepTime(10);
  while (!m_trackerDone) {
    // sleep doing anything
    // cout << "Wainting tracker" << endl;
    this_thread::sleep_for(sleepTime);
  }
  /*************************************************************************************/
  /*                       ADD FEATURES TO TRACKER */
  /*************************************************************************************/
  if (matches.size() < 2) return;
  if (matches.at(0).size() != 2) return;

  for (size_t i = 0; i < matches.size(); i++) {
    // cout << i << endl;
    // const int& queryId = matches[i][0].queryIdx;
    // const int& trainId = matches[i][0].trainIdx;

    const DMatch& fst_match = matches.at(i).at(0);
    const DMatch& scd_match = matches.at(i).at(1);

    if (fst_match.trainIdx < 0 && fst_match.trainIdx >= keypoints.size())
      continue;
    if (fst_match.queryIdx < 0 && fst_match.queryIdx >= m_initKeypoints.size())
      continue;

    // float confidence = 1 - (matches[i][0].distance / 512.0);
    auto ratio = (fst_match.distance / scd_match.distance);

    Status& s = m_pointsStatus.at(fst_match.queryIdx);

    // if (confidence < 0.80f) continue;

    if (ratio > 0.8f) continue;

    if (s == Status::BACKGROUND) continue;

    // if (s == Status::TRACK) continue;

    m_pointsStatus.at(fst_match.queryIdx) = Status::MATCH;
    m_updatedPoints.at(fst_match.queryIdx) =
        keypoints.at(fst_match.trainIdx).pt;
  }
}

bool Tracker::evaluatePose(const float& angle, const float& scale) {
  // discretize angle
  float angle_change = fato::roundDownToNearest(angle, 0.30);
  // discretize scale
  float scale_change = fato::roundDownToNearest(scale - 1, 0.30);

  /****************************************************************************/
  /**                adding to history                                        */
  /****************************************************************************/
  if (m_angle_history.size() > 15) m_angle_history.pop_front();
  m_angle_history.push_back(angle);

  if (m_scale_history.size() > 15) m_scale_history.pop_front();
  m_scale_history.push_back(scale);

  if (m_center_history.size() > 15) m_center_history.pop_front();
  m_center_history.push_back(m_updatedCentroid);
  /****************************************************************************/
  /**                Computing variation                                      */
  /****************************************************************************/

  auto it_scale = m_scale_history.begin();
  auto prev_val_scale = *it_scale;
  auto max_variation_scale = 0.0f;
  while (it_scale != m_scale_history.end()) {
    max_variation_scale =
        std::max(max_variation_scale, abs(prev_val_scale - *it_scale));
    prev_val_scale = *it_scale;
    ++it_scale;
  }

  auto it_angle = m_angle_history.begin();
  auto prev_val_angle = *it_angle;
  auto max_variation_angle = 0.0f;
  while (it_angle != m_angle_history.end()) {
    max_variation_angle =
        std::max(max_variation_angle, abs(prev_val_angle - *it_angle));
    prev_val_angle = *it_angle;
    ++it_angle;
  }

  auto it_mov = m_center_history.begin();
  auto prev_val_mov = *it_mov;
  auto max_variation_mov = 0.0f;
  while (it_mov != m_center_history.end()) {
    max_variation_mov =
        std::max(max_variation_mov, fato::getDistance(prev_val_mov, *it_mov));
    prev_val_mov = *it_mov;
    ++it_mov;
  }

  // check if the new pose in already stored in the tracker
  pair<float, float> p(angle_change, scale_change);

  if (max_variation_angle > 0.9 || max_variation_mov > 3.0 ||
      max_variation_scale > 0.5)
    return false;

  auto it_pose = m_learned_poses.find(p);

  bool is_new_pose = false;
  if (it_pose == m_learned_poses.end()) {
    m_learned_poses.insert(p);
    is_new_pose = true;
  }

  if (is_new_pose) {
    cout << fixed << std::setprecision(2) << "Angle change " << angle_change
         << " scale change " << scale_change << " angle delta "
         << max_variation_angle << " scale delta " << max_variation_scale
         << " mov_delta " << max_variation_mov << endl;

    // waitKey(0);
  }

  return is_new_pose;
}

bool Tracker::getEstimatedPosition(std::vector<Point2f>& bounding_box) {
  bounding_box.resize(4, Point2f(-1, -1));

  bool is_correct = true;

  for (auto i = 0; i < m_boundingBoxUpdated.size(); ++i) {
    auto& pt = m_boundingBoxUpdated.at(i);

    if (pt.x < 0 || pt.y < 0)
      is_correct = false;
    else
      bounding_box.at(i) = pt;
  }

  return is_correct;
}

void Tracker::getTrackedPoints(std::vector<cv::Point2f>& model_pts,
                                 std::vector<cv::Point2f>& current_pts,
                                 const std::vector<int>& ids,
                                 std::vector<cv::Point2f*>& model_valid_pts,
                                 std::vector<cv::Point2f*>& current_valid_pts) {
  for (auto i = 0; i < model_pts.size(); ++i) {
    if (isPointValid(ids[i])) {
      model_valid_pts.push_back(&model_pts[i]);
      current_valid_pts.push_back(&current_pts[i]);
    }
  }
}

void Tracker::getTrackedPoints(std::vector<cv::Point2f>& model_pts,
                                 std::vector<cv::Point2f>& current_pts,
                                 const std::vector<int>& ids,
                                 std::vector<cv::Point3f>& model_valid_pts,
                                 std::vector<cv::Point2f>& current_valid_pts) {
  for (auto i = 0; i < model_pts.size(); ++i) {
    if (isPointValid(ids[i])) {
      model_valid_pts.push_back(Point3f(model_pts[i].x, model_pts[i].y, 0));
      current_valid_pts.push_back(current_pts[i]);
    }
  }
}

void Tracker::projectPointsToModel(const Point2f& model_centroid,
                                     const Point2f& upd_centroid,
                                     const float angle, const float scale,
                                     const std::vector<Point2f>& pts,
                                     std::vector<Point2f>& proj_pts) {
  Mat2f rotMat(2, 2);
  rotMat.at<float>(0, 0) = cosf(angle);
  rotMat.at<float>(0, 1) = -sinf(angle);
  rotMat.at<float>(1, 0) = sinf(angle);
  rotMat.at<float>(1, 1) = cosf(angle);

  for (auto i = 0; i < pts.size(); i++) {
    Point2f rm = pts[i] - upd_centroid;
    if (scale != 0) {
      rm.x = rm.x / scale;
      rm.y = rm.y / scale;
    }
    rm = mult(rotMat, rm);

    proj_pts.push_back(model_centroid + rm);
  }
}

}  // end namespace pinot
