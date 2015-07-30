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

#include "../include/tracker_3d.h"

#include <random>
#include <opencv2/highgui/highgui.hpp>
#include <limits>
#include <future>
#include <tbb/parallel_for.h>
#include <math.h>
#include <boost/math/constants/constants.hpp>
#include <boost/filesystem.hpp>

#include "../include/pose_estimation.h"
#include "../../utilities/include/utilities.h"
#include "../../utilities/include/draw_functions.h"
#include "../../utilities/include/profiler.h"

// TOFIX: replace all vector<T>[] with vector<T>.at() better for debugging

namespace fs = boost::filesystem;
using namespace cv;
using namespace std;
using namespace Eigen;

namespace pinot_tracker {

Tracker3D::~Tracker3D() { clear(); }

int Tracker3D::init(TrackerParams params, cv::Mat& rgb, cv::Mat& points,
                    cv::Point2d& top_left, cv::Point2d& bottom_right) {
  debug_file_.open("/home/alessandro/Debug/debug.txt");
  params_ = params;
  auto mask = getMask(rgb.rows, rgb.cols, top_left, bottom_right);

  m_focal = params_.camera_model.fx();
  m_imageCenter = Point2f(rgb.cols / 2, rgb.rows / 2);

  Mat grayImg;
  // check if image is grayscale
  checkImage(rgb, grayImg);
  // itialize structure to store keypoints
  vector<KeyPoint> keypoints;
  // find keypoints in the starting image
  m_featuresDetector.detect(grayImg, keypoints);
  m_foregroundMask = mask;
  m_currFrame = grayImg;
  m_fstFrame = grayImg;
  // store keypoints of the orginal image
  m_initKeypoints = keypoints;

  // extract descriptors of the keypoint
  m_featuresDetector.compute(m_currFrame, keypoints,
                             m_fstCube.m_faceDescriptors[FACE::FRONT]);

  if (keypoints.size() == 0) {
    cout << log_header << "not enough features extracted.\n";
    clear();
    return -1;
  }

  init_model_point_count_ = 0;
  random_device rd;
  default_random_engine engine(rd());
  uniform_int_distribution<unsigned int> uniform_dist(0, 255);

  m_isPointClustered.resize(6, vector<bool>());
  m_pointsColor.resize(6, vector<Scalar>());

  // count valid keypoints that belong to the object
  for (int kp = 0; kp < keypoints.size(); ++kp) {
    keypoints[kp].class_id = kp;
    Vec3f& tmp = points.at<Vec3f>(keypoints.at(kp).pt);
    if (mask.at<uchar>(keypoints.at(kp).pt) != 0 && tmp[2] != 0) {
      m_fstCube.m_pointStatus.at(FACE::FRONT).push_back(Status::INIT);
      m_isPointClustered.at(FACE::FRONT).push_back(true);
      init_model_point_count_++;
    } else {
      m_fstCube.m_pointStatus.at(FACE::FRONT).push_back(Status::BACKGROUND);
      m_isPointClustered.at(FACE::FRONT).push_back(false);
    }
    m_pointsColor.at(FACE::FRONT)
        .push_back(Scalar(uniform_dist(engine), uniform_dist(engine),
                          uniform_dist(engine)));

    m_fstCube.m_cloudPoints.at(FACE::FRONT)
        .push_back(Point3f(tmp[0], tmp[1], tmp[2]));
  }

  keypoints.swap(m_fstCube.m_faceKeypoints.at(FACE::FRONT));
  // compute starting centroid of the object to track
//  if (!initCentroid(m_fstCube.m_cloudPoints.at(FACE::FRONT), m_fstCube)) {
//    cout
//        << log_header
//        << "not enough valid points to initialized the center of the object \n";
//    clear();
//    return -1;
//  }
  if(init_model_point_count_ == 0)
  {
    cout << log_header << "not enough valid points to initialize the objects" << endl;
    clear();
    return -1;
  }

  bouding_cube_.setPerspective(params_.camera_model.cx(),
                               params_.camera_model.cy(),
                               params.camera_model.fx(),
                               params.camera_model.fy());
  bouding_cube_.initCube(points, top_left, bottom_right);
  m_fstCube.m_center = bouding_cube_.getCentroid();

  // initialize bounding box of the learned face
  initBBox(points);
  //
  // compute relative position of keypoints to the centroid
  initRelativePosition(m_fstCube);
  // set current centroid
  // set first and current frame
  m_currFrame = grayImg;
  m_fstFrame = grayImg;

  m_updatedModel.m_cloudPoints = m_fstCube.m_cloudPoints;
  m_updatedModel.m_pointStatus = m_fstCube.m_pointStatus;
  m_updatedModel.m_faceKeypoints = m_fstCube.m_faceKeypoints;

  // cout << "10 " << endl;

  m_centroidVotes.resize(m_fstCube.m_faceKeypoints.at(FACE::FRONT).size(),
                         Point3f(0, 0, 0));
  m_clusteredCentroidVotes.resize(
      m_fstCube.m_faceKeypoints.at(FACE::FRONT).size(), true);
  m_clusteredBorderVotes.resize(
      m_fstCube.m_faceKeypoints.at(FACE::FRONT).size(), false);

  m_fstCube.m_isLearned.at(FACE::FRONT) = true;
  m_fstCube.m_appearanceRatio.at(FACE::FRONT) = 1;

  //  cout << "11 " << endl;

  m_numFrames = 0;
  m_clusterVoteSize = 0;

  //  cout << "11-a " << endl;

  // adding to the set of learned frames the initial frame with angle 0 and
  // scale 1
  m_learnedFrames.insert(pair<int, int>(0, 0));

  points.copyTo(m_firstCloud);
  points.copyTo(m_currCloud);

  // init learning step
  Mat pov(1, 3, CV_32FC1);
  pov.at<float>(0) = 0;
  pov.at<float>(1) = 0;
  pov.at<float>(2) = 1;

  m_pointsOfView.push_back(pov);

  m_currentFaces.push_back(FACE::FRONT);

  hasAppearanceToChange = true;

  m_faceColors.push_back(Scalar(0, 255, 0));
  m_faceColors.push_back(Scalar(0, 255, 255));
  m_faceColors.push_back(Scalar(255, 0, 255));
  m_faceColors.push_back(Scalar(255, 255, 0));
  m_faceColors.push_back(Scalar(255, 0, 0));
  m_faceColors.push_back(Scalar(0, 0, 255));

  Mat tmp;
  drawLearnedFace(rgb, FACE::FRONT, tmp);
  m_learnedFaces = vector<Mat>(6, Mat(480, 640, CV_8UC3, Scalar::all(0)));
  m_learnedFaces.at(FACE::FRONT) = tmp;

  m_visibleFaces.resize(6, false);
  m_visibleFaces.at(FACE::FRONT) = true;
  m_learnedFaceVisibility.resize(6, 0);
  m_learnedFaceVisibility.at(FACE::FRONT) = 1.0f;
  m_learnedFaceMedianAngle.resize(6, numeric_limits<double>::max());
  m_learnedFaceMedianAngle.at(FACE::FRONT) = 0;

  return 0;
}

void Tracker3D::getActivePoints(std::vector<Point3f*>& points,
                                std::vector<Point3f*>& votes) {
  for (auto i = 0; i < m_votingPoints.size(); ++i) {
    votes.push_back(&m_votedPoints.at(i));
    points.push_back(&m_votingPoints.at(i));
  }
}

void Tracker3D::getCurrentPoints(
    const vector<int>& currentFaces, vector<Status*>& pointsStatus,
    vector<Point3f*>& fstPoints, vector<Point3f*>& updPoints,
    vector<KeyPoint*>& fstKeypoints, vector<KeyPoint*>& updKeypoints,
    vector<Point3f*>& relPointPos, Mat& fstDescriptors, vector<Scalar*>& colors,
    vector<bool*>& isClustered) {
  for (size_t i = 0; i < currentFaces.size(); i++) {
    int face = currentFaces.at(i);

    if (!m_fstCube.m_isLearned.at(face)) continue;

    for (int j = 0; j < m_fstCube.m_pointStatus.at(face).size(); j++) {
      pointsStatus.push_back(&m_fstCube.m_pointStatus.at(face).at(j));
      fstPoints.push_back(&m_fstCube.m_cloudPoints.at(face).at(j));
      fstKeypoints.push_back(&m_fstCube.m_faceKeypoints.at(face).at(j));
      updPoints.push_back(&m_updatedModel.m_cloudPoints.at(face).at(j));
      updKeypoints.push_back(&m_updatedModel.m_faceKeypoints.at(face).at(j));
      relPointPos.push_back(&m_fstCube.m_relativePointsPos.at(face).at(j));
      colors.push_back(&m_pointsColor.at(face).at(j));
    }

    if (fstDescriptors.empty())
      vconcat(m_fstCube.m_faceDescriptors.at(face), fstDescriptors);
    else
      vconcat(m_fstCube.m_faceDescriptors.at(face), fstDescriptors,
              fstDescriptors);
  }
  m_centroidVotes.resize(pointsStatus.size(), Point3f(0, 0, 0));
  m_clusteredCentroidVotes.resize(pointsStatus.size(), false);
  m_clusteredBorderVotes.resize(pointsStatus.size(), false);
}

void Tracker3D::clear() {
  m_initKeypoints.clear();
  m_isPointClustered.clear();
  m_pointsColor.clear();

  m_fstCube.restCube();
  m_updatedModel.restCube();

  m_learnedFrames.clear();
  m_pointsOfView.clear();
  m_currentFaces.clear();

  m_faceColors.clear();

  debug_file_.close();
}

void Tracker3D::next(const Mat& rgb, const Mat& points) {
  debug_file_ << "FRAME " << m_numFrames << "\n";

  // vector<Status> initial, optical, clustering;

  // initial = m_fstCube.m_pointStatus.at(FACE::FRONT);

  Mat grayImg;
  checkImage(rgb, grayImg);
  bool isLost = false;

  int numMatches, numTracked, numBoth;
  numTracked = 0;
  numBoth = 0;
  vector<KeyPoint> nextKeypoints;
  Mat nextDescriptors;

  vector<Status*> pointsStatus;
  vector<Point3f*> updPoints;
  vector<KeyPoint*> updKeypoint;
  vector<Point3f*> fstPoints;
  vector<KeyPoint*> fstKeypoint;
  vector<Point3f*> relPointPos;
  vector<Scalar*> colors;
  vector<bool*> isClustered;

  Mat fstDescriptors;

  auto& profiler = Profiler::getInstance();

  /****************************************************************************/
  /*                             PICK FACES                                   */
  /****************************************************************************/
  profiler->start("pick_face");
  getCurrentPoints(m_currentFaces, pointsStatus, fstPoints, updPoints,
                   fstKeypoint, updKeypoint, relPointPos, fstDescriptors,
                   colors, isClustered);
  profiler->stop("pick_face");
  cout << "0 ";
  /****************************************************************************/
  /*                             TRACKING */
  /****************************************************************************/
  profiler->start("optical_flow");
  int activeKp = trackFeatures(grayImg, points, pointsStatus, updPoints,
                               updKeypoint, numTracked, numBoth);
  profiler->stop("optical_flow");
  cout << "1 ";
  // optical = m_fstCube.m_pointStatus.at(FACE::FRONT);
  /****************************************************************************/
  /*                             ROTATION MATRIX */
  /****************************************************************************/
  profiler->start("rotation");
  Mat rotation = getRotationMatrix(fstPoints, updPoints, pointsStatus);
  profiler->stop("rotation");

  updated_rotation_ = rotation.clone();

  Mat rotation_ransac, translation_ransac;

  vector<Point2f> tracked_points;
  vector<Point3f> model_points;

  for (size_t i = 0; i < pointsStatus.size(); i++) {
    if (*pointsStatus.at(i) == Status::TRACK) {
      tracked_points.push_back(updKeypoint.at(i)->pt);
      model_points.push_back(*fstPoints.at(i));
    }
  }

  // cout << params_.camera_matrix << endl;

  profiler->start("ransac");
  vector<int> inliers;
  getPoseRansac(model_points, tracked_points, params_.ransac_method,
                params_.camera_matrix, params_.ransac_iterations,
                params_.ransac_distance, inliers, rotation_ransac,
                translation_ransac);
  profiler->stop("ransac");

  ransac_translation_ = translation_ransac.clone();
  ransac_rotation_ = rotation_ransac.clone();

  debug_file_ << "ROTATION \n";
  debug_file_ << rotation << "\n";
  debug_file_ << "RANSAC\n";
  debug_file_ << ransac_rotation_ << "\n";

  if (rotation.empty()) {
    isLost = true;
  }
  cout << "2 ";
  /*********************************************************************************************/
  /*                             CALCULATING QUATERNION */
  /*********************************************************************************************/
  double angularDist = 0;
  if (!isLost) {
    profiler->start("quaternion");
    Matrix3d eigRot;
    opencvToEigen(rotation, eigRot);
    Quaterniond q(eigRot);
    updated_quaternion_ = q;
    angularDist = getQuaternionMedianDist(m_quaternionHistory, 10, q);
    m_quaternionHistory.push_back(q);
    profiler->stop("quaternion");
  }
  cout << "3 ";
  /*********************************************************************************************/
  /*                             VOTING */
  /*********************************************************************************************/
  //  start = chrono::system_clock::now();
  if (!isLost) {
    profiler->start("voting");
    getVotes(points, pointsStatus, fstKeypoint, updKeypoint, relPointPos,
             rotation);
    profiler->stop("voting");
  }
  cout << "4 ";
  /*********************************************************************************************/
  /*                             CLUSTERING */
  /*********************************************************************************************/
  int facesSize = m_currentFaces.size();

  vector<int> indices;
  vector<vector<int>> clusters;
  //  start = chrono::system_clock::now();
  if (!isLost) {
    profiler->start("clustering");
    clusterVotesBorder(pointsStatus, m_centroidVotes, indices, clusters);
    profiler->stop("clustering");
  }
  cout << "5 ";
  /*********************************************************************************************/
  /*                             UPDATE VOTES */
  /*********************************************************************************************/
  //  start = chrono::system_clock::now();
  //  end = chrono::system_clock::now();
  //  m_partialTimes[6] +=
  //      chrono::duration_cast<chrono::milliseconds>(end - start).count();
  //  cout << "6 ";
  /*********************************************************************************************/
  /*                             UPDATE OBJECT CENTER */
  /*********************************************************************************************/
  //  start = chrono::system_clock::now();
  if (!isLost) {
    profiler->start("update");
    updateCentroid(pointsStatus, updated_rotation_);
    profiler->stop("update");
  }
  m_updatedModel.m_center = m_updatedCentroid;
  //  end = chrono::system_clock::now();
  //  m_partialTimes[7] +=
  //      chrono::duration_cast<chrono::milliseconds>(end - start).count();
  cout << "6 ";
  /****************************************************************************/
  /*                             UPDATE STATUS OF NOT CLUSTERED POINTS        */
  /****************************************************************************/
  if (!isLost) {
    profiler->start("update_status");
    updatePointsStatus(pointsStatus, m_clusteredCentroidVotes);
    profiler->stop("update_status");
  }
  cout << "7 ";
  /****************************************************************************/
  /*                            STORE CURRENT RESULT                          */
  /****************************************************************************/
  debug_file_ << "STATUS " << m_numFrames << "\n";

  m_votingPoints.clear();
  m_votedPoints.clear();
  int counter = 0;
  for (auto i = 0; i < m_centroidVotes.size(); ++i) {
    Status s = *pointsStatus.at(i);
    if (s == Status::BOTH || s == Status::TRACK || s == Status::MATCH) {
      m_votingPoints.push_back(*updPoints.at(i));
      m_votedPoints.push_back(m_centroidVotes.at(i));
      counter++;
      // debug_file_ << *updPoints.at(i) << " vote " << m_centroidVotes.at(i)
      //            << "\n";
    }
  }
  /****************************************************************************/
  /*                            COMPUTING VISIBILITY                          */
  /****************************************************************************/
  vector<bool> isFaceVisible(6, false);
  vector<float> visibilityRatio(6, 0);
  vector<bool> isVisibleEig(6, false);
  vector<float> visibilityEig(6, 0);
  //  start = chrono::system_clock::now();
  if (!isLost) {
    profiler->start("visibility");
    calculateVisibilityEigen(rotation, m_fstCube, isFaceVisible,
                             visibilityRatio);
    profiler->stop("visibility");
  }
  visibility_ratio_ = visibilityRatio;
  cout << "8 ";
  /****************************************************************************/
  /*                             LEARN NEW APPEARANCE                         */
  /****************************************************************************/
  bool learning = false;
  //  start = chrono::system_clock::now();
  if (!isLost && isAppearanceNew(rotation) && angularDist < 4.00) {
    learning =
        learnFrame(rgb, points, isFaceVisible, visibilityRatio, rotation);
  }

  bool newLearning = false;
  int ftl = -1;
  if (!isLost) {
    newLearning = isCurrentApperanceToLearn(visibilityRatio, angularDist, ftl);
    if (newLearning) {
      learnFrame(rgb, points, ftl, rotation);
    }
  }
  cout << "9 ";
  /*********************************************************************************************/
  /*                            CHOOSE VISIBLE FACES */
  /*********************************************************************************************/
  vector<bool> oldFaces = m_visibleFaces;
  //  start = chrono::system_clock::now();
  if (!isLost && angularDist < 15.0f) {
    vector<int> visibleFaces;

    // m_debugFile << "Visible faces: ";

    stringstream ss;

    m_currentFaces.clear();
    for (int i = 0; i < 6; ++i) {
      if (visibilityRatio.at(i) > 0.6) {
        // m_debugFile << faceToString(i) << " ";
        m_currentFaces.push_back(i);
        m_visibleFaces.at(i) = true;
      } else {
        m_visibleFaces.at(i) = false;
      }
    }
  }
  cout << "10 ";
  /*********************************************************************************************/
  /*                             EXTRACTION */
  /*********************************************************************************************/
  //  start = chrono::system_clock::now();
  profiler->start("extraction");
  m_featuresDetector.detect(grayImg, nextKeypoints);
  m_featuresDetector.compute(grayImg, nextKeypoints, nextDescriptors);
  profiler->stop("extraction");
  cout << "11 ";
  /*********************************************************************************************/
  /*                             MATCHING */
  /*********************************************************************************************/
  //  start = chrono::system_clock::now();
  if (!isLost) {
    bool statusChanged = false;
    for (size_t i = 0; i < oldFaces.size(); i++) {
      if (m_visibleFaces[i] != oldFaces[i]) {
        statusChanged = true;
        break;
      }
    }

    if (!statusChanged) {
      // numMatches =
      //    matchFeaturesCustom(fstDescriptors, fstKeypoint, nextDescriptors,
      //                        nextKeypoints, updKeypoint, pointsStatus);
      // TOFIX:check here, seems wrong
      profiler->start("matching");
      if (m_currentFaces.size() > 0) {
        for (size_t i = 0; i < m_currentFaces.size(); i++) {
          int face = m_currentFaces[i];
          fstKeypoint.clear();
          updKeypoint.clear();
          pointsStatus.clear();
          Mat& descriptors = m_fstCube.m_faceDescriptors.at(face);

          for (int j = 0; j < m_fstCube.m_faceKeypoints.at(face).size(); j++) {
            fstKeypoint.push_back(&m_fstCube.m_faceKeypoints.at(face).at(j));
            pointsStatus.push_back(&m_fstCube.m_pointStatus.at(face).at(j));
            updKeypoint.push_back(
                &m_updatedModel.m_faceKeypoints.at(face).at(j));
          }

          numMatches =
              matchFeaturesCustom(descriptors, fstKeypoint, nextDescriptors,
                                  nextKeypoints, updKeypoint, pointsStatus);
        }
      }
      profiler->stop("matching");
    } else {
    }
  }
  cout << "14 ";
  /****************************************************************************/
  /*                            RECOVERY                                      */
  /****************************************************************************/
  //  start = chrono::system_clock::now();
  if (isLost) {
    int faceFound, matchedNum;

    bool found = findObject(m_fstCube, m_updatedModel, nextDescriptors,
                            nextKeypoints, faceFound, matchedNum);

    if (found) {
      m_currentFaces.clear();
      m_currentFaces.push_back(faceFound);
      isLost = false;
    }
  }
  cout << "15 ";
  /*********************************************************************************************/
  /*                            SAVING */
  /*********************************************************************************************/
  //  start = chrono::system_clock::now();
  m_currFrame = grayImg;
  points.copyTo(m_currCloud);
  m_numFrames++;
  cout << "16\n";
}

void Tracker3D::close() {
  cout << "Crushing here 1" << endl;

  float frames = static_cast<float>(m_numFrames);

  //  for (size_t i = 0; i < m_partialTimes.size(); ++i) {
  //    m_timeFile << "Step " << i << ": " << (m_partialTimes[i] / frames) <<
  // "\n";
  //  }

  //  m_timeFile << "Average time per frame: " << (m_frameTime / frames) <<
  // "\n";
  //  m_timeFile << "FPS: " << 1000 / (m_frameTime / frames) << "\n";
  //  m_timeFile << "FPS no draw: "
  //             << 1000 / ((m_frameTime - m_partialTimes[10]) / frames);

  //  m_timeFile.close();

  //  // m_matrixFile << "\n Quaternions: \n" << m_quaterionStream.str();

  //  // m_matrixFile.close();

  //  m_resultFile.close();

  //  cout << "Crushing here 2" << endl;
}

int Tracker3D::matchFeaturesCustom(const Mat& fstDescriptors,
                                   const vector<KeyPoint*>& fstKeypoints,
                                   const Mat& nextDescriptors,
                                   const vector<KeyPoint>& extractedKeypoints,
                                   vector<KeyPoint*>& updKeypoints,
                                   vector<Status*>& pointsStatus) {
  vector<vector<DMatch>> matches;

  Mat currDescriptors;

  m_customMatcher.match(nextDescriptors, fstDescriptors, 2, matches);

  int matchesCount = 0;


  //TODO: if the 3d point has no valid depth should not be updated!!
  for (size_t i = 0; i < matches.size(); i++) {
    int queryId = matches[i][0].queryIdx;
    int trainId = matches[i][0].trainIdx;

    if (queryId < 0 && queryId >= extractedKeypoints.size()) continue;

    if (trainId < 0 && trainId >= fstKeypoints.size()) continue;

    float confidence = 1 - (matches[i][0].distance / 512.0);
    float ratio = matches[i][0].distance / matches[i][1].distance;

    Status* s = pointsStatus[trainId];

    if (*s == Status::BACKGROUND)
      continue;
    else if (confidence >= 0.80f && ratio <= 0.8) {
      if (*s == Status::TRACK) {
        *pointsStatus[trainId] = Status::BOTH;
        // updKeypoints[trainId]->pt = extractedKeypoints[queryId].pt;
      } else if (*s == Status::LOST || *s == Status::NOCLUSTER) {
        *pointsStatus[trainId] = Status::MATCH;
        updKeypoints[trainId]->pt = extractedKeypoints[queryId].pt;
      }

      matchesCount++;
    }
  }
  return matchesCount;
}

int Tracker3D::trackFeatures(const Mat& grayImg, const Mat& cloud,
                             vector<Status*>& keypointStatus,
                             vector<Point3f*>& updPoints,
                             vector<KeyPoint*>& updKeypoints, int& trackedCount,
                             int& bothCount) {
  vector<Point2f> nextPoints;
  vector<Point2f> currPoints;
  vector<int> ids;
  vector<Point2f> prevPoints;

  vector<uchar> prevStatus;
  vector<float> prevErrors;

  for (size_t i = 0; i < updKeypoints.size(); i++) {
    if (*keypointStatus[i] == Status::TRACK ||
        *keypointStatus[i] == Status::BOTH ||
        *keypointStatus[i] == Status::MATCH ||
        *keypointStatus[i] == Status::INIT ||
        *keypointStatus[i] == Status::NOCLUSTER) {
      currPoints.push_back(updKeypoints[i]->pt);
      ids.push_back(i);
    }
  }

  if (currPoints.empty()) return 0;

  calcOpticalFlowPyrLK(m_currFrame, grayImg, currPoints, nextPoints, m_status,
                       m_errors);

  calcOpticalFlowPyrLK(grayImg, m_currFrame, nextPoints, prevPoints, m_status,
                       prevErrors);

  int activeKeypoints = 0;

  // m_debugFile << "Updating cloud:\n";

  // debug_file_ << "OPTICAL FLOW: \n";

  for (int i = 0; i < nextPoints.size(); ++i) {
    float error = getDistance(currPoints[i], prevPoints[i]);

    Status* s = keypointStatus[ids[i]];

    if (m_status[i] == 1 && error < 20) {
      int& id = ids[i];

      // debug_file_ << toString(*keypointStatus.at(id)) << " ";

      // FIXME: potential value of 0 in the cloud due to errors in the kinect
      // is this the case?
      const Vec3f& tmp = cloud.at<Vec3f>(nextPoints[i]);

      if (*s == Status::MATCH) {
        *keypointStatus[id] = Status::TRACK;
        bothCount++;
        activeKeypoints++;
      } else if (*s == Status::LOST)
        *keypointStatus[id] = Status::LOST;
      else if (*s == Status::NOCLUSTER) {
        *keypointStatus[id] = Status::NOCLUSTER;
        trackedCount++;
        activeKeypoints++;
      } else if (tmp[2] == 0) {
        // FIXME: ugly fix to manage the error in kinect sensor
        *keypointStatus[id] = Status::LOST;
      } else {
        *keypointStatus[id] = Status::TRACK;
        trackedCount++;
        activeKeypoints++;
      }

      // m_debugFile << toString(m_firstFrameCloud[id])  << " : "
      //	 << toString(m_updatedFrameCloud[id]) << " -> " << toString(tmp)

      //<< "\n";

      // m_trackedPoints[id] = nextPoints[i];
      updKeypoints[id]->pt = nextPoints[i];
      *updPoints[id] = tmp;
      // debug_file_ << toString(*keypointStatus.at(id)) << " " <<
      // *updPoints[id]
      //            << "\n";
    } else {
      *keypointStatus[ids[i]] = Status::LOST;
    }
  }

  // debug_file_ << "\n";

  return activeKeypoints;
}

void Tracker3D::checkImage(const Mat& src, Mat& gray) {
  if (src.channels() > 1)
    cvtColor(src, gray, CV_BGR2GRAY);
  else
    gray = src;
}

cv::Mat Tracker3D::getRotationMatrix(const vector<Point3f*>& initPoints,
                                     const vector<Point3f*>& updPoints,
                                     const vector<Status*>& pointsStatus) {
  Point3f fstMean(0, 0, 0);
  Point3f scdMean(0, 0, 0);
  int validCount = 0;
  for (size_t i = 0; i < pointsStatus.size(); ++i) {
    if (isKeypointValid(*pointsStatus[i]) && initPoints[i]->z != 0 &&
        updPoints[i]->z != 0) {
      validCount++;
    }
  }

  if (validCount > 0) {
    Mat currPoints(validCount, 3, CV_32FC1);
    Mat fstPoints(validCount, 3, CV_32FC1);
    validCount = 0;
    for (size_t i = 0; i < pointsStatus.size(); ++i) {
      if (isKeypointValid(*pointsStatus[i]) && initPoints[i]->z != 0 &&
          updPoints[i]->z != 0) {
        fstPoints.at<float>(validCount, 0) = initPoints[i]->x;
        fstPoints.at<float>(validCount, 1) = initPoints[i]->y;
        fstPoints.at<float>(validCount, 2) = initPoints[i]->z;

        currPoints.at<float>(validCount, 0) = updPoints[i]->x;
        currPoints.at<float>(validCount, 1) = updPoints[i]->y;
        currPoints.at<float>(validCount, 2) = updPoints[i]->z;

        fstMean += *initPoints[i];
        scdMean += *updPoints[i];

        validCount++;
      }
    }
    fstMean.x = fstMean.x / static_cast<float>(validCount);
    fstMean.y = fstMean.y / static_cast<float>(validCount);
    fstMean.z = fstMean.z / static_cast<float>(validCount);
    scdMean.x = scdMean.x / static_cast<float>(validCount);
    scdMean.y = scdMean.y / static_cast<float>(validCount);
    scdMean.z = scdMean.z / static_cast<float>(validCount);

    vector<float> cb = {scdMean.x, scdMean.y, scdMean.z};
    vector<float> ca = {m_fstCube.m_center.x, m_fstCube.m_center.y,
                        m_fstCube.m_center.z};

    return getRigidTransform(currPoints, fstPoints, cb, ca).inv();
  } else {
    return Mat();
  }
}

cv::Mat Tracker3D::getRotationMatrixDebug(const Mat& rgbImg,
                                          const vector<Point3f*>& initPoints,
                                          const vector<Point3f*>& updPoints,
                                          const vector<Status*>& pointsStatus) {
  // m_debugFile << "Computing rotation matrix for frame: " << m_numFrames <<
  // "\n\n";
  Mat debugImg;
  rgbImg.copyTo(debugImg);

  Point3f fstMean(0, 0, 0);
  Point3f scdMean(0, 0, 0);
  int validCount = 0;
  for (size_t i = 0; i < pointsStatus.size(); ++i) {
    if (isKeypointValid(*pointsStatus[i]) && initPoints[i]->z != 0 &&
        updPoints[i]->z != 0) {
      validCount++;
      // m_debugFile << i << ": " <<toString(*initPoints[i]) << " "
      //	        << toString(*updPoints[i]) << "\n";
      Point2f tmp;
      projectPoint(m_focal, m_imageCenter, *updPoints[i], tmp);
      circle(debugImg, tmp, 5, Scalar(255, 0, 0), 3, 1);
    }
  }

  //  stringstream imgName;
  //  imgName << m_configFile.m_dstPath << "debug/" << m_numFrames << ".png";
  //  imwrite(imgName.str(), debugImg);

  if (validCount > 0) {
    Mat currPoints(validCount, 3, CV_32FC1);
    Mat fstPoints(validCount, 3, CV_32FC1);
    validCount = 0;
    for (size_t i = 0; i < pointsStatus.size(); ++i) {
      if (isKeypointValid(*pointsStatus[i]) && initPoints[i]->z != 0 &&
          updPoints[i]->z != 0) {
        fstPoints.at<float>(validCount, 0) = initPoints[i]->x;
        fstPoints.at<float>(validCount, 1) = initPoints[i]->y;
        fstPoints.at<float>(validCount, 2) = initPoints[i]->z;

        currPoints.at<float>(validCount, 0) = updPoints[i]->x;
        currPoints.at<float>(validCount, 1) = updPoints[i]->y;
        currPoints.at<float>(validCount, 2) = updPoints[i]->z;

        fstMean += *initPoints[i];
        scdMean += *updPoints[i];

        validCount++;
      }
    }
    fstMean.x = fstMean.x / static_cast<float>(validCount);
    fstMean.y = fstMean.y / static_cast<float>(validCount);
    fstMean.z = fstMean.z / static_cast<float>(validCount);
    scdMean.x = scdMean.x / static_cast<float>(validCount);
    scdMean.y = scdMean.y / static_cast<float>(validCount);
    scdMean.z = scdMean.z / static_cast<float>(validCount);

    vector<float> cb = {scdMean.x, scdMean.y, scdMean.z};
    vector<float> ca = {m_fstCube.m_center.x, m_fstCube.m_center.y,
                        m_fstCube.m_center.z};

    return getRigidTransform(currPoints, fstPoints, cb, ca).inv();
  } else {
    return Mat();
  }
}

void Tracker3D::getVotes(const cv::Mat& cloud, vector<Status*> pointsStatus,
                         vector<KeyPoint*>& fstKeypoints,
                         vector<KeyPoint*>& updKeypoints,
                         vector<Point3f*>& relPointPos, const Mat& rotation) {
  for (size_t i = 0; i < fstKeypoints.size(); i++) {
    if (isKeypointValid(*pointsStatus[i])) {
      const Point2f& p = updKeypoints[i]->pt;
      const Vec3f& tmp = cloud.at<Vec3f>(p);
      Point3f a = Point3f(tmp[0], tmp[1], tmp[2]);
      const Point3f& rm = *relPointPos[i];
      Point3f rmUpdated;
      rotatePoint(rm, rotation, rmUpdated);

      m_centroidVotes[i] = a - rmUpdated;
    }
  }
}

void Tracker3D::clusterVotes(vector<Status>& keypointStatus) {
  DBScanClustering<Point3f*> clusterer;

  vector<Point3f*> votes;
  vector<unsigned int> indices;

  for (unsigned int i = 0; i < keypointStatus.size(); i++) {
    if (isKeypointTracked(keypointStatus[i])) {
      votes.push_back(&m_centroidVotes[i]);
      indices.push_back(i);
    }
  }

  clusterer.clusterPoints(
      &votes, params_.eps, params_.min_points, [](Point3f* a, Point3f* b) {
        return sqrt(pow(a->x - b->x, 2) + pow(a->y - b->y, 2) +
                    pow(a->z - b->z, 2));
      });

  auto res = clusterer.getClusters();
  // std::cout << "Size of clusters " << res.size() << "\n" << std::endl;
  int maxVal = 0;
  int maxId = -1;

  for (size_t i = 0; i < res.size(); i++) {
    // std::cout << "Size of cluster " << i << ":" << res[i].size() << "\n" <<
    // std::endl;
    if (res[i].size() > maxVal) {
      maxVal = res[i].size();
      maxId = i;
    }
  }

  // std::cout << "Picked cluster " << maxId << " size " << res[maxId].size() <<
  // "\n" << std::endl;

  vector<bool> clustered(m_clusteredCentroidVotes.size(), false);

  m_clusterVoteSize = 0;

  if (maxId > -1) {
    for (size_t i = 0; i < res[maxId].size(); i++) {
      unsigned int& id = indices[res[maxId][i]];
      clustered[id] = true;
    }

    m_clusterVoteSize = res[maxId].size();
  }

  m_clusteredCentroidVotes.swap(clustered);
}

void Tracker3D::clusterVotesBorder(vector<Status*>& keypointStatus,
                                   vector<Point3f>& centroidVotes,
                                   vector<int>& indices,
                                   vector<vector<int>>& clusters) {
  DBScanClustering<Point3f*> clusterer;

  vector<Point3f*> votes;
  indices.clear();

  for (unsigned int i = 0; i < keypointStatus.size(); i++) {
    if (isKeypointTracked(*keypointStatus[i])) {
      votes.push_back(&centroidVotes[i]);
      indices.push_back(i);
    }
  }

  clusterer.clusterPoints(
      &votes, params_.eps, params_.min_points, [](Point3f* a, Point3f* b) {
        return sqrt(pow(a->x - b->x, 2) + pow(a->y - b->y, 2) +
                    pow(a->z - b->z, 2));
      });

  vector<bool> border;
  clusterer.getBorderClusters(clusters, border);

  int maxVal = 0;
  int maxId = -1;

  for (size_t i = 0; i < clusters.size(); i++) {
    if (clusters[i].size() > maxVal) {
      maxVal = clusters[i].size();
      maxId = i;
    }
  }

  vector<bool> clustered(m_clusteredCentroidVotes.size(), false);
  vector<bool> borderPoints(m_clusteredBorderVotes.size(), false);

  // m_debugFile << "Clustered points: \n";

  if (maxId > -1) {
    for (size_t i = 0; i < clusters[maxId].size(); i++) {
      int& id = indices[clusters[maxId][i]];
      clustered[id] = true;
      borderPoints[id] = border[clusters[maxId][i]];
      // m_debugFile << id << " ";
    }
  }

  // m_debugFile << "\n";

  m_clusteredCentroidVotes.swap(clustered);
  m_clusteredBorderVotes.swap(borderPoints);
}

bool Tracker3D::initCentroid(const vector<Point3f>& points, ObjectModel& cube) {
  const vector<Status>& status = cube.m_pointStatus.at(FACE::FRONT);

  Point3f& centroid = cube.m_center;
  centroid.x = 0;
  centroid.y = 0;
  centroid.z = 0;
  int validKp = 0;

  for (int i = 0; i < points.size(); ++i) {
    const Point3f& tmp = points.at(i);

    if (tmp.z != 0 && isKeypointValid(status.at(i))) {
      centroid += tmp;
      validKp++;
    }
  }

  if (validKp == 0) return false;

  centroid.x = centroid.x / static_cast<float>(validKp);
  centroid.y = centroid.y / static_cast<float>(validKp);
  centroid.z = centroid.z / static_cast<float>(validKp);

  return true;
}

void Tracker3D::initRelativePosition(ObjectModel& cube) {
  const vector<Status>& status = cube.m_pointStatus[FACE::FRONT];
  const vector<Point3f>& points = cube.m_cloudPoints[FACE::FRONT];
  vector<Point3f>& relativePointsPos = cube.m_relativePointsPos[FACE::FRONT];

  relativePointsPos.resize(points.size(), Point3f(0, 0, 0));

  for (size_t i = 0; i < points.size(); i++) {
    const Point3f& tmp = points[i];
    if (tmp.z != 0 && isKeypointValid(status[i])) {
      relativePointsPos[i] = tmp - m_fstCube.m_center;
    }
  }
}

void Tracker3D::updateCentroid(const vector<Status*>& keypointStatus,
                               const Mat& rotation) {
  m_updatedCentroid.x = 0;
  m_updatedCentroid.y = 0;
  m_updatedCentroid.z = 0;

  int validKp = 0;

  for (int i = 0; i < m_centroidVotes.size(); ++i) {
    if (isKeypointValid(*keypointStatus[i]) && m_clusteredCentroidVotes[i]) {
      m_updatedCentroid += m_centroidVotes[i];
      validKp++;
    }
  }

  m_updatedCentroid.x = m_updatedCentroid.x / static_cast<float>(validKp);
  m_updatedCentroid.y = m_updatedCentroid.y / static_cast<float>(validKp);
  m_updatedCentroid.z = m_updatedCentroid.z / static_cast<float>(validKp);

  m_validKeypoints = validKp;

  vector<Point3f> updatedPoints;
  rotateBBox(m_fstCube.m_relativeDistFront, rotation, updatedPoints);

  for (int i = 0; i < m_fstCube.m_pointsFront.size(); ++i) {
    m_updatedModel.m_pointsFront.at(i) = m_updatedCentroid + updatedPoints.at(i);
  }

  updatedPoints.clear();
  rotateBBox(m_fstCube.m_relativeDistBack, rotation, updatedPoints);

  for (int i = 0; i < m_fstCube.m_pointsBack.size(); ++i) {
    m_updatedModel.m_pointsBack.at(i) = m_updatedCentroid + updatedPoints.at(i);
  }
}

void Tracker3D::updatePointsStatus(vector<Status*>& pointsStatus,
                                   vector<bool>& isClustered) {
  for (int i = 0; i < pointsStatus.size(); i++) {
    Status* s = pointsStatus[i];
    if ((*s == Status::INIT || *s == Status::TRACK || *s == Status::MATCH ||
         *s == Status::BOTH) &&
        !isClustered[i]) {
      *s = Status::LOST;
    }
  }
}

void Tracker3D::initBBox(const Mat& cloud) {
  float minX = numeric_limits<float>::max();
  float minY = numeric_limits<float>::max();
  float maxX = -numeric_limits<float>::max();
  float maxY = -numeric_limits<int>::max();
  ;

  for (int i = 0; i < m_foregroundMask.cols; i++) {
    for (int y = 0; y < m_foregroundMask.rows; y++) {
      if (m_foregroundMask.at<uchar>(y, i) != 0) {
        const Vec3f& tmp = cloud.at<Vec3f>(y, i);

        if (tmp[0] == 0 || tmp[1] == 0) continue;

        minX = min(tmp[0], minX);
        minY = min(tmp[1], minY);
        maxX = max(tmp[0], maxX);
        maxY = max(tmp[1], maxY);
      }
    }
  }

  maxY = maxY;
  minY = minY;

  float deltaX = maxX - minX;
  float deltaY = maxY - minY;
  float depth = min(deltaX, deltaY);
  float relativeD = (depth / 2.0f);

  float cx = m_fstCube.m_center.x;
  float cy = m_fstCube.m_center.y;
  float cz = m_fstCube.m_center.z;
  m_fstCube.m_center.z += relativeD;
  m_updatedCentroid = m_fstCube.m_center;

  m_fstCube.m_relativeDistFront.push_back(
      Point3f(minX - cx, minY - cy, -relativeD));
  m_fstCube.m_relativeDistFront.push_back(
      Point3f(maxX - cx, minY - cy, -relativeD));
  m_fstCube.m_relativeDistFront.push_back(
      Point3f(maxX - cx, maxY - cy, -relativeD));
  m_fstCube.m_relativeDistFront.push_back(
      Point3f(minX - cx, maxY - cy, -relativeD));

  m_fstCube.m_pointsFront.push_back(Point3f(minX, minY, cz));
  m_fstCube.m_pointsFront.push_back(Point3f(maxX, minY, cz));
  m_fstCube.m_pointsFront.push_back(Point3f(maxX, maxY, cz));
  m_fstCube.m_pointsFront.push_back(Point3f(minX, maxY, cz));

  float posZ = cz + depth;
  m_fstCube.m_pointsBack.push_back(Point3f(minX, minY, posZ));
  m_fstCube.m_pointsBack.push_back(Point3f(maxX, minY, posZ));
  m_fstCube.m_pointsBack.push_back(Point3f(maxX, maxY, posZ));
  m_fstCube.m_pointsBack.push_back(Point3f(minX, maxY, posZ));

  m_fstCube.m_relativeDistBack.push_back(
      Point3f(minX - cx, minY - cy, relativeD));
  m_fstCube.m_relativeDistBack.push_back(
      Point3f(maxX - cx, minY - cy, relativeD));
  m_fstCube.m_relativeDistBack.push_back(
      Point3f(maxX - cx, maxY - cy, relativeD));
  m_fstCube.m_relativeDistBack.push_back(
      Point3f(minX - cx, maxY - cy, relativeD));

  m_updatedModel = m_fstCube;
}

bool Tracker3D::isAppearanceNew(const Mat& rotation) {
  float minAngle = numeric_limits<float>::max();

  Mat pov(1, 3, CV_32FC1);
  pov.at<float>(0) = rotation.at<float>(0, 2);
  pov.at<float>(1) = rotation.at<float>(1, 2);
  pov.at<float>(2) = rotation.at<float>(2, 2);

  for (int i = 0; i < m_pointsOfView.size(); ++i) {
    float angle = acosf(pov.dot(m_pointsOfView[i])) * 180 /
                  boost::math::constants::pi<float>();
    ;
    minAngle = min(minAngle, angle);
  }

  if (minAngle >= 30) {
    m_pointsOfView.push_back(pov);
    return true;
  } else {
    return false;
  }
}

bool Tracker3D::isCurrentApperanceToLearn(const vector<float>& visibilityRatio,
                                          const double& medianAngle,
                                          int& faceToLearn) {
  if (medianAngle > 12.0) return false;

  bool toLearn = false;
  for (size_t i = 0; i < 6; i++) {
    if (visibilityRatio[i] > 0.5 && visibilityRatio[i] < 0.85) {
      if (visibilityRatio[i] >= m_learnedFaceVisibility[i] + 0.1f) {
        faceToLearn = i;
        toLearn = true;
      } else if (medianAngle <= m_learnedFaceMedianAngle[i] / 2.0f) {
        faceToLearn = i;
        toLearn = true;
      }
    } else if (visibilityRatio[i] > 0.85) {
      if (visibilityRatio[i] >= m_learnedFaceVisibility[i] + 0.05f) {
        faceToLearn = i;
        toLearn = true;
      } else if (medianAngle <= m_learnedFaceMedianAngle[i] / 2.0f) {
        faceToLearn = i;
        toLearn = true;
      }
    }
  }

  if (toLearn) {
    m_learnedFaceVisibility[faceToLearn] = visibilityRatio[faceToLearn];
    m_learnedFaceMedianAngle[faceToLearn] = medianAngle;

    //    m_debugFile << m_numFrames << ": v[";
    //    for (size_t i = 0; i < 6; i++) {
    //      m_debugFile << m_learnedFaceVisibility[i] << " ";
    //    }
    //    m_debugFile << "] a[";
    //    for (size_t i = 0; i < 6; i++) {
    //      m_debugFile << m_learnedFaceMedianAngle[i] << " ";
    //    }
    //    m_debugFile << "]\n";
  }

  return toLearn;
}

bool Tracker3D::learnFrame(const Mat& rgb, const Mat& cloud,
                           const vector<bool>& isFaceVisible,
                           const vector<float>& visibilityRatio,
                           const Mat& rotation) {
  bool hasLearned = false;
  int faceToLearn = -1;
  int maxRatio = 0;
  for (int i = 0; i < 6; ++i) {
    if (isFaceVisible[i] && visibilityRatio[i] > 0.5 &&
        visibilityRatio[i] > maxRatio &&
        m_fstCube.m_appearanceRatio[i] < visibilityRatio[i]) {
      faceToLearn = i;
      maxRatio = visibilityRatio[i];
    }
  }

  if (faceToLearn > -1 && faceToLearn < 6 && faceToLearn != FACE::FRONT) {
    Mat result;
    Mat1b mask(rgb.rows, rgb.cols, static_cast<uchar>(0));
    Mat3b maskResult(rgb.rows, rgb.cols, Vec3b(0, 0, 0));
    rgb.copyTo(result);
    drawBoundingCube(m_updatedCentroid, m_updatedModel.m_pointsFront,
                     m_updatedModel.m_pointsBack, m_focal, m_imageCenter,
                     result);
    vector<Point3f> facePoints = m_updatedModel.getFacePoints(faceToLearn);

    Point2f a, b, c, d;
    bool isValA, isValB, isValC, isValD;

    isValA = projectPoint(m_focal, m_imageCenter, facePoints[0], a);
    isValB = projectPoint(m_focal, m_imageCenter, facePoints[1], b);
    isValC = projectPoint(m_focal, m_imageCenter, facePoints[2], c);
    isValD = projectPoint(m_focal, m_imageCenter, facePoints[3], d);

    if (isValA && isValB && isValC && isValD) {
      drawTriangleMask(a, b, c, mask);
      drawTriangleMask(a, c, d, mask);
      learnFace(mask, rgb, cloud, rotation, faceToLearn, m_fstCube,
                m_updatedModel);

      drawTriangle(a, b, c, Scalar(0, 255, 0), 0.5, result);
      drawTriangle(a, c, d, Scalar(0, 255, 0), 0.5, result);

      rgb.copyTo(maskResult, mask);

      //      int imageCount =
      //          getFilesCount(m_configFile.m_dstPath + "learn3d",
      //          m_resultName);

      //      stringstream imgName;
      //      imgName.precision(1);
      //      // imgName << "[" << rotation.at<float>(0, 2) << "," <<
      //      // rotation.at<float>(1, 2) <<
      //      //	"," << rotation.at<float>(2, 2) << "]";
      //      imgName << m_resultName << "_" << imageCount;

      //      imwrite(m_configFile.m_dstPath + "learn3d/" + imgName.str() +
      //      ".png",
      //              result);
      //      imwrite(m_configFile.m_dstPath + "learn3d/" + imgName.str() +
      //      "_mask.png",
      //              maskResult);

      m_fstCube.m_appearanceRatio[faceToLearn] = visibilityRatio[faceToLearn];
      m_fstCube.m_isLearned[faceToLearn] = true;

      hasLearned = true;

      Mat tmp;
      drawLearnedFace(rgb, faceToLearn, tmp);
      m_learnedFaces[faceToLearn] = tmp;
    }
  }

  return hasLearned;
}

void Tracker3D::learnFrame(const Mat& rgb, const Mat& cloud,
                           const int& faceToLearn, const Mat& rotation) {
  Mat result;
  Mat1b mask(rgb.rows, rgb.cols, static_cast<uchar>(0));
  Mat3b maskResult(rgb.rows, rgb.cols, Vec3b(0, 0, 0));
  rgb.copyTo(result);
  drawBoundingCube(m_updatedCentroid, m_updatedModel.m_pointsFront,
                   m_updatedModel.m_pointsBack, m_focal, m_imageCenter, result);
  vector<Point3f> facePoints = m_updatedModel.getFacePoints(faceToLearn);

  Point2f a, b, c, d;
  bool isValA, isValB, isValC, isValD;

  isValA = projectPoint(m_focal, m_imageCenter, facePoints[0], a);
  isValB = projectPoint(m_focal, m_imageCenter, facePoints[1], b);
  isValC = projectPoint(m_focal, m_imageCenter, facePoints[2], c);
  isValD = projectPoint(m_focal, m_imageCenter, facePoints[3], d);

  if (isValA && isValB && isValC && isValD) {
    drawTriangleMask(a, b, c, mask);
    drawTriangleMask(a, c, d, mask);

    drawTriangle(a, b, c, Scalar(0, 255, 0), 0.5, result);
    drawTriangle(a, c, d, Scalar(0, 255, 0), 0.5, result);

    int count = learnFaceDebug(mask, rgb, cloud, rotation, faceToLearn,
                               m_fstCube, m_updatedModel, result);

    rgb.copyTo(maskResult, mask);

    //    int imageCount =
    //        getFilesCount(m_configFile.m_dstPath + "learn3d", m_resultName);

    //    stringstream imgName;
    //    imgName.precision(1);
    //    imgName << m_resultName << "_[" <<
    //    m_learnedFaceVisibility[faceToLearn]
    //            << "," << m_learnedFaceMedianAngle[faceToLearn] << "]";

    //    stringstream info;
    //    info.precision(2);

    //    info << std::fixed << "V: " << m_learnedFaceVisibility[faceToLearn]
    //         << " A: " << m_learnedFaceMedianAngle[faceToLearn] << " KP: " <<
    //         count
    //         << "/" << m_fstCube.m_faceKeypoints[faceToLearn].size();

    //    drawInformationHeader(Point2f(10, 10), info.str(), 0.5, 640, 30,
    //    result);

    //    imwrite(m_configFile.m_dstPath + "learn3d/" + imgName.str() + ".png",
    //            result);
    //    imwrite(m_configFile.m_dstPath + "learn3d/" + imgName.str() +
    //    "_mask.png",
    //            maskResult);

    m_fstCube.m_appearanceRatio[faceToLearn] =
        m_learnedFaceVisibility[faceToLearn];
    m_fstCube.m_isLearned[faceToLearn] = true;

    debugLearnedModel(rgb, faceToLearn, rotation);

    // Mat tmp;
    // drawLearnedFace(rgb, faceToLearn, tmp);
    m_learnedFaces[faceToLearn] = result;
  }
}

void Tracker3D::calculateVisibility(const Mat& rotation,
                                    const ObjectModel& fstCube,
                                    vector<bool>& isFaceVisible,
                                    vector<float>& visibilityRatio) {
  Mat pov(1, 3, CV_32FC1);
  pov.at<float>(0) = rotation.at<float>(0, 2);
  pov.at<float>(1) = rotation.at<float>(1, 2);
  pov.at<float>(2) = rotation.at<float>(2, 2);

  Mat normals_T;
  transpose(m_fstCube.m_faceNormals, normals_T);
  Mat res = pov * normals_T;

  for (size_t i = 0; i < 6; ++i) {
    if (res.at<float>(i) > 0)
      isFaceVisible[i] = true;
    else
      isFaceVisible[i] = false;

    visibilityRatio[i] = res.at<float>(i);
  }
}

void Tracker3D::calculateVisibilityEigen(const Mat& rotation,
                                         const ObjectModel& fstCube,
                                         vector<bool>& isFaceVisible,
                                         vector<float>& visibilityRatio) {
  // FIXME: calculation is non correct for perspective camera
  VectorXd pov(3);
  pov(0) = rotation.at<float>(0, 2);
  pov(1) = rotation.at<float>(1, 2);
  pov(2) = rotation.at<float>(2, 2);

  Matrix3d rot;
  rot(0, 0) = rotation.at<float>(0, 0);
  rot(0, 1) = rotation.at<float>(0, 1);
  rot(0, 2) = rotation.at<float>(0, 2);
  rot(1, 0) = rotation.at<float>(1, 0);
  rot(1, 1) = rotation.at<float>(1, 1);
  rot(1, 2) = rotation.at<float>(1, 2);
  rot(2, 0) = rotation.at<float>(2, 0);
  rot(2, 1) = rotation.at<float>(2, 1);
  rot(2, 2) = rotation.at<float>(2, 2);

  MatrixXd res = rot * m_fstCube.m_eigNormals.transpose();

  for (size_t i = 0; i < 6; ++i) {
    if (res(2, i) > 0)
      isFaceVisible[i] = true;
    else
      isFaceVisible[i] = false;

    visibilityRatio[i] = res(2, i);
  }
}

void Tracker3D::learnFace(const Mat1b& mask, const Mat& rgb, const Mat& cloud,
                          const Mat& rotation, const int& face,
                          ObjectModel& fstCube, ObjectModel& updatedCube) {
  random_device rd;
  default_random_engine engine(rd());
  uniform_int_distribution<unsigned int> uniform_dist(0, 255);

  // clearing face informations
  m_fstCube.resetFace(face);
  m_updatedModel.resetFace(face);

  Mat grayImg;
  checkImage(rgb, grayImg);

  // extract descriptors of the keypoint
  vector<Point3f>& fstPoints3d = fstCube.m_cloudPoints[face];
  vector<Point3f>& updatedPoint3d = updatedCube.m_cloudPoints[face];

  vector<Status>& pointStatus = fstCube.m_pointStatus[face];
  Mat& faceDescriptor = fstCube.m_faceDescriptors[face];
  vector<KeyPoint>& keypoints = fstCube.m_faceKeypoints[face];
  vector<Point3f>& pointRelPos = fstCube.m_relativePointsPos[face];

  m_featuresDetector.detect(grayImg, keypoints);
  m_foregroundMask = mask;
  m_featuresDetector.compute(grayImg, keypoints, faceDescriptor);

  updatedCube.m_faceKeypoints[face] = keypoints;

  pointRelPos.resize(keypoints.size(), Point3f(0, 0, 0));
  fstPoints3d.resize(keypoints.size(), Point3f(0, 0, 0));
  updatedPoint3d.resize(keypoints.size(), Point3f(0, 0, 0));

  for (int kp = 0; kp < keypoints.size(); ++kp) {
    keypoints[kp].class_id = kp;
    const Vec3f& point = cloud.at<Vec3f>(keypoints[kp].pt);

    if (mask.at<uchar>(keypoints[kp].pt) != 0 && point[2] != 0) {
      pointStatus.push_back(Status::MATCH);

      init_model_point_count_++;
      updatedPoint3d[kp] = Point3f(point[0], point[1], point[2]);
      Point3f projPoint(0, 0, 0);

      Point3f tmp(point[0] - m_updatedCentroid.x,
                  point[1] - m_updatedCentroid.y,
                  point[2] - m_updatedCentroid.z);
      rotatePoint(tmp, rotation.inv(), projPoint);
      pointRelPos[kp] = projPoint;

      fstPoints3d[kp] = projPoint + fstCube.m_center;

      m_isPointClustered[face].push_back(true);
    } else {
      pointStatus.push_back(Status::BACKGROUND);
      m_isPointClustered[face].push_back(false);
    }

    m_pointsColor[face].push_back(Scalar(
        uniform_dist(engine), uniform_dist(engine), uniform_dist(engine)));
  }
}

int Tracker3D::learnFaceDebug(const Mat1b& mask, const Mat& rgb,
                              const Mat& cloud, const Mat& rotation,
                              const int& face, ObjectModel& fstCube,
                              ObjectModel& updatedCube, Mat& out) {
  random_device rd;
  default_random_engine engine(rd());
  uniform_int_distribution<unsigned int> uniform_dist(0, 255);

  // clearing face informations
  m_fstCube.resetFace(face);
  m_updatedModel.resetFace(face);

  Mat grayImg;
  checkImage(rgb, grayImg);

  // extract descriptors of the keypoint
  vector<Point3f>& fstPoints3d = fstCube.m_cloudPoints[face];
  vector<Point3f>& updatedPoint3d = updatedCube.m_cloudPoints[face];

  vector<Status>& pointStatus = fstCube.m_pointStatus[face];
  Mat& faceDescriptor = fstCube.m_faceDescriptors[face];
  vector<KeyPoint>& keypoints = fstCube.m_faceKeypoints[face];
  vector<Point3f>& pointRelPos = fstCube.m_relativePointsPos[face];

  m_featuresDetector.detect(grayImg, keypoints);
  m_foregroundMask = mask;
  m_featuresDetector.compute(grayImg, keypoints, faceDescriptor);

  updatedCube.m_faceKeypoints[face] = keypoints;

  pointRelPos.resize(keypoints.size(), Point3f(0, 0, 0));
  fstPoints3d.resize(keypoints.size(), Point3f(0, 0, 0));
  updatedPoint3d.resize(keypoints.size(), Point3f(0, 0, 0));

  stringstream extracted, normalized, projected, initial;

  extracted << "extracted = np.mat([";
  projected << "projected = np.mat([";
  initial << "initial = np.mat([";
  normalized << "normalized = np.mat([";

  int kpCount = 0;

  for (int kp = 0; kp < keypoints.size(); ++kp) {
    keypoints[kp].class_id = kp;
    const Vec3f& point = cloud.at<Vec3f>(keypoints[kp].pt);

    if (mask.at<uchar>(keypoints[kp].pt) != 0 && point[2] != 0) {
      pointStatus.push_back(Status::MATCH);

      updatedPoint3d[kp] = Point3f(point[0], point[1], point[2]);
      Point3f projPoint(0, 0, 0);

      Point3f tmp(point[0] - m_updatedCentroid.x,
                  point[1] - m_updatedCentroid.y,
                  point[2] - m_updatedCentroid.z);
      rotatePoint(tmp, rotation.inv(), projPoint);
      pointRelPos[kp] = projPoint;

      fstPoints3d[kp] = projPoint + fstCube.m_center;

      m_isPointClustered[face].push_back(true);

      circle(out, keypoints[kp].pt, 3, Scalar(255, 0, 0), -1);

      extracted << toString(point) << ",";
      normalized << toString(tmp) << ",";
      projected << toString(projPoint) << ",";
      initial << toString(fstPoints3d[kp]) << ",";
      kpCount++;

    } else {
      pointStatus.push_back(Status::BACKGROUND);
      m_isPointClustered[face].push_back(false);

      // circle(out, keypoints[kp].pt, 3, Scalar(0, 0, 255), -1);
    }

    m_pointsColor[face].push_back(Scalar(
        uniform_dist(engine), uniform_dist(engine), uniform_dist(engine)));
  }

  extracted << "])\n";
  projected << "])\n";
  initial << "])\n";
  normalized << "])\n";

  return kpCount;
}

// void Tracker3D::debugTrackingStep(
//    const cv::Mat& fstFrame, const cv::Mat& scdFrame,
//    const std::vector<int>& indices, std::vector<vector<int>>& clusters,
//    const bool isLost, const bool isLearning,
//    const vector<Status*>& pointsStatus, vector<KeyPoint*>& fstKeypoints,
//    const vector<Point3f*>& fstPoints, const vector<Point3f*>& updPoints,
//    const vector<KeyPoint*>& updKeypoints, const vector<Scalar*>& colors,
//    std::vector<bool> visibleFaces, std::vector<float>& visibilityRatio,
//    double angularDistance, Mat& out) {
//  random_device rd;
//  default_random_engine engine(rd());
//  uniform_int_distribution<unsigned int> uniform_dist(0, 255);

//  int cols = fstFrame.cols;
//  int rows = fstFrame.rows;

//  buildCompositeImg(fstFrame, scdFrame, out);

//  int matchedPoints = 0;
//  int trackedPoints = 0;
//  int bothPoints = 0;

//  if (!isLost) {
//    //    drawObjectLocation(m_fstCube, m_updatedCube, visibleFaces, m_focal,
//    //                       m_imageCenter, out);
//  }
//  // cout << "11-1 ";
//  //  if (m_configFile.m_showMatching) {
//  //    // cout << fstPoints.size() << " " << updPoints.size() << " " <<
//  //    // pointsStatus.size()
//  //    //	<< " " << colors.size() << "\n";
//  //    drawPointsMatching(fstPoints, updPoints, pointsStatus, colors,
//  //                       matchedPoints, trackedPoints, bothPoints,
//  //                       m_configFile.m_drawMatchingLines, m_focal,
//  //                       m_imageCenter,
//  //                       out);
//  //  } else {
//  //    countKeypointsMatching(pointsStatus, matchedPoints, trackedPoints,
//  //                           bothPoints);
//  //  }
//  // cout << "11-2 ";
//  //  if (m_configFile.m_showVoting) {
//  //    // stringstream sstmp;
//  //    // sstmp << m_configFile.m_dstPath + "/debug/" << m_numFrames <<
//  ".txt";
//  //    // ofstream tmpFile(sstmp.str());
//  //    drawCentroidVotes(
//  //        updPoints, m_centroidVotes, m_clusteredCentroidVotes,
//  //        m_clusteredBorderVotes, pointsStatus,
//  //        m_configFile.m_drawVotingLines,
//  //        m_configFile.m_drawVotingFalse, m_focal, m_imageCenter, out);
//  //    // tmpFile.close();
//  //  }
//  // cout << "11-3 ";
//  // drawKeipointsStats(m_initKPCount, matchedPoints, trackedPoints,
//  bothPoints,
//  // out);
//  // cout << "11-4 ";
//  for (size_t i = 0; i < 6; i++) {
//    if (visibleFaces[i] && visibilityRatio[i] > 0.5f) {
//      vector<Point3f> facePoints = m_updatedCube.getFacePoints(i);

//      Point2f a, b, c, d;
//      bool isValA, isValB, isValC, isValD;

//      isValA = projectPoint(m_focal, m_imageCenter, facePoints[0], a);
//      isValB = projectPoint(m_focal, m_imageCenter, facePoints[1], b);
//      isValC = projectPoint(m_focal, m_imageCenter, facePoints[2], c);
//      isValD = projectPoint(m_focal, m_imageCenter, facePoints[3], d);

//      if (isValA && isValB && isValC && isValD) {
//        drawTriangle(a, b, c, m_faceColors[i], 0.4, out);
//        drawTriangle(a, c, d, m_faceColors[i], 0.4, out);
//      }
//    }
//  }

//  int width = 106;
//  int height = 80;
//  int left = out.cols - width;
//  int top = 0;
//  for (int i = 0; i < 6; i++) {
//    Mat tmp;
//    resize(m_learnedFaces[i], tmp, Size(width, height));
//    rectangle(tmp, Rect(0, 0, 54, 18), Scalar(0, 0, 0), -1);
//    rectangle(tmp, Rect(0, 0, out.cols, out.rows), m_faceColors[i], 3);
//    putText(tmp, faceToString(i), Point2f(5, 15), FONT_HERSHEY_PLAIN, 1,
//            m_faceColors[i], 1);

//    tmp.copyTo(out(Rect(left, top, width, height)));
//    top += height;
//  }

//  int totM = matchedPoints + bothPoints;
//  int totT = trackedPoints + bothPoints;

//  stringstream ss;
//  ss.precision(2);

//  ss << std::fixed << "Frame: " << m_numFrames << " ANG: " << angularDistance
//     << " VISIBLE: ";

//  ss << "[";
//  for (size_t i = 0; i < 6; i++) {
//    ss << visibilityRatio[i];
//    if (i < 5) ss << ",";
//  }
//  ss << "]";

//  if (isLearning) ss << " Learning Frame";
//  if (isLost) ss << " Lost";

//  drawInformationHeader(Point2f(10, 10), ss.str(), 0.5, 640, 30, out);

//  m_debugWriter << out;
//}

// void Tracker3D::debugTrackingStepICRA(
//    const cv::Mat& fstFrame, const cv::Mat& scdFrame,
//    const std::vector<int>& indices, std::vector<vector<int>>& clusters,
//    const bool isLost, const bool isLearning,
//    const vector<Status*>& pointsStatus, vector<KeyPoint*>& fstKeypoints,
//    const vector<Point3f*>& fstPoints, const vector<Point3f*>& updPoints,
//    const vector<KeyPoint*>& updKeypoints, const vector<Scalar*>& colors,
//    std::vector<bool> visibleFaces, std::vector<float>& visibilityRatio,
//    double angularDistance, Mat& out) {
//  random_device rd;
//  default_random_engine engine(rd());
//  uniform_int_distribution<unsigned int> uniform_dist(0, 255);

//  int cols = fstFrame.cols;
//  int rows = fstFrame.rows;

//  Size size(cols + 214, rows + 160);
//  out.create(size, fstFrame.type());

//  fstFrame.copyTo(out(Rect(0, 0, cols, rows)));

//  if (out.channels() == 1) cvtColor(out, out, CV_GRAY2BGR);

//  int matchedPoints = 0;
//  int trackedPoints = 0;
//  int bothPoints = 0;

//  if (!isLost) {
//    //    drawObjectLocation(m_updatedCube.m_pointsBack,
//    //    m_updatedCube.m_pointsFront,
//    //                       m_updatedCube.m_center, visibleFaces, m_focal,
//    //                       m_imageCenter, out);
//  }

//  //  // cout << "11-2 ";
//  //  if (m_configFile.m_showVoting) {
//  //    // stringstream sstmp;
//  //    // sstmp << m_configFile.m_dstPath + "/debug/" << m_numFrames <<
//  ".txt";
//  //    // ofstream tmpFile(sstmp.str());
//  //    drawCentroidVotes(
//  //        updPoints, m_centroidVotes, m_clusteredCentroidVotes,
//  //        m_clusteredBorderVotes, pointsStatus,
//  //        m_configFile.m_drawVotingLines,
//  //        m_configFile.m_drawVotingFalse, m_focal, m_imageCenter, out);
//  //    // tmpFile.close();
//  //  }
//  // cout << "11-3 ";
//  // drawKeipointsStats(m_initKPCount, matchedPoints, trackedPoints,
//  bothPoints,
//  // out);
//  // cout << "11-4 ";*/
//  for (size_t i = 0; i < 6; i++) {
//    if (visibleFaces.at(i) && visibilityRatio.at(i) > 0.5f) {
//      vector<Point3f> facePoints = m_updatedCube.getFacePoints(i);

//      Point2f a, b, c, d;
//      bool isValA, isValB, isValC, isValD;

//      isValA = projectPoint(m_focal, m_imageCenter, facePoints.at(0), a);
//      isValB = projectPoint(m_focal, m_imageCenter, facePoints.at(1), b);
//      isValC = projectPoint(m_focal, m_imageCenter, facePoints.at(2), c);
//      isValD = projectPoint(m_focal, m_imageCenter, facePoints.at(3), d);

//      if (isValA && isValB && isValC && isValD) {
//        drawTriangle(a, b, c, m_faceColors.at(i), 0.4, out);
//        drawTriangle(a, c, d, m_faceColors.at(i), 0.4, out);
//      }
//    }
//  }

//  //  int width = 214;
//  //  int height = 160;
//  //  int left = 0;
//  //  int top = 0;
//  //  for (int i = 0; i < 6; i++) {
//  //    Mat tmp;
//  //    //resize(m_learnedFaces.at(i), tmp, Size(width, height));
//  //    rectangle(tmp, Rect(0, 0, 214, 18), Scalar(0, 0, 0), -1);
//  //    rectangle(tmp, Rect(0, 0, tmp.cols, tmp.rows), m_faceColors[i], 3);

//  //    stringstream ss;
//  //    ss << faceToString(i) << " " << fixed << setprecision(2)
//  //       << visibilityRatio[i];

//  //    putText(tmp, ss.str(), Point2f(5, 15), FONT_HERSHEY_PLAIN, 1,
//  //            m_faceColors[i], 1);

//  //    if (i >= 3) {
//  //      left = (i % 3) * width;
//  //      top = out.rows - height;
//  //    } else {
//  //      left = out.cols - width;
//  //      top = (i % 3) * height;
//  //    }

//  //    tmp.copyTo(out(Rect(left, top, width, height)));
//  //  }

//  stringstream ss, ss1, ss2;
//  ss.precision(2);
//  ss1.precision(2);
//  ss2.precision(2);

//  ss << std::fixed << "Frame: " << m_numFrames;
//  ss1 << "ANGLE: " << angularDistance;

//  ss2 << "[";
//  for (size_t i = 0; i < 6; i++) {
//    ss2 << fixed << std::setprecision(2) << visibilityRatio[i];
//    if (i < 5) ss2 << ",";
//  }
//  ss2 << "]";
//  ss2.precision(2);

//  // if (isLearning)
//  //	ss << " Learning Frame";
//  // if (isLost)
//  //	ss << " Lost";
//  Point2f resolution(640, 480);
//  drawInformationHeaderICRA(resolution, ss.str(), ss1.str(), ss2.str(), 0.5,
//                            240, 160, out);

//  m_debugWriter << out;
//}

void Tracker3D::debugLearnedModel(const Mat& rgb, int face,
                                  const Mat& rotation) {
  Mat result;
  rgb.copyTo(result);
  //  m_debugFile << "\n\n\n Debugging learned mode of face: " <<
  // faceToString(face)
  //              << "\n";

  //  m_debugFile << "Centroid\n";
  //  m_debugFile << toString(m_updatedCentroid);

  //  m_debugFile << "\n Bounding Box:\n";
  //  m_debugFile << "[" << toString(m_updatedCube.m_pointsFront[0]) << ","
  //              << toString(m_updatedCube.m_pointsFront[1]) << ","
  //              << toString(m_updatedCube.m_pointsFront[2]) << ","
  //              << toString(m_updatedCube.m_pointsFront[3]) << ","
  //              << toString(m_updatedCube.m_pointsBack[0]) << ","
  //              << toString(m_updatedCube.m_pointsBack[1]) << ","
  //              << toString(m_updatedCube.m_pointsBack[2]) << ","
  //              << toString(m_updatedCube.m_pointsBack[3]) << "]\n\n";

  //  m_debugFile << "Rotation matrix:\n";
  //  m_debugFile << toPythonString(rotation);

  //  m_debugFile << "\nProjected points:\n";

  //  vector<Point3f>& points2 = m_updatedCube.m_cloudPoints[face];
  //  vector<Point3f>& points = m_fstCube.m_cloudPoints[face];
  //  m_debugFile << "[";
  //  for (size_t j = 0; j < points2.size(); j++) {
  //    if (points[j].z != 0) m_debugFile << toString(points[j]);
  //    if (points[j].z != 0 && j < points2.size() - 1) m_debugFile << ",";
  //  }
  //  m_debugFile << "]";
  //  m_debugFile << "\nExtracted points:\n";
  //  m_debugFile << "[";
  //  for (size_t j = 0; j < points2.size(); j++) {
  //    if (points[j].z != 0) m_debugFile << toString(points2[j]);
  //    if (points[j].z != 0 && j < points.size() - 1) m_debugFile << ",";
  //  }
  //  m_debugFile << "]\n";
}

string Tracker3D::getResultName(string path) {
  fs::path dir(path);
  fs::directory_iterator end_iter;

  int count = 0;
  if (fs::exists(dir) && fs::is_directory(dir)) {
    for (fs::directory_iterator dir_iter(dir); dir_iter != end_iter;
         ++dir_iter) {
      if (fs::is_regular_file(dir_iter->status())) {
        string path = dir_iter->path().filename().string();
        if (path.find("result") != string::npos) count++;
      }
    }
  }

  stringstream dirname;
  dirname << "result_" << std::setw(4) << std::setfill('0') << count;
  return dirname.str();
}

string Tracker3D::toPythonString2(const vector<Point3f>& firstFrameCloud,
                                  const vector<Point3f>& updatedFrameCloud,
                                  const vector<Status>& keypointStatus) {
  stringstream ss;
  ss << "[";
  for (size_t j = 0; j < firstFrameCloud.size(); j++) {
    if (isKeypointValid(keypointStatus[j]) && firstFrameCloud[j].z != 0 &&
        updatedFrameCloud[j].z != 0)
      ss << toString(firstFrameCloud[j]) << ",";
  }
  ss << "]\n\n";

  ss << "[";
  for (size_t j = 0; j < firstFrameCloud.size(); j++) {
    if (isKeypointValid(keypointStatus[j]) && firstFrameCloud[j].z != 0 &&
        updatedFrameCloud[j].z != 0)
      ss << toString(updatedFrameCloud[j]) << ",";
  }
  ss << "]\n\n";

  return ss.str();
}

int Tracker3D::getFilesCount(string path, string substring) {
  fs::path dir(path);
  fs::directory_iterator end_iter;

  int count = 0;
  if (fs::exists(dir) && fs::is_directory(dir)) {
    for (fs::directory_iterator dir_iter(dir); dir_iter != end_iter;
         ++dir_iter) {
      if (fs::is_regular_file(dir_iter->status())) {
        string path = dir_iter->path().filename().string();
        if (path.find(substring) != string::npos) count++;
      }
    }
  }

  return count;
}

void Tracker3D::drawLearnedFace(const Mat& rgb, int face, Mat& out) {
  rgb.copyTo(out);

  vector<Point3f> facePoints = m_updatedModel.getFacePoints(face);

  Point2f a, b, c, d;
  bool isValA, isValB, isValC, isValD;

  isValA = projectPoint(m_focal, m_imageCenter, facePoints[0], a);
  isValB = projectPoint(m_focal, m_imageCenter, facePoints[1], b);
  isValC = projectPoint(m_focal, m_imageCenter, facePoints[2], c);
  isValD = projectPoint(m_focal, m_imageCenter, facePoints[3], d);

  if (isValA && isValB && isValC && isValD) {
    line(out, a, b, m_faceColors[face], 3);
    line(out, b, c, m_faceColors[face], 3);
    line(out, c, d, m_faceColors[face], 3);
    line(out, d, a, m_faceColors[face], 3);
    drawTriangle(a, b, c, m_faceColors[face], 0.2, out);
    drawTriangle(a, c, d, m_faceColors[face], 0.2, out);
  }
}

double Tracker3D::getQuaternionMedianDist(const vector<Quaterniond>& history,
                                          int window, const Quaterniond& q) {
  int numElems = (window < history.size()) ? window : history.size();

  if (numElems == 0) return 0;

  vector<double> quaternionDist(numElems, 0);

  for (int i = 0; i < numElems; ++i) {
    const Quaterniond& other = history[history.size() - numElems + i];
    quaternionDist[i] = q.angularDistance(other) * 180 / 3.14159265;
  }

  sort(quaternionDist.begin(), quaternionDist.end());

  double median;

  if (numElems % 2 == 0) {
    median =
        (quaternionDist[numElems / 2 - 1] + quaternionDist[numElems / 2]) / 2;
  } else {
    median = quaternionDist[numElems / 2];
  }

  return median;
}

bool Tracker3D::findObject(ObjectModel& fst, ObjectModel& upd,
                           const Mat& extractedDescriptors,
                           const vector<KeyPoint>& extractedKeypoints,
                           int& faceFound, int& matchesNum) {
  int maxMatches = 0;
  int maxMatchesFace = -1;
  bool foundFace = false;

  for (int i = 0; i < 6; ++i) {
    if (fst.m_isLearned[i]) {
      const Mat& fstDescriptors = fst.m_faceDescriptors[i];
      vector<KeyPoint*> fstKeypoints;
      vector<KeyPoint*> updKeypoints;
      vector<Status*> pointsStatus;

      for (int j = 0; j < fst.m_faceKeypoints[i].size(); j++) {
        fstKeypoints.push_back(&fst.m_faceKeypoints[i][j]);
        updKeypoints.push_back(&upd.m_faceKeypoints[i][j]);
        pointsStatus.push_back(&fst.m_pointStatus[i][j]);
      }

      int numMatches = matchFeaturesCustom(
          fstDescriptors, fstKeypoints, extractedDescriptors,
          extractedKeypoints, updKeypoints, pointsStatus);

      if (numMatches > maxMatches) {
        maxMatches = numMatches;
        maxMatchesFace = i;
        foundFace = true;
      }

      //      m_debugFile << faceToString(i) << " descriptors " <<
      // fstDescriptors.size()
      //                  << " " << numMatches << "\n";
    }
  }

  faceFound = maxMatchesFace;
  matchesNum = maxMatches;

  return foundFace;
}

void Tracker3D::drawObjectLocation(Mat& out) {
  drawBoundingCube(m_updatedModel.m_center, m_updatedModel.m_pointsBack,
                   m_updatedModel.m_pointsFront, m_focal, m_imageCenter, out);
}

void Tracker3D::drawRansacEstimation(Mat& out) {
  Point3f obj_centroid = m_updatedCentroid;

  auto type = ransac_rotation_.type();

  vector<Point3f> rel_updates, front_points, back_points;

  rotateBBox(m_fstCube.m_relativeDistFront, ransac_rotation_, rel_updates);

  for (int i = 0; i < m_fstCube.m_pointsFront.size(); ++i) {
    front_points.push_back(obj_centroid + rel_updates[i]);
  }

  rel_updates.clear();
  rotateBBox(m_fstCube.m_relativeDistBack, ransac_rotation_, rel_updates);

  for (int i = 0; i < m_fstCube.m_pointsBack.size(); ++i) {
    back_points.push_back(obj_centroid + rel_updates[i]);
  }

  drawBoundingCube(m_updatedModel.m_center, back_points, front_points, m_focal,
                   m_imageCenter, out);
}

}  // end namespace
