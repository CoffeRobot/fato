#include "../include/tracker_2d.h"

#include <random>
#include <opencv2/highgui/highgui.hpp>
#include <limits>
#include <future>
#include <math.h>
#include <boost/math/constants/constants.hpp>
#include <utilities.h>

using namespace std;
using namespace cv;

namespace pinot_tracker {

Tracker2D::~Tracker2D() {}

void Tracker2D::init(Mat& src, const Point2d& fst, const Point2d& scd) {
  auto mask = getMask(src.rows, src.cols, fst, scd);
  init(src, mask);
}

void Tracker2D::init(Mat& src, Mat& mask) {
  Mat grayImg;
  // check if image is grayscale
  checkImage(src, grayImg);
  // itialize structure to store keypoints
  vector<KeyPoint> keypoints;
  // find keypoints in the starting image
  m_featuresDetector.detect(grayImg, keypoints);
  m_foregroundMask = mask;
  m_currFrame = grayImg;
  m_fstFrame = grayImg;
  // store keypoints of the orginal image
  m_initKeypoints = keypoints;

  cout << "Image " << grayImg.cols << " " << grayImg.rows << " " << grayImg.channels()
       << endl;
  cout << "Number of keypoints extracted: " << keypoints.size() << endl;

  // extract descriptors of the keypoint
  // TODO: modify the name of the descriptors
  m_featuresDetector.compute(m_currFrame, keypoints, m_firstFrameDescriptors);
  cout << "Feature descriptor size: " << m_firstFrameDescriptors.size()
       << std::endl;

  // if (m_configFile.m_useSysthetic)
  //	extractSyntheticKeypoints(m_fstFrame, keypoints,
  //m_firstFrameDescriptors);

  cout << "Feature descriptor size: " << m_firstFrameDescriptors.size()
       << std::endl;
  // calcualte centroid of object
  // calculateCentroid(keypoints, 1, 0, m_firstCentroid);

  m_initKPCount = 0;

  random_device rd;
  default_random_engine engine(rd());
  uniform_int_distribution<unsigned int> uniform_dist(0, 255);

  // count valid keypoints that belong to the object
  for (int kp = 0; kp < keypoints.size(); ++kp) {
    keypoints[kp].class_id = kp;
    if (mask.at<uchar>(keypoints[kp].pt) != 0) {
      m_keypointStatus.push_back(Status::INIT);
      m_initKPCount++;
    } else {
      m_keypointStatus.push_back(Status::BACKGROUND);
    }
    m_pointsColor.push_back(Scalar(uniform_dist(engine), uniform_dist(engine),
                                   uniform_dist(engine)));
  }

  keypoints.swap(m_firstFrameKeypoints);

  initCentroid(m_firstFrameKeypoints);
  initRelativePosition();
  // initialize bounding box
  initBBox();
  // set current centroid
  // set first and current frame
  m_currFrame = grayImg;
  m_fstFrame = grayImg;

  m_updatedKeypoints.insert(m_updatedKeypoints.end(),
                            m_firstFrameKeypoints.begin(),
                            m_firstFrameKeypoints.end());

  m_matchedPoints.resize(m_updatedKeypoints.size(), Point2f(0, 0));
  m_trackedPoints.resize(m_updatedKeypoints.size(), Point2f(0, 0));

  m_centroidVotes.resize(m_firstFrameKeypoints.size(), Point2f(0, 0));
  m_clusteredCentroidVotes.resize(m_firstFrameKeypoints.size(), true);
  m_clusteredBorderVotes.resize(m_firstFrameKeypoints.size(), false);

  m_mAngle = 0;
  m_mScale = 1;

  m_extractT = m_matchT = m_trackT = m_scaleT = m_rotT = m_votingT = m_updateT =
      m_clusterT = m_centroidT = m_drawT = 0;
  m_updateVoteT = 0;
  m_numFrames = 0;
  m_clusterVoteSize = 0;

  // adding to the set of learned frames the initial frame with angle 0 and
  // scale 1
  m_learnedFrames.insert(pair<int, int>(0, 0));
}

void Tracker2D::extractSyntheticKeypoints(const cv::Mat& src,
                                          std::vector<cv::KeyPoint>& points,
                                          cv::Mat& descriptors) {
  // apply different image transformations and extract features to track
  int kernelSize = 5;
  float elemVal = 1.0 / static_cast<float>(kernelSize);

  Mat1f hk(kernelSize, kernelSize, 0.0);
  Mat1f tr(kernelSize, kernelSize, 0.0);
  Mat1f br(kernelSize, kernelSize, 0.0);
  Mat1f vk(kernelSize, kernelSize, 0.0);
  ;

  for (size_t i = 0; i < kernelSize; i++) {
    for (size_t j = 0; j < kernelSize; j++) {
      if (i == j) tr.at<float>(i, j) = elemVal;
      if (i == kernelSize / 2 + 1) hk.at<float>(i, j) = elemVal;
      if (j == kernelSize / 2 + 1) vk.at<float>(i, j) = elemVal;
      if (i == kernelSize - j - 1) br.at<float>(i, j) = elemVal;
    }
  }

  Mat horizontal, vertical, diagonal1, diagonal2;
  filter2D(src, horizontal, -1, hk);
  filter2D(src, vertical, -1, vk);
  filter2D(src, diagonal1, -1, tr);
  filter2D(src, diagonal2, -1, br);

  // imshow("H Blur debug", hOut);
  // imshow("V Blur debug", vOut);
  // imshow("D Blur debug", dOut1);
  // imshow("D2 Blur debug", dOut2);
}

void Tracker2D::computeNext(Mat& next, Mat& out) {
  // m_debugFile << "Frame: " << m_numFrames << "\n";

  std::chrono::time_point<std::chrono::system_clock> start, end;

  Mat grayImg;
  checkImage(next, grayImg);

  int numMatches, numTracked, numBoth;
  numTracked = 0;
  numBoth = 0;
  vector<KeyPoint> nextKeypoints;
  Mat nextDescriptors;

  auto& profiler = Profiler::getInstance();

  /*********************************************************************************************/
  /*                             TRACKING */
  /*********************************************************************************************/
  start = chrono::system_clock::now();
  trackFeatures(grayImg, numTracked, numBoth);
  end = chrono::system_clock::now();
  m_trackT += chrono::duration_cast<chrono::milliseconds>(end - start).count();
  //	m_debugFile << "tracked " << numTracked << endl;
  cout << "2 ";
  /*********************************************************************************************/
  /*                             ANGLE */
  /*********************************************************************************************/
  float mAngle = 0;
  start = chrono::system_clock::now();
  mAngle = getMedianRotation();
  end = chrono::system_clock::now();
  m_rotT += chrono::duration_cast<chrono::milliseconds>(end - start).count();
  cout << "3 ";
  /*********************************************************************************************/
  /*                             SCALE */
  /*********************************************************************************************/
  float mScale = 1;
  start = chrono::system_clock::now();
  mScale = getMedianScale();
  end = chrono::system_clock::now();
  m_scaleT += chrono::duration_cast<chrono::milliseconds>(end - start).count();
  cout << "4 ";
  /*********************************************************************************************/
  /*                             KEYPOINT STATUS */
  /*********************************************************************************************/
  start = chrono::system_clock::now();
  updateKeypoints();
  end = chrono::system_clock::now();
  m_updateT += chrono::duration_cast<chrono::milliseconds>(end - start).count();
  cout << "5 ";
  /*********************************************************************************************/
  /*                             VOTING */
  /*********************************************************************************************/
  start = chrono::system_clock::now();
  getVotes(mScale, mAngle);
  end = chrono::system_clock::now();
  m_votingT += chrono::duration_cast<chrono::milliseconds>(end - start).count();
  /*********************************************************************************************/
  /*                             CLUSTERING */
  /*********************************************************************************************/
  vector<int> indices;
  vector<vector<int>> clusters;
  start = chrono::system_clock::now();
  if (params_.filter_border)
    clusterVotesBorder(indices, clusters);
  else
    clusterVotesDebug(indices, clusters);
  end = chrono::system_clock::now();
  m_clusterT +=
      chrono::duration_cast<chrono::milliseconds>(end - start).count();
  cout << "7 ";
  /*********************************************************************************************/
  /*                             UPDATE VOTES */
  /*********************************************************************************************/
  start = chrono::system_clock::now();
  if (params_.update_votes) updateVotes();
  end = chrono::system_clock::now();
  m_updateVoteT +=
      chrono::duration_cast<chrono::milliseconds>(end - start).count();
  cout << "8 ";
  /*********************************************************************************************/
  /*                             UPDATE OBJECT CENTER */
  /*********************************************************************************************/
  start = chrono::system_clock::now();
  updateCentroid();
  end = chrono::system_clock::now();
  m_centroidT +=
      chrono::duration_cast<chrono::milliseconds>(end - start).count();
  cout << "9 ";

  /*********************************************************************************************/
  /*                             MATCHING */
  /*********************************************************************************************/
  start = chrono::system_clock::now();
  m_featuresDetector.detect(grayImg, nextKeypoints);
  m_featuresDetector.compute(grayImg, nextKeypoints, nextDescriptors);
  end = chrono::system_clock::now();
  m_extractT +=
      chrono::duration_cast<chrono::milliseconds>(end - start).count();
  start = chrono::system_clock::now();
  numMatches = matchFeaturesCustom(grayImg, nextKeypoints, nextDescriptors);
  end = chrono::system_clock::now();
  m_matchT += chrono::duration_cast<chrono::milliseconds>(end - start).count();

  /*
      if (m_configFile.m_useCustomMatcher)
      {

      }
      else
      {
              start = chrono::system_clock::now();
              matchFeatures(grayImg);
              end = chrono::system_clock::now();
              m_matchT += chrono::duration_cast<chrono::milliseconds>(end -
     start).count();
      }
      m_debugFile << "Matches " << numMatches << "\n";
  */
  cout << "10 ";
  /*********************************************************************************************/
  /*                             DRAW */
  /*********************************************************************************************/
  start = chrono::system_clock::now();
  debugTrackingStep(grayImg, m_fstFrame, indices, clusters, out);
  end = chrono::system_clock::now();
  m_drawT += chrono::duration_cast<chrono::milliseconds>(end - start).count();
  cout << "10 ";

  cout << "\n";

  m_currFrame = grayImg;
  m_mAngle = mAngle;
  m_mScale = mScale;
  m_numFrames++;
}

void Tracker2D::close() {
  float frames = static_cast<float>(m_numFrames);

  /*
      m_timeFile << "Average timing of the steps: \n";
      m_timeFile << "Extraction time: " << (m_extractT / frames) << "\n";
      m_timeFile << "Match time: " << (m_matchT / frames) << "\n";
      m_timeFile << "Track time: " << (m_trackT / frames) << "\n";
      m_timeFile << "Rotation time: " << (m_rotT / frames) << "\n";
      m_timeFile << "Scale time: " << (m_scaleT / frames) << "\n";
      m_timeFile << "Update time: " << (m_updateT / frames) << "\n";
      m_timeFile << "Vote time: " << (m_votingT / frames) << "\n";
      m_timeFile << "Cluster time: " << (m_clusterT / frames) << "\n";
      m_timeFile << "Centroid time: " << (m_centroidT / frames) << "\n";
      m_timeFile << "Draw time: " << (m_drawT / frames) << "\n";


      float totTime = m_extractT + m_matchT + m_trackT + m_rotT + m_scaleT +
     m_updateT + m_votingT
              + m_clusterT + m_centroidT;

      m_timeFile << "Average time per frame: " << totTime / frames << "\n";
      m_timeFile << "FPS no draw: " << 1000 / (totTime / frames) << "\n";
      m_timeFile << "FPS with draw: " << 1000 / ((totTime + m_drawT) / frames)
     << "\n";

      m_timeFile.close();
  */
}

void Tracker2D::matchFeatures(const Mat& grayImg) {
  vector<KeyPoint> nextKeypoints;
  vector<DMatch> matches;

  Mat currDescriptors, nextDescriptors;

  m_featuresDetector.detect(grayImg, nextKeypoints);
  m_featuresDetector.compute(grayImg, nextKeypoints, nextDescriptors);

  m_featureMatcher->match(nextDescriptors, m_firstFrameDescriptors, matches);

  for (size_t i = 0; i < matches.size(); i++) {
    int queryId = matches[i].queryIdx;
    int trainId = matches[i].trainIdx;

    if (queryId < 0 && queryId >= nextKeypoints.size()) continue;

    if (trainId < 0 && trainId >= m_firstFrameKeypoints.size()) continue;

    float confidence = 1 - (matches[i].distance / 512.0);
    Status& s = m_keypointStatus[trainId];

    if (s == Status::BACKGROUND)
      continue;
    else if (confidence >= 0.75f) {
      m_keypointStatus[trainId] = Status::MATCH;
      // TODO: remove comment to see behaviour when merging match and track
      // m_updatedKeypoints[trainId] = nextKeypoints[queryId].pt;
      m_matchedPoints[trainId] = nextKeypoints[queryId].pt;
    } else if (confidence < 0.75f) {
      m_keypointStatus[trainId] = Status::NOMATCH;
    }
  }
}

int Tracker2D::matchFeaturesCustom(const Mat& grayImg,
                                   vector<KeyPoint>& nextKeypoints,
                                   Mat& nextDescriptors) {
  vector<vector<DMatch>> matches;

  Mat currDescriptors;

  m_customMatcher.match(nextDescriptors, m_firstFrameDescriptors, 2, matches);

  int matchesCount = 0;

  for (size_t i = 0; i < matches.size(); i++) {
    int queryId = matches[i][0].queryIdx;
    int trainId = matches[i][0].trainIdx;

    if (queryId < 0 && queryId >= nextKeypoints.size()) continue;

    if (trainId < 0 && trainId >= m_firstFrameKeypoints.size()) continue;

    float confidence = 1 - (matches[i][0].distance / 512.0);
    float ratio = matches[i][0].distance / matches[i][1].distance;

    Status& s = m_keypointStatus[trainId];

    if (s == Status::BACKGROUND)
      continue;
    else if (confidence >= 0.75f && ratio <= 0.8) {
      if (s == Status::TRACK)
        m_keypointStatus[trainId] = Status::BOTH;
      else {
        m_keypointStatus[trainId] = Status::MATCH;
        m_updatedKeypoints[trainId].pt = nextKeypoints[queryId].pt;
      }

      // TODO: remove comment to see behaviour when merging match and track
      // m_updatedKeypoints[trainId] = nextKeypoints[queryId];
      m_matchedPoints[trainId] = nextKeypoints[queryId].pt;
      matchesCount++;
    }
  }

  return matchesCount;
}

void Tracker2D::trackFeatures(const Mat& grayImg, int& trackedCount,
                              int& bothCount) {
  vector<Point2f> nextPoints;
  vector<Point2f> currPoints;
  vector<int> ids;
  vector<Point2f> prevPoints;

  vector<uchar> prevStatus;
  vector<float> prevErrors;

  // m_debugFile << "Keypoint size: " << m_updatedKeypoints.size() << "\n";

  for (size_t i = 0; i < m_updatedKeypoints.size(); i++) {
    if (m_keypointStatus[i] == Status::TRACK ||
        m_keypointStatus[i] == Status::BOTH ||
        m_keypointStatus[i] == Status::MATCH ||
        m_keypointStatus[i] == Status::INIT) {
      currPoints.push_back(m_updatedKeypoints[i].pt);
      ids.push_back(i);
    }
  }

  if (currPoints.empty()) return;

  calcOpticalFlowPyrLK(m_currFrame, grayImg, currPoints, nextPoints, m_status,
                       m_errors);

  calcOpticalFlowPyrLK(grayImg, m_currFrame, nextPoints, prevPoints, m_status,
                       prevErrors);

  for (int i = 0; i < nextPoints.size(); ++i) {
    float error = getDistance(currPoints[i], prevPoints[i]);

    Status& s = m_keypointStatus[ids[i]];

    if (m_status[i] == 1 && error < 20) {
      int& id = ids[i];

      if (s == Status::MATCH) {
        m_keypointStatus[id] = Status::BOTH;
        bothCount++;
        trackedCount++;
      } else if (s == Status::LOST)
        m_keypointStatus[id] = Status::LOST;
      else {
        m_keypointStatus[id] = Status::TRACK;
        trackedCount++;
      }

      m_trackedPoints[id] = nextPoints[i];
      m_updatedKeypoints[id].pt = nextPoints[i];
    } else {
      m_keypointStatus[ids[i]] = Status::LOST;
    }
  }
}

void Tracker2D::checkImage(const Mat& src, Mat& gray) {
  if (src.channels() > 1)
    cvtColor(src, gray, CV_BGR2GRAY);
  else
    gray = src;
}

float Tracker2D::getMedianRotation() {
  vector<double> angles;
  angles.reserve(m_updatedKeypoints.size() * 2);
  const double pi = boost::math::constants::pi<double>();

  ofstream file;

  for (size_t i = 0; i < m_updatedKeypoints.size(); ++i) {
    for (size_t j = 0; j < m_firstFrameKeypoints.size(); j++) {
      if (isKeypointValid(i) && isKeypointValid(j) && i != j) {
        Point2f a = m_updatedKeypoints[i].pt - m_updatedKeypoints[j].pt;
        Point2f b = m_firstFrameKeypoints[i].pt - m_firstFrameKeypoints[j].pt;

        double val = atan2(a.y, a.x) - atan2(b.y, b.x);

        if (abs(val) > pi) {
          int sign = (val < 0) ? -1 : 1;
          val = val - sign * 2 * pi;
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

float Tracker2D::getMedianScale() {
  vector<float> scales;

  for (size_t i = 0; i < m_updatedKeypoints.size(); ++i) {
    for (size_t j = 0; j < m_firstFrameKeypoints.size(); j++) {
      if (isKeypointValid(i) && isKeypointValid(j)) {
        float nextDistance =
            getDistance(m_updatedKeypoints[i], m_updatedKeypoints[j]);
        float currDistance =
            getDistance(m_firstFrameKeypoints[i], m_firstFrameKeypoints[j]);

        if (currDistance != 0 && i != j) {
          scales.push_back(nextDistance / currDistance);
        }
      }
    }
  }

  sort(scales.begin(), scales.end());

  float median;
  size_t size = scales.size();

  if (size == 0) return 1;

  if (size % 2 == 0) {
    median = (scales[size / 2 - 1] + scales[size / 2]) / 2;
  } else {
    median = scales[size / 2];
  }

  return median;
}

void Tracker2D::getVotes(float scale, float angle) {
  Mat2f rotMat(2, 2);

  rotMat.at<float>(0, 0) = cosf(angle);
  rotMat.at<float>(0, 1) = sinf(angle);
  rotMat.at<float>(1, 0) = -sinf(angle);
  rotMat.at<float>(1, 1) = cosf(angle);

  for (size_t i = 0; i < m_firstFrameKeypoints.size(); i++) {
    if (isKeypointValid(i)) {
      Point2f& a = m_updatedKeypoints[i].pt;
      Point2f& rm = m_relativePointsPos[i];

      m_centroidVotes[i] = a - scale * mult(rotMat, rm);
    }
  }
}

void Tracker2D::clusterVotes() {
  DBScanClustering<Point2f*> clusterer;

  vector<Point2f*> votes;
  vector<unsigned int> indices;

  for (unsigned int i = 0; i < m_keypointStatus.size(); i++) {
    if (isKeypointTracked(i)) {
      votes.push_back(&m_centroidVotes[i]);
      indices.push_back(i);
    }
  }

  clusterer.clusterPoints(
      &votes, params_.eps, params_.min_points, [](Point2f* a, Point2f* b) {
        return sqrt(pow(a->x - b->x, 2) + pow(a->y - b->y, 2));
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

void Tracker2D::clusterVotesDebug(vector<int>& indices,
                                  vector<vector<int>>& clusters) {
  DBScanClustering<Point2f*> clusterer;

  vector<Point2f*> votes;
  indices.clear();

  for (unsigned int i = 0; i < m_keypointStatus.size(); i++) {
    if (isKeypointTracked(i)) {
      votes.push_back(&m_centroidVotes[i]);
      indices.push_back(i);
    }
  }

  clusterer.clusterPoints(
      &votes, params_.eps, params_.min_points, [](Point2f* a, Point2f* b) {
        return sqrt(pow(a->x - b->x, 2) + pow(a->y - b->y, 2));
      });

  clusters = clusterer.getClusters();
  // std::cout << "Size of clusters " << res.size() << "\n" << std::endl;
  int maxVal = 0;
  int maxId = -1;

  for (size_t i = 0; i < clusters.size(); i++) {
    // std::cout << "Size of cluster " << i << ":" << res[i].size() << "\n" <<
    // std::endl;
    if (clusters[i].size() > maxVal) {
      maxVal = clusters[i].size();
      maxId = i;
    }
  }

  // std::cout << "Picked cluster " << maxId << " size " << res[maxId].size() <<
  // "\n" << std::endl;

  vector<bool> clustered(m_clusteredCentroidVotes.size(), false);

  if (maxId > -1) {
    for (size_t i = 0; i < clusters[maxId].size(); i++) {
      int& id = indices[clusters[maxId][i]];
      clustered[id] = true;
    }
  }

  m_clusteredCentroidVotes.swap(clustered);
}

void Tracker2D::clusterVotesBorder(vector<int>& indices,
                                   vector<vector<int>>& clusters) {
  DBScanClustering<Point2f*> clusterer;

  vector<Point2f*> votes;
  indices.clear();

  for (unsigned int i = 0; i < m_keypointStatus.size(); i++) {
    if (isKeypointTracked(i)) {
      votes.push_back(&m_centroidVotes[i]);
      indices.push_back(i);
    }
  }

  clusterer.clusterPoints(
      &votes, params_.eps, params_.min_points, [](Point2f* a, Point2f* b) {
        return sqrt(pow(a->x - b->x, 2) + pow(a->y - b->y, 2));
      });

  // clusters = clusterer.getClusters();
  // vector<vector<int> > clusters;
  vector<bool> border;

  clusterer.getBorderClusters(clusters, border);

  // std::cout << "Size of clusters " << res.size() << "\n" << std::endl;
  int maxVal = 0;
  int maxId = -1;

  for (size_t i = 0; i < clusters.size(); i++) {
    // std::cout << "Size of cluster " << i << ":" << res[i].size() << "\n" <<
    // std::endl;
    if (clusters[i].size() > maxVal) {
      maxVal = clusters[i].size();
      maxId = i;
    }
  }

  // std::cout << "Picked cluster " << maxId << " size " << res[maxId].size() <<
  // "\n" << std::endl;

  vector<bool> clustered(m_clusteredCentroidVotes.size(), false);
  vector<bool> borderPoints(m_clusteredBorderVotes.size(), false);

  if (maxId > -1) {
    for (size_t i = 0; i < clusters[maxId].size(); i++) {
      int& id = indices[clusters[maxId][i]];
      clustered[id] = true;
      borderPoints[id] = border[clusters[maxId][i]];
    }
  }

  m_clusteredCentroidVotes.swap(clustered);
  m_clusteredBorderVotes.swap(borderPoints);
}

void Tracker2D::initCentroid(vector<KeyPoint>& keypoints) {
  m_firstCentroid.x = 0;
  m_firstCentroid.y = 0;

  int validKp = 0;

  for (int i = 0; i < keypoints.size(); ++i) {
    if (isKeypointValid(i)) {
      m_firstCentroid += keypoints[i].pt;
      validKp++;
    }
  }

  m_firstCentroid.x = m_firstCentroid.x / static_cast<float>(validKp);
  m_firstCentroid.y = m_firstCentroid.y / static_cast<float>(validKp);
  m_updatedCentroid = m_firstCentroid;
}

void Tracker2D::initRelativePosition() {
  m_relativePointsPos.resize(m_firstFrameKeypoints.size(), Point2f(0, 0));

  for (size_t i = 0; i < m_firstFrameKeypoints.size(); i++) {
    if (isKeypointValid(i))
      m_relativePointsPos[i] = m_firstFrameKeypoints[i].pt - m_firstCentroid;
  }
}

void Tracker2D::calculateCentroid(vector<KeyPoint>& keypoints, float scale,
                                  float angle, Point2f& p) {
  p.x = 0;
  p.y = 0;

  int validKp = 0;

  for (int i = 0; i < keypoints.size(); ++i) {
    if (isKeypointValid(i)) {
      p += keypoints[i].pt;
      validKp++;
    }
  }

  p.x = p.x / static_cast<float>(validKp);
  p.y = p.y / static_cast<float>(validKp);
}

void Tracker2D::updateCentroid() {
  m_updatedCentroid.x = 0;
  m_updatedCentroid.y = 0;

  int validKp = 0;

  for (int i = 0; i < m_centroidVotes.size(); ++i) {
    if (isKeypointValid(i) && m_clusteredCentroidVotes[i]) {
      m_updatedCentroid += m_centroidVotes[i];
      validKp++;
    }
  }

  m_updatedCentroid.x = m_updatedCentroid.x / static_cast<float>(validKp);
  m_updatedCentroid.y = m_updatedCentroid.y / static_cast<float>(validKp);

  m_validKeypoints = validKp;

  Mat2f rotMat(2, 2);

  rotMat.at<float>(0, 0) = cosf(m_mAngle);
  rotMat.at<float>(0, 1) = sinf(m_mAngle);
  rotMat.at<float>(1, 0) = -sinf(m_mAngle);
  rotMat.at<float>(1, 1) = cosf(m_mAngle);

  for (int i = 0; i < m_updatedBBPoints.size(); ++i) {
    m_updatedBBPoints[i] =
        m_updatedCentroid + m_mScale * mult(rotMat, m_fstRelativeBBPoints[i]);
  }
}

void Tracker2D::getClusteredEstimations(vector<float>& scales,
                                        vector<float>& angles) {
  const double pi = boost::math::constants::pi<double>();

  for (size_t i = 0; i < m_updatedKeypoints.size(); ++i) {
    for (size_t j = 0; j < m_firstFrameKeypoints.size(); j++) {
      if (isKeypointValidCluster(i) && isKeypointValidCluster(j) && i != j) {
        Point2f a = m_updatedKeypoints[i].pt - m_updatedKeypoints[j].pt;
        Point2f b = m_firstFrameKeypoints[i].pt - m_firstFrameKeypoints[j].pt;
        double val = atan2(a.y, a.x) - atan2(b.y, b.x);
        if (abs(val) > pi) {
          int sign = (val < 0) ? -1 : 1;
          val = val - sign * 2 * pi;
        }
        angles.push_back(val);
        // angles.push_back(atan2(a.y, a.x) - atan2(b.y, b.x));
        float nextDistance =
            getDistance(m_updatedKeypoints[i], m_updatedKeypoints[j]);
        float currDistance =
            getDistance(m_firstFrameKeypoints[i], m_firstFrameKeypoints[j]);

        if (currDistance != 0) {
          scales.push_back(nextDistance / currDistance);
        }
      }
    }
  }
}

void Tracker2D::updateVotes() {
  int iterNum = 2;

  // m_debugFile << "Median values: " << m_mScale << " " << m_mAngle << "\n";
  // m_debugFile << "Updating votes... \n";

  int numClusteredPoints = 0;

  float medianAngle = 0;
  float medianScale = 1;

  for (size_t i = 0; i < iterNum; i++) {
    vector<float> angles;
    vector<float> scales;
    // compute new centroid position
    getClusteredEstimations(scales, angles);

    // if (numClusteredPoints > angles.size())
    //	break;

    numClusteredPoints = angles.size();

    if (angles.size() > 1) {
      sort(angles.begin(), angles.end());
      sort(scales.begin(), scales.end());

      size_t size = angles.size();
      if (size % 2 == 0)
        medianAngle = (angles[size / 2 - 1] + angles[size / 2]) / 2;
      else
        medianAngle = angles[size / 2];
      size_t size2 = scales.size();
      if (size2 % 2 == 0)
        medianScale = (scales[size / 2 - 1] + scales[size / 2]) / 2;
      else
        medianScale = scales[size / 2];
    }

    // vote again for the new centroid position
    getVotes(medianScale, medianAngle);
    // cluster votes again
    clusterVotes();

    // m_debugFile << "Iter: " << i <<" scale: " << medianScale << " angle: " <<
    // m_mAngle << "\n";
  }

  for (int i = 0; i < m_keypointStatus.size(); i++) {
    if (isKeypointValid(i) && !isKeypointValidCluster(i))
      m_keypointStatus[i] = Status::LOST;
  }

  // m_debugFile << "Update scale: " << m_mScale << " -> " << medianScale <<
  // "\n"
  //		        << "Update angle: " << m_mAngle << " -> " << medianAngle <<
  //"\n";

  m_mAngle = medianAngle;
  m_mScale = medianScale;
}

void Tracker2D::updateVotesDebug(vector<int>& indices,
                                 vector<vector<int>>& clusters) {
  int iterNum = 5;

  // m_debugFile << "Median values: " << m_mScale << " " << m_mAngle << "\n";
  // m_debugFile << "Updating votes... \n";

  int numClusteredPoints = 0;

  float medianAngle = 0;
  float medianScale = 1;

  for (size_t i = 0; i < iterNum; i++) {
    vector<float> angles;
    vector<float> scales;
    // compute new centroid position
    getClusteredEstimations(scales, angles);

    if (numClusteredPoints > angles.size()) break;

    numClusteredPoints = angles.size();

    if (angles.size() > 1) {
      sort(angles.begin(), angles.end());
      sort(scales.begin(), scales.end());

      size_t size = angles.size();
      if (size % 2 == 0)
        medianAngle = (angles[size / 2 - 1] + angles[size / 2]) / 2;
      else
        medianAngle = angles[size / 2];
      size_t size2 = scales.size();
      if (size2 % 2 == 0)
        medianScale = (scales[size / 2 - 1] + scales[size / 2]) / 2;
      else
        medianScale = scales[size / 2];
    }

    // vote again for the new centroid position
    getVotes(medianScale, medianAngle);
    // cluster votes again
    indices.clear();
    clusters.clear();
    clusterVotesDebug(indices, clusters);

    // m_debugFile << "Iter: " << i << " values: " << medianScale << " " <<
    // medianAngle
    //	<< " points " << angles.size() << "\n";
  }

  for (int i = 0; i < m_keypointStatus.size(); i++) {
    if (isKeypointValid(i) && !isKeypointValidCluster(i))
      m_keypointStatus[i] = Status::LOST;
  }

  // m_debugFile << "Update scale: " << m_mScale << " -> " << medianScale <<
  // "\n"
  //	<< "Update angle: " << m_mAngle << " -> " << medianAngle << "\n";

  m_mAngle = medianAngle;
  m_mScale = medianScale;
}

void Tracker2D::initBBox() {
  int minX = numeric_limits<int>::max();
  int minY = numeric_limits<int>::max();
  int maxX = 0, maxY = 0;

  for (int i = 0; i < m_foregroundMask.cols; i++) {
    for (int y = 0; y < m_foregroundMask.rows; y++) {
      if (m_foregroundMask.at<uchar>(y, i) != 0) {
        minX = min(i, minX);
        minY = min(y, minY);
        maxX = max(i, maxX);
        maxY = max(y, maxY);
      }
    }
  }

  float cx = m_firstCentroid.x;  // minX + ((maxX - minX) / 2.0f);
  float cy = m_firstCentroid.y;  // minY + ((maxY - minY) / 2.0f);

  m_fstRelativeBBPoints.push_back(Point2f(minX - cx, minY - cy));
  m_fstRelativeBBPoints.push_back(Point2f(maxX - cx, minY - cy));
  m_fstRelativeBBPoints.push_back(Point2f(maxX - cx, maxY - cy));
  m_fstRelativeBBPoints.push_back(Point2f(minX - cx, maxY - cy));

  m_updatedBBPoints.push_back(Point2f(minX, minY));
  m_updatedBBPoints.push_back(Point2f(maxX, minY));
  m_updatedBBPoints.push_back(Point2f(maxX, maxY));
  m_updatedBBPoints.push_back(Point2f(minX, maxY));

  m_fstBBPoints = m_updatedBBPoints;
}

bool Tracker2D::learnFrame(float scale, float angle) {
  /*float scaleChange = scale - 1;
      float angleChange = angle;

      int scaleId = static_cast<int>(scaleChange /
     m_configFile.m_scaleLearnStep);
      int angleId = static_cast<int>(angleChange /
     m_configFile.m_angleLearnStep);

      m_debugFile << "Change step " << scaleId << " " << angleId << "\n";

      auto it = m_learnedFrames.find(pair<int, int>(scaleId, angleId));

      return it == m_learnedFrames.end();
  */
}

void Tracker2D::addLearnedFrame(float scale, float angle) {
  /*
      float scaleChange = scale - 1;
      float angleChange = angle;

      int scaleId = static_cast<int>(scaleChange /
     m_configFile.m_scaleLearnStep);
      int angleId = static_cast<int>(angleChange /
     m_configFile.m_angleLearnStep);

       m_learnedFrames.insert(pair<int, int>(scaleId, angleId));
   */
}

void Tracker2D::learnKeypoints(const int numMatches, const int numTracked,
                               const int numBoth, const Mat& grayImg,
                               const Mat& next,
                               const std::vector<cv::KeyPoint>& keypoints) {
  int confidence = numMatches + numBoth;
  vector<const KeyPoint*> newPoints;

  /*m_debugFile << "Learning " << learnFrame(m_mScale, m_mAngle) << " "
              << (m_validKeypoints > (m_initKPCount * 0.1)) << " "
              << (confidence > m_configFile.m_matchConfidence) <<
              "\n";


      if (learnFrame(m_mScale, m_mAngle) && m_validKeypoints > (m_initKPCount *
     0.1) &&
              confidence > m_configFile.m_matchConfidence)
      {
              Mat tmp, out;
              int rows = next.rows;
              int cols = next.cols;
              Mat1b mask(rows, cols, static_cast<uchar>(0));
              grayImg.copyTo(tmp);

              next.copyTo(out);

              drawTriangle(m_updatedBBPoints[0], m_updatedBBPoints[1],
     m_updatedBBPoints[2],
                      Scalar(0, 255, 0), 0.5, out);
              drawTriangle(m_updatedBBPoints[0], m_updatedBBPoints[2],
     m_updatedBBPoints[3],
                      Scalar(0, 0, 255), 0.5, out);
              drawTriangleMask(m_updatedBBPoints[0], m_updatedBBPoints[1],
     m_updatedBBPoints[2], mask);
              drawTriangleMask(m_updatedBBPoints[0], m_updatedBBPoints[2],
     m_updatedBBPoints[3], mask);

              line(out, m_updatedBBPoints[0], m_updatedBBPoints[1], Scalar(255,
     0, 0), 2);
              line(out, m_updatedBBPoints[1], m_updatedBBPoints[2], Scalar(255,
     0, 0), 2);
              line(out, m_updatedBBPoints[2], m_updatedBBPoints[3], Scalar(255,
     0, 0), 2);
              line(out, m_updatedBBPoints[3], m_updatedBBPoints[0], Scalar(255,
     0, 0), 2);

              int kpCount = 0;
              for (int i = 0; i < keypoints.size(); i++)
              {
                      if (mask.at<uchar>(keypoints[i].pt) ==
     static_cast<uchar>(255))
                      {
                              kpCount++;
                              //circle(out, keypoints[i].pt, 2, Scalar(255, 0,
     0), 1);
                              newPoints.push_back(&keypoints[i]);
                      }
              }


              // modify structures to include new keypoints
              int newSize = m_keypointStatus.size() + newPoints.size();
              random_device rd;
              default_random_engine engine(rd());
              uniform_int_distribution<unsigned int> uniform_dist(0, 255);

              Mat result;
              Mat fst;
              cvtColor(m_fstFrame, fst, CV_GRAY2BGR);
              cout << "Building composite image: " << out.channels() << " " <<
     fst.channels() << "\n";
              buildCompositeImg(out, fst, result);

              backprojectKeypoins(newPoints, result);

              int scaleId = static_cast<int>((m_mScale - 1) /
     m_configFile.m_scaleLearnStep);
              int angleId = static_cast<int>((m_mAngle) /
     m_configFile.m_angleLearnStep);

              stringstream ss;
              ss << m_numFrames << "( " << confidence << " )" << "[" << scaleId
     << ","
                      << angleId << "].png";
              imwrite(m_configFile.m_dstPath + "learn/" + ss.str(), result);

              addLearnedFrame(m_mScale, m_mAngle);
      }

  */
}

void Tracker2D::backprojectKeypoins(const vector<const KeyPoint*>& points,
                                    Mat& out) {
  Mat2f rotMat(2, 2);

  rotMat.at<float>(0, 0) = cosf(m_mAngle);
  rotMat.at<float>(0, 1) = -sinf(m_mAngle);
  rotMat.at<float>(1, 0) = sinf(m_mAngle);
  rotMat.at<float>(1, 1) = cosf(m_mAngle);

  random_device rd;
  default_random_engine engine(rd());
  uniform_int_distribution<unsigned int> uniform_dist(0, 255);

  for (size_t i = 0; i < points.size(); i++) {
    Point2f rm = points[i]->pt - m_updatedCentroid;
    if (m_mScale != 0) {
      rm.x = rm.x / m_mScale;
      rm.y = rm.y / m_mScale;
    }
    rm = mult(rotMat, rm);

    Point2f original = m_firstCentroid + rm;
    Point2f offSetpoint = original;
    offSetpoint.x += out.cols / 2;

    if (offSetpoint.x < 0 || offSetpoint.x >= out.cols) continue;
    if (offSetpoint.y < 0 || offSetpoint.y >= out.rows) continue;

    Scalar color = Scalar(uniform_dist(engine), uniform_dist(engine),
                          uniform_dist(engine));

    circle(out, points[i]->pt, 3, color, 1);
    circle(out, offSetpoint, 3, color, 1);
    line(out, points[i]->pt, offSetpoint, color, 1);

    /*
            if (m_configFile.m_useLearning)
            {
                    KeyPoint p;
                    p = *points[i];
                    p.pt = original;

                    m_keypointStatus.push_back(Status::MATCH);
                    m_firstFrameKeypoints.push_back(p);
                    m_updatedKeypoints.push_back(*points[i]);
                    m_matchedPoints.push_back(points[i]->pt);
                    m_trackedPoints.push_back(points[i]->pt);
                    m_centroidVotes.push_back(m_updatedCentroid);
                    m_clusteredCentroidVotes.push_back(false);
                    m_clusteredBorderVotes.push_back(false);
                    m_relativePointsPos.push_back(original - m_firstCentroid);
                    m_pointsColor.push_back(color);
                    m_initKPCount++;
            }
    */
  }
}

void Tracker2D::debugTrackingStep(const cv::Mat& fstFrame,
                                  const cv::Mat& scdFrame,
                                  const std::vector<int>& indices,
                                  std::vector<vector<int>>& clusters,
                                  Mat& out) {
  random_device rd;
  default_random_engine engine(rd());
  uniform_int_distribution<unsigned int> uniform_dist(0, 255);

  int cols = fstFrame.cols;
  int rows = fstFrame.rows;

  buildCompositeImg(fstFrame, scdFrame, out);

  int matchedPoints = 0;
  int trackedPoints = 0;
  int bothPoints = 0;

  if (m_validKeypoints > (m_initKPCount * 0.01)) {
    drawObjectLocation(m_firstCentroid, m_fstBBPoints, m_updatedCentroid,
                       m_updatedBBPoints, out);
  }

  /*if (m_configFile.m_showMatching)
      {
              drawKeypointsMatching(m_firstFrameKeypoints, m_updatedKeypoints,
  m_keypointStatus,
                      m_pointsColor, matchedPoints, trackedPoints, bothPoints,
                      m_configFile.m_drawMatchingLines, out);
      }
      else
      {
              countKeypointsMatching(m_firstFrameKeypoints, m_updatedKeypoints,
  m_keypointStatus,
                      matchedPoints, trackedPoints, bothPoints);
      }

      if (m_configFile.m_showVoting)
      {
              drawCentroidVotes(m_updatedKeypoints, m_centroidVotes,
  m_clusteredCentroidVotes,
                      m_clusteredBorderVotes, m_keypointStatus,
  m_configFile.m_drawVotingLines,
                      m_configFile.m_drawVotingFalse, out);
  }*/

  drawKeipointsStats(m_initKPCount, matchedPoints, trackedPoints, bothPoints,
                     out);

  int totM = matchedPoints + bothPoints;
  int totT = trackedPoints + bothPoints;

  drawInformationHeader(m_numFrames, m_mScale, m_mAngle, m_clusterVoteSize,
                        totM, totT, out);

  // m_debugFile << "Match " << totM << " Track " << totT
  //	<< " scale " << m_mScale << " angle " << m_mAngle << "\n";

  // m_debugWriter << out;
}

//#ifdef _WIN64
void Tracker2D::parDebugtrackingStep(const cv::Mat& fstFrame,
                                     const cv::Mat& scdFrame) {
  /*
  random_device rd;
      default_random_engine engine(rd());
      uniform_int_distribution<unsigned int> uniform_dist(0, 255);

      Mat out;

      unsigned int cols = scdFrame.cols;
      unsigned int rows = scdFrame.rows;

      Size size(cols * 2, rows);
      out.create(size, scdFrame.type());

      fstFrame.copyTo(out(Rect(0, 0, cols, rows)));
      scdFrame.copyTo(out(Rect(cols, 0, cols, rows)));

      cvtColor(out, out, CV_GRAY2BGR);

      int matchedPoints = 0;
      int trackedPoints = 0;
      int bothPoints = 0;

      parallel_for(blocked_range<size_t>(0, m_updatedKeypoints.size()),
              [&](const blocked_range<size_t>& r)
      {
              for (size_t i = r.begin(); i < r.end(); i++)
              {
                      Scalar color(uniform_dist(engine), uniform_dist(engine),
  uniform_dist(engine));

                      Status& s = m_keypointStatus[i];

                      Point2f & fst = m_firstFrameKeypoints[i].pt;
                      Point2f & scd = m_updatedKeypoints[i].pt;

                      Point2f fstOffset = fst;
                      fstOffset.x += cols;

                      if (s == Status::MATCH)
                      {
                              circle(out, fstOffset, 3, color, 1);
                              circle(out, scd, 3, color, 1);
                              line(out, scd, fstOffset, color, 1);
                              //matchedPoints++;
                      }
                      else if (s == Status::TRACK)
                      {
                              Rect prev(fst.x - 2 + cols, fst.y - 2, 5, 5);
                              Rect next(scd.x - 2, scd.y - 2, 5, 5);

                              rectangle(out, prev, color, 1);
                              rectangle(out, next, color, 1);
                              line(out, scd, fstOffset, color, 1);
                              //trackedPoints++;
                      }
                      else if (s == Status::BACKGROUND)
                      {
                              //circle(out, fstOffset, 3, color, 1);
                              //circle(out, scd, 3, color, 1);
                      }
                      else if (s == Status::BOTH)
                      {
                              cross(fstOffset, color, 1, out);
                              cross(scd, color, 1, out);
                              line(out, scd, fstOffset, color, 1);
                              //bothPoints++;
                      }

                      if (s == Status::BOTH || s == Status::MATCH || s ==
  Status::TRACK)
                      {
                              if (m_clusteredCentroidVotes[i])
                              {
                                      circle(out, m_centroidVotes[i], 3,
  Scalar(0, 255, 0), -1);
                                      line(out, m_centroidVotes[i], scd,
  Scalar(0, 255, 0), 1);
                              }
                              else
                              {
                                      circle(out, m_centroidVotes[i], 3,
  Scalar(0, 0, 255), -1);
                                      line(out, m_centroidVotes[i], scd,
  Scalar(0, 0, 255), 1);
                              }

                      }

                      Point2f tmp = m_firstCentroid;
                      tmp.x += cols;

                      Point2f offset(cols, 0);

                      circle(out, tmp, 5, Scalar(255, 0, 0), -1);
                      line(out, m_fstBBPoints[0] + offset, m_fstBBPoints[1] +
  offset, Scalar(255, 0, 0), 2);
                      line(out, m_fstBBPoints[1] + offset, m_fstBBPoints[2] +
  offset, Scalar(255, 0, 0), 2);
                      line(out, m_fstBBPoints[2] + offset, m_fstBBPoints[3] +
  offset, Scalar(255, 0, 0), 2);
                      line(out, m_fstBBPoints[3] + offset, m_fstBBPoints[0] +
  offset, Scalar(255, 0, 0), 2);

                      circle(out, m_updatedCentroid, 5, Scalar(255, 0, 0), -1);
                      line(out, m_updatedBBPoints[0], m_updatedBBPoints[1],
  Scalar(255, 0, 0), 2);
                      line(out, m_updatedBBPoints[1], m_updatedBBPoints[2],
  Scalar(255, 0, 0), 2);
                      line(out, m_updatedBBPoints[2], m_updatedBBPoints[3],
  Scalar(255, 0, 0), 2);
                      line(out, m_updatedBBPoints[3], m_updatedBBPoints[0],
  Scalar(255, 0, 0), 2);
              }

      });



      int histWidth = 30;
      int windowHeight = 50;
      int windowY = rows - windowHeight;
      int windowX = 2 * cols - histWidth * 3;

      Rect histR(windowX, windowY, 3 * histWidth, windowHeight);

      int nelems = 0, nMatch = 0, nTrack = 0, nboth = 0;
      nelems = bothPoints + matchedPoints + trackedPoints;

      rectangle(out, histR, Scalar(0, 0, 0), -1);

      if (m_initKPCount > 0)
      {
              nMatch = (windowHeight * (matchedPoints + bothPoints)) /
  m_initKPCount;
              nTrack = (windowHeight * (trackedPoints + bothPoints)) /
  m_initKPCount;
              nboth = (windowHeight * bothPoints) / m_initKPCount;

              cout << "nMatch " << nMatch << " nTrack " << nTrack << " nboth "
  << nboth << "\n";

              Rect mR(windowX, windowY + (windowHeight - nMatch), histWidth,
  nMatch);
              Rect tR(windowX + histWidth, windowY + (windowHeight - nTrack),
  histWidth, nTrack);
              Rect bR(windowX + 2 * histWidth, windowY + (windowHeight - nboth),
  histWidth, nboth);

              rectangle(out, mR, Scalar(255, 0, 0), -1);
              rectangle(out, tR, Scalar(0, 255, 0), -1);
              rectangle(out, bR, Scalar(0, 0, 255), -1);

      }

      stringstream ss;
      ss << "Frame: " << m_numFrames;

      rectangle(out, Rect(0, 0, 200, 40), Scalar(0, 0, 0), -1);

      putText(out, ss.str(), Point2f(10, 10), FONT_HERSHEY_PLAIN, 1,
              Scalar(255, 255, 255), 1);


      m_debugWriter << out;

      imshow("Debug Window", out);
  */
}
//#endif

void Tracker2D::updateKeypoints() {
  for (size_t i = 0; i < m_updatedKeypoints.size(); i++) {
    Status& s = m_keypointStatus[i];

    if (s == Status::MATCH) {
      m_updatedKeypoints[i].pt = m_matchedPoints[i];
    } else if (s == Status::TRACK || s == Status::LOST || s == Status::BOTH) {
      m_updatedKeypoints[i].pt = m_trackedPoints[i];
    }
  }
}

void Tracker2D::updateClusteredParams(float scale, float angle) {
  scale = 1;
  angle = 0;

  vector<float> angles;
  vector<float> scales;

  getClusteredEstimations(scales, angles);

  if (angles.size() > 1) {
    sort(angles.begin(), angles.end());
    sort(scales.begin(), scales.end());

    size_t size = angles.size();
    if (size % 2 == 0)
      angle = (angles[size / 2 - 1] + angles[size / 2]) / 2;
    else
      angle = angles[size / 2];
    size_t size2 = scales.size();
    if (size2 % 2 == 0)
      scale = (scales[size / 2 - 1] + scales[size / 2]) / 2;
    else
      scale = scales[size / 2];
  }
}

void Tracker2D::printKeypointStatus(string filename, string message) {
  ofstream file(filename, ios::app);

  file << message << "\n";
  file << "Keypoint generated: " << m_firstFrameKeypoints.size() << "\n";

  int match = 0, track = 0, init = 0, bg = 0, lost = 0, nomatch = 0;

  for (size_t i = 0; i < m_firstFrameKeypoints.size(); i++) {
    switch (m_keypointStatus[i]) {
      case Status::BACKGROUND:
        bg++;
        break;
      case Status::INIT:
        init++;
        break;
      case Status::MATCH:
        match++;
        break;
      case Status::TRACK:
        track++;
        break;
      case Status::LOST:
        lost++;
        break;
      case Status::NOMATCH:
        nomatch++;
        break;
    }
  }

  file << "BG " << bg << " init " << init << " match " << match << " track "
       << track << " lost " << lost << " nomatch " << nomatch << "\n";

  file.close();
}

void Tracker2D::drawResult(Mat& out) {
  // circle(out, m_updatedCentroid, 5, Scalar(0, 255, 0), -1);
  line(out, m_updatedBBPoints[0], m_updatedBBPoints[1], Scalar(0, 255, 0), 3);
  line(out, m_updatedBBPoints[1], m_updatedBBPoints[2], Scalar(0, 255, 0), 3);
  line(out, m_updatedBBPoints[2], m_updatedBBPoints[3], Scalar(0, 255, 0), 3);
  line(out, m_updatedBBPoints[3], m_updatedBBPoints[0], Scalar(0, 255, 0), 3);

  Scalar white(255, 255, 255);
  Scalar green(0, 255, 0);

  for (size_t i = 0; i < m_clusteredCentroidVotes.size(); i++) {
    if (isKeypointValid(i) && isKeypointValidCluster(i)) {
      cross(m_updatedKeypoints[i].pt, white, 1, out);
      cross(m_centroidVotes[i], green, 1, out);
    }
  }

  rectangle(out, Rect(0, 0, 640, 20), Scalar(0, 0, 0), -1);
  stringstream ss;
  ss.precision(2);
  ss << "F: " << m_numFrames << " S: " << m_mScale << " A: " << m_mAngle;

  putText(out, ss.str(), Point2f(10, 15), FONT_HERSHEY_PLAIN, 1,
          Scalar(255, 255, 255), 1);
}

void Tracker2D::debugCalculations() {
  /*ofstream file(m_configFile.m_dstPath + "calc/paramsCalculation.txt");
      ofstream median(m_configFile.m_dstPath + "calc/median.txt");
      ofstream vote(m_configFile.m_dstPath + "calc/vote.txt");

      vector<double> angles;
      vector<float> scales;

      for (size_t i = 0; i < m_updatedKeypoints.size(); ++i)
      {
              for (size_t j = 0; j < m_firstFrameKeypoints.size(); j++)
              {
                      if (isKeypointValid(i) && isKeypointValid(j) && i != j)
                      {

                              Point2f a = m_updatedKeypoints[i].pt -
     m_updatedKeypoints[j].pt;
                              Point2f b = m_firstFrameKeypoints[i].pt -
     m_firstFrameKeypoints[j].pt;
                              Point2f c = a - b;
                              angles.push_back(atan2(c.y, c.x));// - atan2(b.y,
     b.x));
                              float nextDistance =
     getDistance(m_updatedKeypoints[i], m_updatedKeypoints[j]);
                              float currDistance =
     getDistance(m_firstFrameKeypoints[i],
                                      m_firstFrameKeypoints[j]);

                              if (currDistance != 0 && i != j)
                              {
                                      scales.push_back(nextDistance /
     currDistance);
                              }



                              file << a.x << "," << a.y << "," << b.x << "," <<
     b.y << ","
                                      << atan2(c.y, c.x) << "," << (nextDistance
     / currDistance) << "\n";
                      }
              }
      }

      double medianAngle;
      float medianScale;

      if (angles.size() > 1)
      {
              sort(angles.begin(), angles.end());
              sort(scales.begin(), scales.end());

              size_t size = angles.size();
              if (size % 2 == 0)
                      medianAngle = (angles[size / 2 - 1] + angles[size / 2]) /
     2;
              else
                      medianAngle = angles[size / 2];
              size_t size2 = scales.size();
              if (size2 % 2 == 0)
                      medianScale = (scales[size / 2 - 1] + scales[size / 2]) /
     2;
              else
                      medianScale = scales[size / 2];
      }

      median << medianScale << " " << medianAngle;

      Mat2f rotMat(2, 2);

      rotMat.at<float>(0, 0) = cosf(medianAngle);
      rotMat.at<float>(0, 1) = sinf(medianAngle);
      rotMat.at<float>(1, 0) = -sinf(medianAngle);
      rotMat.at<float>(1, 1) = cosf(medianAngle);

      for (size_t i = 0; i < m_firstFrameKeypoints.size(); i++)
      {
              if (isKeypointValid(i))
              {
                      Point2f& a = m_updatedKeypoints[i].pt;
                      Point2f& rm = m_relativePointsPos[i];

                      m_centroidVotes[i] = a - medianScale * mult(rotMat, rm);
              }
      }

      file.close();
      median.close();
      vote.close();
  */
}

}  // end namespace
