#ifndef BORGCUBE_H
#define BORGCUBE_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef __unix__
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#else
#include <Eigen/Dense>
#endif

#include "../../utilities/include/Constants.h"

namespace pinot_tracker{

class BorgCube {
 public:
  BorgCube();
  virtual ~BorgCube();

  void initCube(cv::Point3f& centroid, std::vector<cv::Point3f>& front,
                std::vector<cv::Point3f>& back);

  std::vector<cv::Point3f> getFacePoints(int face);

  std::vector<bool> getVisibility(const Mat& pov);

  void resetFace(int face) {
    m_cloudPoints[face].clear();
    m_pointStatus[face].clear();
    m_faceDescriptors[face] = Mat();
    m_faceKeypoints[face].clear();
    m_relativePointsPos[face].clear();
  };

  cv::Point3f m_center;

  float m_width;
  float m_height;
  float m_depth;

  std::vector<cv::Point3f> m_pointsFront;
  std::vector<cv::Point3f> m_pointsBack;
  std::vector<cv::Point3f> m_relativeDistFront;
  std::vector<cv::Point3f> m_relativeDistBack;

  cv::Mat m_faceNormals;
  Eigen::MatrixXd m_eigNormals;

  /*********************************************************************************************/
  /*                           INFORMATIONS FOR EACH FACE */
  /*********************************************************************************************/
  std::vector<std::vector<cv::Point3f> > m_cloudPoints;
  std::vector<std::vector<Status> > m_pointStatus;
  std::vector<cv::Mat> m_faceDescriptors;
  std::vector<std::vector<cv::KeyPoint> > m_faceKeypoints;
  std::vector<std::vector<cv::Point3f> > m_relativePointsPos;
  std::vector<bool> m_isLearned;
  std::vector<float> m_appearanceRatio;

 private:
  void initNormals();
};

} // end namespace

#endif
