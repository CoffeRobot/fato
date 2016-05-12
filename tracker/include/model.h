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

#ifndef OBJECTMODEL_H
#define OBJECTMODEL_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef __unix__
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#else
#include <Eigen/Dense>
#endif

#include "../../utilities/include/constants.h"

namespace fato{

class ObjectModel {
 public:
  ObjectModel();
  virtual ~ObjectModel();

  void initCube(cv::Point3f& centroid, std::vector<cv::Point3f>& front,
                std::vector<cv::Point3f>& back);

  std::vector<cv::Point3f> getFacePoints(int face);

  std::vector<bool> getVisibility(const cv::Mat& pov);

  void restCube();

  void resetFace(int face) {
    m_cloudPoints[face].clear();
    m_pointStatus[face].clear();
    m_faceDescriptors[face] = cv::Mat();
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
  std::vector<std::vector<FatoStatus> > m_pointStatus;
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
