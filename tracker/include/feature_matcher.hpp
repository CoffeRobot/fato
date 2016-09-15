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

#ifndef FEATURE_MATCHER_HPP
#define FEATURE_MATCHER_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <string>
#include <iostream>
#include <memory>

#include "config.h"
#include "matcher.h"

namespace libAKAZECU {
    class AKAZE;
}

namespace fato {

enum FEATURE_TYPE {
  BRISK = 0,
  ORB,
  AKAZE,
  SURF,
  SIFT,
  AKAZE_CUS,
  CUDA_SIFT,
  CUDA_AKAZE,
  CUDA_ORB,
  FEATURE_NUM
};

struct Match {
  int query_id;
  int train_id;
  float confidence;  // score of the match
  float ambiguity;   // ratio with the second best match

  friend std::ostream& operator<<(std::ostream& out, const Match& m) {
    out << "fst " << m.query_id << " scd " << m.train_id << " conf "
        << m.confidence << " amb " << m.ambiguity << "\n";
    return out;
  }
};

static std::string featureId2Name(int id) {
  switch (id) {
    case FEATURE_TYPE::BRISK:
      return "brisk";
      break;
    case FEATURE_TYPE::ORB:
      return "orb";
      break;
    case FEATURE_TYPE::AKAZE:
      return "akaze";
      break;
    case FEATURE_TYPE::SURF:
      return "surf";
      break;
    case FEATURE_TYPE::SIFT:
      return "sift";
      break;
    case FEATURE_TYPE::AKAZE_CUS:
      return "akazec";
      break;
    case FEATURE_TYPE::CUDA_SIFT:
      return "cuda_sift";
      break;
    case FEATURE_TYPE::CUDA_AKAZE:
      return "cuda_akaze";
      break;
    case FEATURE_TYPE::CUDA_ORB:
      return "cuda_orb";
      break;
    default:
      break;
  }
}

class FeatureMatcher {
 public:
  virtual ~FeatureMatcher() = 0;

  virtual void extractTarget(const cv::Mat& img) = 0;

    virtual void setTarget(const cv::Mat& descriptors) = 0;

  virtual void match(const cv::Mat& img,
                     std::vector<cv::KeyPoint>& query_keypoints,
                     cv::Mat& query_descriptors,
                     std::vector<std::vector<cv::DMatch>>& matches) = 0;

  virtual std::pair<float, float> matchP(
      const cv::Mat& img, std::vector<cv::KeyPoint>& query_keypoints,
      cv::Mat& query_descriptors,
      std::vector<std::vector<cv::DMatch>>& matches) = 0;

  virtual void extract(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints,
                       cv::Mat& descriptors) = 0;

  virtual std::vector<cv::KeyPoint>& getTargetPoints() = 0;
  virtual cv::Mat& getTargetDescriptors() = 0;
  virtual float maxDistance() = 0;

 protected:
  FeatureMatcher(){};
};



class AkazeMatcher : public FeatureMatcher {
public:
 AkazeMatcher();

 ~AkazeMatcher();

 void init(int feature_id);

 void extractTarget(const cv::Mat& img);

    void setTarget(const cv::Mat& descriptors);

 void match(const cv::Mat& img, std::vector<cv::KeyPoint>& query_keypoints,
            cv::Mat& query_descriptors,
            std::vector<std::vector<cv::DMatch>>& matches);

 std::pair<float, float> matchP(const cv::Mat& img,
                                std::vector<cv::KeyPoint>& query_keypoints,
                                cv::Mat& query_descriptors,
                                std::vector<std::vector<cv::DMatch>>& matches);

 std::vector<cv::KeyPoint>& getTargetPoints();

 cv::Mat& getTargetDescriptors();

 cv::Mat getTrainDescriptors();

 void extract(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints,
              cv::Mat& descriptors);

    float maxDistance(){return 486.f;}

private:

#ifdef __arm__
 cv::BFMatcher cv_matcher_;
#else
 CustomMatcher matcher_custom_;
#endif

 int feature_id_;
 std::string feature_name;
 libAKAZECU::AKAZE* akaze_detector_;

 cv::Mat train_descriptors_;
 std::vector<cv::KeyPoint> train_keypoints_;

    cv::Mat train_desc_gpu_;
    cv::Mat query_desc_gpu_;

 void initExtractor();
};


class BriskMatcher : public FeatureMatcher {
 public:
  BriskMatcher();

  ~BriskMatcher();

  void init(int feature_id);

  void extractTarget(const cv::Mat& img);

  void setTarget(const cv::Mat& descriptors);

  void match(const cv::Mat& img, std::vector<cv::KeyPoint>& query_keypoints,
             cv::Mat& query_descriptors,
             std::vector<std::vector<cv::DMatch>>& matches);

  std::pair<float, float> matchP(const cv::Mat& img,
                                 std::vector<cv::KeyPoint>& query_keypoints,
                                 cv::Mat& query_descriptors,
                                 std::vector<std::vector<cv::DMatch>>& matches);

  std::vector<cv::KeyPoint>& getTargetPoints();

  cv::Mat& getTargetDescriptors();

  cv::Mat getTrainDescriptors();

  void extract(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints,
               cv::Mat& descriptors);

  float maxDistance(){return 512.0;}

 private:

#ifdef __arm__
  cv::BFMatcher cv_matcher_;
#else
  CustomMatcher matcher_custom_;
#endif

  int feature_id_;
  std::string feature_name;
  cv::BRISK opencv_detector_;

  cv::Mat train_descriptors_;
  std::vector<cv::KeyPoint> train_keypoints_;

  void initExtractor();
};

class OrbMatcher : public FeatureMatcher {
 public:
  OrbMatcher();

  ~OrbMatcher();

  void extractTarget(const cv::Mat& img);

  void setTarget(const cv::Mat& descriptors);

  void match(const cv::Mat& img, std::vector<cv::KeyPoint>& query_keypoints,
             cv::Mat& query_descriptors,
             std::vector<std::vector<cv::DMatch>>& matches);

  std::vector<cv::KeyPoint>& getTargetPoints();

  cv::Mat& getTargetDescriptors();

  void extract(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints,
               cv::Mat& descriptors);

  float maxDistance(){return 256.0;}

 private:
  CustomMatcher matcher_custom_;

  int feature_id_;
  std::string feature_name;
  cv::ORB opencv_detector_;

  cv::Mat train_descriptors_;
  std::vector<cv::KeyPoint> train_keypoints_;

  void initExtractor();
};

}  // end namespace

#endif  // FEATURE_MATCHER_HPP
