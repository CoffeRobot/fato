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

#include "../include/feature_matcher.hpp"
#include "../../cuda_sift/include/cudaImage.h"
#include "../include/profiler.h"

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>

#include "AKAZE.h"

using namespace std;
using namespace cv;
using namespace std::chrono;

namespace fato {

FeatureMatcher::~FeatureMatcher() {}

// ============================================= ===== ================================================ //
// ============================================= BRISK ================================================ //
// ============================================= ===== ================================================ //

BriskMatcher::BriskMatcher()
 {

#ifdef __arm__
  cv_matcher_ = cv::BFMatcher(NORM_HAMMING);
  #endif


    feature_id_ = -1;
    initExtractor();
}

BriskMatcher::~BriskMatcher() {}

void BriskMatcher::init(int feature_id) {
  feature_id_ = feature_id;

  initExtractor();
}

void BriskMatcher::extractTarget(const Mat &img) {

  if (img.empty()) {
    cerr << "feature_matcher: image is empty!" << endl;
    return;
  }

  if (img.channels() > 1) {
    cerr << "feature_matcher: image must be grayscale!" << endl;
    return;
  }

  if (train_keypoints_.size() > 0) train_keypoints_.clear();
  if (train_descriptors_.rows > 0) train_descriptors_ = Mat();

  extract(img, train_keypoints_, train_descriptors_);
}

void BriskMatcher::setTarget(const Mat& descriptors)
{
    train_descriptors_ = descriptors.clone();
}

void BriskMatcher::initExtractor() {

      cout << "FeatureMatcher: BRISK initialization " << endl;
      feature_name = "brisk";
}

void BriskMatcher::extract(const Mat &img, std::vector<KeyPoint> &keypoints,
                        Mat &descriptors) {
  opencv_detector_.detect(img, keypoints);
  opencv_detector_.compute(img, keypoints, descriptors);
}

void BriskMatcher::match(const Mat &img, std::vector<KeyPoint> &query_keypoints,
                      Mat &query_descriptors,
                      std::vector<vector<DMatch>> &matches) {  
  extract(img, query_keypoints, query_descriptors);
  if (query_descriptors.rows == 0) {
    return;
  }
  //TOFIX: opencv matcher does not work in 2.4
 #ifdef __arm__
  cv_matcher_.knnMatch(query_descriptors, train_descriptors_, matches, 2);
#else
  matcher_custom_.matchV2(query_descriptors, train_descriptors_, matches);
#endif
}

std::pair<float,float> BriskMatcher::matchP(const Mat &img, std::vector<KeyPoint> &query_keypoints,
                      Mat &query_descriptors,
                      std::vector<vector<DMatch>> &matches) {

  auto begin = high_resolution_clock::now();
  extract(img, query_keypoints, query_descriptors);
  auto end = high_resolution_clock::now();
  float ext_time = duration_cast<nanoseconds>(end-begin).count();
  if (query_descriptors.rows == 0) {
    return pair<float,float>(ext_time, 0);
  }
  //TOFIX: opencv matcher does not work in 2.4
  begin = high_resolution_clock::now();
#ifdef __arm__
  cv_matcher_.knnMatch(query_descriptors, train_descriptors_, matches, 2);
#else
  matcher_custom_.matchV2(query_descriptors, train_descriptors_, matches);
#endif
  end = high_resolution_clock::now();
  float mtc_time = duration_cast<nanoseconds>(end-begin).count();

  return pair<float,float>(ext_time, mtc_time);
}

std::vector<cv::KeyPoint> &BriskMatcher::getTargetPoints() {
  return train_keypoints_;
}

cv::Mat &BriskMatcher::getTargetDescriptors() { return train_descriptors_; }


// ============================================= ===== ================================================ //
// ============================================= Akaze ================================================ //
// ============================================= ===== ================================================ //

AkazeMatcher::AkazeMatcher()
 {

#ifdef __arm__
  cv_matcher_ = cv::BFMatcher(NORM_HAMMING);
  #endif


    feature_id_ = -1;
    initExtractor();
}

AkazeMatcher::~AkazeMatcher() {

    delete akaze_detector_;
}

void AkazeMatcher::init(int feature_id) {
  feature_id_ = feature_id;

  initExtractor();
}

void AkazeMatcher::extractTarget(const Mat &img) {

  if (img.empty()) {
    cerr << "feature_matcher: image is empty!" << endl;
    return;
  }

  if (img.channels() > 1) {
    cerr << "feature_matcher: image must be grayscale!" << endl;
    return;
  }

  if (train_keypoints_.size() > 0) train_keypoints_.clear();
  if (train_descriptors_.rows > 0) train_descriptors_ = Mat();

  extract(img, train_keypoints_, train_descriptors_);
}

void AkazeMatcher::setTarget(const Mat& descriptors)
{
    train_descriptors_ = descriptors.clone();
}

void AkazeMatcher::initExtractor() {
    AKAZEOptions options;

    options.img_height = 480;
    options.img_width = 640;

    akaze_detector_ = new libAKAZECU::AKAZE(options);

    cout << "FeatureMatcher: CUDA AKAZE initialization " << endl;
    feature_name = "cuda_akaze";
}

void AkazeMatcher::extract(const Mat &img, std::vector<KeyPoint> &keypoints,
                        Mat &descriptors) {
    cv::Mat img_32;
    img.convertTo(img_32, CV_32F, 1.0 / 255.0, 0);


  akaze_detector_->Create_Nonlinear_Scale_Space(img_32);
  akaze_detector_->Feature_Detection(keypoints);
  akaze_detector_->Compute_Descriptors(keypoints,descriptors);

}

void AkazeMatcher::match(const Mat &img, std::vector<KeyPoint> &query_keypoints,
                      Mat &query_descriptors,
                      std::vector<vector<DMatch>> &matches) {
  extract(img, query_keypoints, query_descriptors);
  if (query_descriptors.rows == 0) {
    return;
  }
  //TOFIX: opencv matcher does not work in 2.4
 #ifdef __arm__
  cv_matcher_.knnMatch(query_descriptors, train_descriptors_, matches, 2);
#else
  matcher_custom_.matchV2(query_descriptors, train_descriptors_, matches);
#endif
}

std::pair<float,float> AkazeMatcher::matchP(const Mat &img, std::vector<KeyPoint> &query_keypoints,
                      Mat &query_descriptors,
                      std::vector<vector<DMatch>> &matches) {

  auto begin = high_resolution_clock::now();
  extract(img, query_keypoints, query_descriptors);
  auto end = high_resolution_clock::now();
  float ext_time = duration_cast<nanoseconds>(end-begin).count();
  if (query_descriptors.rows == 0) {
    return pair<float,float>(ext_time, 0);
  }
  //TOFIX: opencv matcher does not work in 2.4
  begin = high_resolution_clock::now();
#ifdef __arm__
  cv_matcher_.knnMatch(query_descriptors, train_descriptors_, matches, 2);
#else
  matcher_custom_.matchV2(query_descriptors, train_descriptors_, matches);
#endif
  end = high_resolution_clock::now();
  float mtc_time = duration_cast<nanoseconds>(end-begin).count();

  return pair<float,float>(ext_time, mtc_time);
}

std::vector<cv::KeyPoint> &AkazeMatcher::getTargetPoints() {
  return train_keypoints_;
}

cv::Mat &AkazeMatcher::getTargetDescriptors() { return train_descriptors_; }




// ============================================= === ================================================ //
// ============================================= ORB ================================================ //
// ============================================= === ================================================ //

OrbMatcher::OrbMatcher() {
    feature_id_ = -1;
    initExtractor();
}

OrbMatcher::~OrbMatcher() {}


void OrbMatcher::extractTarget(const Mat &img) {

  if (img.empty()) {
    cerr << "feature_matcher: image is empty!" << endl;
    return;
  }

  if (img.channels() > 1) {
    cerr << "feature_matcher: image must be grayscale!" << endl;
    return;
  }

  if (train_keypoints_.size() > 0) train_keypoints_.clear();
  if (train_descriptors_.rows > 0) train_descriptors_ = Mat();

  extract(img, train_keypoints_, train_descriptors_);
}

void OrbMatcher::setTarget(const Mat& descriptors)
{
    train_descriptors_ = descriptors.clone();
}

void OrbMatcher::initExtractor() {

      cout << "FeatureMatcher: ORB initialization " << endl;
      feature_name = "orb";
      opencv_detector_ = cv::ORB(500,1.2,3);

}

void OrbMatcher::extract(const Mat &img, std::vector<KeyPoint> &keypoints,
                        Mat &descriptors) {
  opencv_detector_.detect(img, keypoints);
  opencv_detector_.compute(img, keypoints, descriptors);
}

void OrbMatcher::match(const Mat &img, std::vector<KeyPoint> &query_keypoints,
                      Mat &query_descriptors,
                      std::vector<vector<DMatch>> &matches) {
  extract(img, query_keypoints, query_descriptors);
  if (query_descriptors.rows == 0) {
    return;
  }
  //TOFIX: opencv matcher does not work in 2.4
  //matcher_->knnMatch(train_descriptors_, query_descriptors, matches, 1);
  matcher_custom_.match32(train_descriptors_, query_descriptors, 2, matches);
}

std::vector<cv::KeyPoint> &OrbMatcher::getTargetPoints() {
  return train_keypoints_;
}

cv::Mat &OrbMatcher::getTargetDescriptors() { return train_descriptors_; }

}  // end namespace
