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

using namespace std;
using namespace cv;

namespace fato {

FeatureMatcher::~FeatureMatcher() {}

BriskMatcher::BriskMatcher() { feature_id_ = -1; }

BriskMatcher::~BriskMatcher() {}

void BriskMatcher::init(int feature_id) {
  feature_id_ = feature_id;

  initExtractor();
}

void BriskMatcher::setTarget(const Mat &img) {
  if (feature_id_ == -1) {
    cerr << "feature_matcher: not initialized properly, params and feature "
            "type need to be set!" << endl;
    return;
  }

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

void BriskMatcher::initExtractor() {
  switch (feature_id_) {
    case BRISK:
      cout << "FeatureMatcher: BRISK initialization " << endl;
      feature_name = "brisk";
      matcher_ = DescriptorMatcher::create("BruteForce-Hamming");
      break;
    default:
      break;
  }
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
  matcher_->knnMatch(train_descriptors_, query_descriptors, matches, 2);
}

std::vector<cv::KeyPoint> &BriskMatcher::getTrainingPoints() {
  return train_keypoints_;
}

cv::Mat &BriskMatcher::getTrainingDescriptors() { return train_descriptors_; }

}  // end namespace
