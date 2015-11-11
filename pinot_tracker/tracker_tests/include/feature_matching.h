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
#ifndef FEATURE_MATCHING_H
#define FEATURE_MATCHING_H

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <map>

namespace pinot_tracker {

struct BenchResults {
  int true_positive;
  int false_positive;
  int true_negative;
  int false_negative;
  float average_time;
  int features_count;
  std::string feature_name;

  BenchResults()
      : true_positive(0),
        false_positive(0),
        true_negative(0),
        false_negative(0),
        average_time(0.0),
        features_count(0),
        feature_name(""){};
};

class FeatureBenchmark {
 public:
  FeatureBenchmark();

  virtual ~FeatureBenchmark();

  void testVideo(std::string path);

 private:
  void saveVideoResult(std::string path);


  void parseGT(std::string path, std::vector<cv::Rect>& ground_truth);

  void initBrisk(const cv::Mat& in, cv::Rect& bbox);

  void initOrb(const cv::Mat& in, cv::Rect& bbox);

  void initSift(const cv::Mat& in, cv::Rect& bbox);

  void matchBrisk(const cv::Mat& in, cv::Rect& bbox, cv::Mat& out);

  void matchOrb(const cv::Mat& in, cv::Rect& bbox, cv::Mat& out);

  void matchSift(const cv::Mat& in, cv::Rect& bbox, cv::Mat& out);

  cv::BRISK brisk_detector_;
  cv::ORB orb_detector_;


  std::vector<cv::KeyPoint> brisk_keypoints_, orb_keypoints_, sift_keypoints_;
  cv::Mat brisk_descriptors_, orb_descriptors_, sift_descriptors_;
  std::vector<bool> brisk_label_, orb_label_, sift_label_;


  std::map<std::string, BenchResults> resuls_;
};
}

#endif  // FEATURE_MATCHING_H
