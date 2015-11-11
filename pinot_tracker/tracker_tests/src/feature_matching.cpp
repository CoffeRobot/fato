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
#include <iostream>
#include <memory>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <algorithm>
#include <ros/ros.h>
#include <fstream>

#include "../../tracker/include/matcher.h"

#include "../../utilities/include/profiler.h"
#include "../../utilities/include/draw_functions.h"
#include "../../io/include/filemanager.h"
#include "../include/feature_matching.h"

using namespace cv;
using namespace std;

namespace pinot_tracker {

FeatureBenchmark::FeatureBenchmark() {}

FeatureBenchmark::~FeatureBenchmark() {}

void FeatureBenchmark::testVideo(string path) {
  vector<string> image_names;
  vector<Rect> boxes;

  parseGT(path + "groundtruth.txt", boxes);
  getFiles(path + "imgs/", ".png", image_names);

  sort(image_names.begin(), image_names.end());

  cout << "GT " << boxes.size() << " " << image_names.size() << endl;

  if (image_names.size() == 0) return;

  Mat img;
  img = imread(path + "imgs/" + image_names.at(0), 1);
  Mat gray;
  cvtColor(img, gray, CV_BGR2GRAY);

  initBrisk(gray, boxes.at(0));
  initOrb(gray, boxes.at(0));
  initSift(gray, boxes.at(0));

  for (auto i = 1; i < image_names.size(); ++i) {
    img = imread(path + "imgs/" + image_names.at(i), 1);

    // check if image is grayscale
    cvtColor(img, gray, CV_BGR2GRAY);

    if (i < boxes.size()) {
      matchBrisk(gray, boxes.at(i), img);
      matchOrb(gray, boxes.at(i), img);
      matchSift(gray, boxes.at(i), img);
      rectangle(img, boxes.at(i), Scalar(0, 0, 255), 3);
    }

    imshow("benchmark", img);
    waitKey(30);
  }

  saveVideoResult(path);
}

void FeatureBenchmark::saveVideoResult(string path) {
  auto &profiler = Profiler::getInstance();
  ofstream file(path + "benchmark.txt");
  file << "FEATURE | EXTRACTION | MATCHING | MODEL | TP | FP | FN | TN \n";

  float avg_ext = profiler->getTime("brisk_extract");
  float avg_mat = profiler->getTime("brisk_match");
  auto &result = resuls_.find("brisk")->second;
  file << "brisk | " << avg_ext << " | " << avg_mat << " | "
       << result.features_count << " | " << result.true_positive << " | "
       << result.false_positive << " | " << result.false_negative << "\n";

  brisk_keypoints_.clear();
  brisk_descriptors_ = Mat();
  brisk_label_.clear();

  avg_ext = profiler->getTime("orb_extract");
  avg_mat = profiler->getTime("orb_match");
  auto &oresult = resuls_.find("orb")->second;
  file << "orb | " << avg_ext << " | " << avg_mat << " | "
       << oresult.features_count << " | " << oresult.true_positive << " | "
       << oresult.false_positive << " | " << oresult.false_negative << "\n";

  orb_keypoints_.clear();
  orb_descriptors_ = Mat();
  orb_label_.clear();

  avg_ext = profiler->getTime("sift_extract");
  auto &sres = resuls_.find("sift")->second;
  file << "sift | " << avg_ext << " | " << sres.features_count << "\n";

  sift_keypoints_.clear();
  sift_descriptors_ = Mat();
  sift_label_.clear();
}

void FeatureBenchmark::parseGT(string path, std::vector<Rect> &ground_truth) {
  ifstream file(path);

  if (!file.is_open()) {
    cout << "Cannot open gt file in path: " << path << "!" << endl;

    return;
  }

  string line;
  while (getline(file, line)) {
    vector<string> words;
    boost::split(words, line, boost::is_any_of(","));

    if (words.size() == 4) {
      auto x = atof(words[0].c_str());
      auto y = atof(words[1].c_str());
      auto w = atof(words[2].c_str());
      auto h = atof(words[3].c_str());

      ground_truth.push_back(cv::Rect(Point2f(x, y), Point2f(x + w, y + h)));
    }
  }

  cout << "GT size " << ground_truth.size() << endl;
}

void FeatureBenchmark::initBrisk(const Mat &in, Rect &bbox) {
  BRISK brisk_detector;
  brisk_detector.detect(in, brisk_keypoints_);
  brisk_detector.compute(in, brisk_keypoints_, brisk_descriptors_);

  brisk_label_.resize(brisk_keypoints_.size(), false);

  BenchResults brisk_result;
  brisk_result.features_count = brisk_keypoints_.size();
  brisk_result.feature_name = "brisk";

  resuls_.insert(pair<string, BenchResults>("brisk", brisk_result));

  for (auto i = 0; i < brisk_keypoints_.size(); ++i) {
    auto &kp = brisk_keypoints_.at(i).pt;

    if (kp.x > bbox.tl().x && kp.x < bbox.br().x && kp.y > bbox.tl().y &&
        kp.y < bbox.br().y) {
      brisk_label_.at(i) = true;
    }
  }
}

void FeatureBenchmark::initOrb(const Mat &in, Rect &bbox) {
  orb_detector_.detect(in, orb_keypoints_);
  orb_detector_.compute(in, orb_keypoints_, orb_descriptors_);

  orb_label_.resize(orb_keypoints_.size(), false);

  BenchResults orb_result;
  orb_result.features_count = orb_keypoints_.size();
  orb_result.feature_name = "orb";

  resuls_.insert(pair<string, BenchResults>("orb", orb_result));

  for (auto i = 0; i < orb_keypoints_.size(); ++i) {
    auto &kp = orb_keypoints_.at(i).pt;

    if (kp.x > bbox.tl().x && kp.x < bbox.br().x && kp.y > bbox.tl().y &&
        kp.y < bbox.br().y) {
      orb_label_.at(i) = true;
    }
  }
}

void FeatureBenchmark::initSift(const Mat &in, Rect &bbox) {
  SIFT detector;
  detector(in, Mat(), sift_keypoints_, sift_descriptors_);

  BenchResults sift_result;
  sift_result.features_count = sift_keypoints_.size();
  sift_result.feature_name = "sift";

  sift_label_.resize(orb_keypoints_.size(), false);

  resuls_.insert(pair<string, BenchResults>("sift", sift_result));

  for (auto i = 0; i < sift_keypoints_.size(); ++i) {
    auto &kp = sift_keypoints_.at(i).pt;

    if (kp.x > bbox.tl().x && kp.x < bbox.br().x && kp.y > bbox.tl().y &&
        kp.y < bbox.br().y) {
      sift_label_.at(i) = true;
    }
  }
}

void FeatureBenchmark::matchBrisk(const Mat &in, Rect &bbox, Mat &out) {
  vector<KeyPoint> kps;
  Mat descriptors;

  Profiler::getInstance()->start("brisk_extract");
  brisk_detector_.detect(in, kps);
  brisk_detector_.compute(in, kps, descriptors);
  Profiler::getInstance()->stop("brisk_extract");

  vector<vector<DMatch>> matches;
  pinot_tracker::CustomMatcher cm;

  Profiler::getInstance()->start("brisk_match");
  cm.match(descriptors, brisk_descriptors_, 2, matches);
  Profiler::getInstance()->stop("brisk_match");

  auto &result = resuls_.find("brisk")->second;

  for (size_t i = 0; i < matches.size(); i++) {
    int queryId = matches[i][0].queryIdx;
    int trainId = matches[i][0].trainIdx;

    if (queryId < 0 && queryId >= kps.size()) continue;

    if (trainId < 0 && trainId >= brisk_keypoints_.size()) continue;

    float confidence = 1 - (matches[i][0].distance / 512.0);
    float ratio = matches[i][0].distance / matches[i][1].distance;

    if (confidence >= 0.80f && ratio <= 0.8) {
      auto &obj_pt = brisk_keypoints_.at(trainId).pt;
      auto &pt = kps.at(queryId).pt;

      if (brisk_label_.at(trainId)) {
        if (pt.x > bbox.tl().x && pt.x < bbox.br().x && pt.y > bbox.tl().y &&
            pt.y < bbox.br().y) {
          // true positive
          circle(out, pt, 3, Scalar(0, 255, 0));
          result.true_positive++;
        } else {
          // false positive
          circle(out, pt, 3, Scalar(0, 0, 255));
          result.false_positive++;
        }
      } else {
        if (pt.x > bbox.tl().x && pt.x < bbox.br().x && pt.y > bbox.tl().y &&
            pt.y < bbox.br().y) {
          // false negative
          circle(out, pt, 3, Scalar(255, 0, 0));
          result.false_negative++;
        }
      }
    }
  }
}

void FeatureBenchmark::matchOrb(const Mat &in, Rect &bbox, Mat &out) {
  vector<KeyPoint> kps;
  Mat descriptors;

  Profiler::getInstance()->start("orb_extract");
  orb_detector_.detect(in, kps);
  orb_detector_.compute(in, kps, descriptors);
  Profiler::getInstance()->stop("orb_extract");

  vector<vector<DMatch>> matches;
  pinot_tracker::CustomMatcher cm;
  Profiler::getInstance()->start("orb_match");
  cm.match32(descriptors, orb_descriptors_, 2, matches);
  Profiler::getInstance()->stop("orb_match");

  auto &result = resuls_.find("orb")->second;

  for (size_t i = 0; i < matches.size(); i++) {
    int queryId = matches[i][0].queryIdx;
    int trainId = matches[i][0].trainIdx;

    if (queryId < 0 && queryId >= kps.size()) continue;

    if (trainId < 0 && trainId >= orb_keypoints_.size()) continue;

    float confidence = 1 - (matches[i][0].distance / 256.0);
    float ratio = matches[i][0].distance / matches[i][1].distance;

    if (confidence >= 0.80f && ratio <= 0.8) {
      auto &obj_pt = orb_keypoints_.at(trainId).pt;
      auto &pt = kps.at(queryId).pt;

      if (orb_label_.at(trainId)) {
        if (pt.x > bbox.tl().x && pt.x < bbox.br().x && pt.y > bbox.tl().y &&
            pt.y < bbox.br().y) {
          // true positive
          cross(out, pt, Scalar(0, 255, 0));
          result.true_positive++;
        } else {
          // false positive
          cross(out, pt, Scalar(0, 0, 255));
          result.false_positive++;
        }
      } else {
        if (pt.x > bbox.tl().x && pt.x < bbox.br().x && pt.y > bbox.tl().y &&
            pt.y < bbox.br().y) {
          // false negative
          cross(out, pt, Scalar(255, 0, 0));
          result.false_negative++;
        }
      }
    }
  }
}

void FeatureBenchmark::matchSift(const Mat &in, Rect &bbox, Mat &out) {
  vector<KeyPoint> kps;
  Mat descriptors;

  SIFT detector;

  Profiler::getInstance()->start("sift_extract");
  detector(in, Mat(), kps, descriptors);
  Profiler::getInstance()->stop("sift_extract");
}

}  // end namespace

int main(int argc, char *argv[]) {
  ROS_INFO("Starting tracker input");
  ros::init(argc, argv, "pinot_tracker_feature_matching");

  vector<Rect> gt;

  pinot_tracker::FeatureBenchmark fb;
  fb.testVideo("/media/alessandro/Super Fat/Dataset/tracking_clean/ball/");

  return 0;
}
