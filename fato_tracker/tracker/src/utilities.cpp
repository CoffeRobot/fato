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

#include "../include/utilities.h"
#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;


namespace fato {

cv::Mat1b getMask(int rows, int cols, const cv::Point2d& begin,
                  const cv::Point2d& end) {
  cv::Mat1b mask(rows, cols, static_cast<uchar>(0));
  rectangle(mask, begin, end, static_cast<uchar>(255), -1);
  return mask;
}

void parseGT(string path, std::vector<cv::Rect> &ground_truth) {
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

      ground_truth.push_back(cv::Rect(cv::Point2f(x, y), cv::Point2f(x + w, y + h)));
    }
  }

  cout << "GT size " << ground_truth.size() << endl;
}

void parseGT(
    string path, std::vector<std::vector<cv::Point2f>> &ground_truth) {
  ifstream file(path);

  if (!file.is_open()) {
    cout << "Cannot open gt file in path: " << path << "!" << endl;

    return;
  }

  string line;
  while (getline(file, line)) {
    vector<string> words;
    boost::split(words, line, boost::is_any_of(","));

    if (words.size() == 8) {
      vector<cv::Point2f> points;
      points.push_back(cv::Point2f(atoi(words[0].c_str()), atoi(words[1].c_str())));
      points.push_back(cv::Point2f(atoi(words[2].c_str()), atoi(words[3].c_str())));
      points.push_back(cv::Point2f(atoi(words[4].c_str()), atoi(words[5].c_str())));
      points.push_back(cv::Point2f(atoi(words[6].c_str()), atoi(words[7].c_str())));

      ground_truth.push_back(points);
    }
  }

  cout << "GT size " << ground_truth.size() << endl;
}


void opencvToEigen(const cv::Mat& rot, Eigen::Matrix3d& rotation) {
  rotation = Eigen::Matrix3d(3, 3);

  rotation(0, 0) = static_cast<double>(rot.at<float>(0, 0));
  rotation(0, 1) = static_cast<double>(rot.at<float>(0, 1));
  rotation(0, 2) = static_cast<double>(rot.at<float>(0, 2));
  rotation(1, 0) = static_cast<double>(rot.at<float>(1, 0));
  rotation(1, 1) = static_cast<double>(rot.at<float>(1, 1));
  rotation(1, 2) = static_cast<double>(rot.at<float>(1, 2));
  rotation(2, 0) = static_cast<double>(rot.at<float>(2, 0));
  rotation(2, 1) = static_cast<double>(rot.at<float>(2, 1));
  rotation(2, 2) = static_cast<double>(rot.at<float>(2, 2));
}

void eigenToOpencv(const Eigen::Matrix3d& src, cv::Mat& dst) {
  dst = cv::Mat(3, 3, CV_32FC1, 0.0f);

  dst.at<float>(0, 0) = static_cast<float>(src(0, 0));
  dst.at<float>(0, 1) = static_cast<float>(src(0, 1));
  dst.at<float>(0, 2) = static_cast<float>(src(0, 2));

  dst.at<float>(1, 0) = static_cast<float>(src(1, 0));
  dst.at<float>(1, 1) = static_cast<float>(src(1, 1));
  dst.at<float>(1, 2) = static_cast<float>(src(1, 2));

  dst.at<float>(2, 0) = static_cast<float>(src(2, 0));
  dst.at<float>(2, 1) = static_cast<float>(src(2, 1));
  dst.at<float>(2, 2) = static_cast<float>(src(2, 1));
}

cv::Point2f projectPoint(const float focal, const cv::Point2f& center,
                         const cv::Point3f& src) {
  cv::Point2f dst;

  dst.x = (focal * src.x / src.z) + center.x;
  dst.y = (center.y - (focal * src.y / src.z));

  return dst;
}

cv::Point2f projectPoint(const float focal, const cv::Point2f& center,
                         const cv::Point3f* src) {
  cv::Point2f dst;

  dst.x = (focal * src->x / src->z) + center.x;
  dst.y = (center.y - (focal * src->y / src->z));

  return dst;
}

bool projectPoint(const float focal, const cv::Point2f& center,
                  const cv::Point3f& src, cv::Point2f& dst) {
  if (src.z == 0) return false;

  dst.x = (focal * src.x / src.z) + center.x;
  dst.y = (center.y - (focal * src.y / src.z));

  if (dst.x < 0 || dst.x > center.x * 2) return false;
  if (dst.y < 0 || dst.y > center.y * 2) return false;

  if (isnan(dst.x) || isnan(dst.y)) return false;

  return true;
}

bool projectPoint(const float focal, const cv::Point2f& center,
                  const cv::Point3f* src, cv::Point2f& dst) {
  if (src->z == 0) return false;

  dst.x = (focal * src->x / src->z) + center.x;
  dst.y = (center.y - (focal * src->y / src->z));

  if (dst.x < 0 || dst.x > center.x * 2) return false;
  if (dst.y < 0 || dst.y > center.y * 2) return false;

  if (isnan(dst.x) || isnan(dst.y)) return false;

  return true;
}

}  // end namespace
