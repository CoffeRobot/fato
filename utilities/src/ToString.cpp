/*****************************************************************************/
/*  Copyright (c) 2016, Alessandro Pieropan                                  */
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

#include "../include/ToString.h"

#include <iomanip>

using namespace std;
using namespace cv;
using namespace Eigen;

namespace fato{

string toString(const Point2f& p)
{
	stringstream ss;
	ss << fixed << setprecision(2) <<"[" << p.x << "," << p.y << "] ";
	return ss.str();
}


string toString(const FatoStatus& s)
{
	switch (s)
	{

	case FatoStatus::BACKGROUND:
		return "BACKGROUND";
	case FatoStatus::INIT:
		return "INIT";
	case FatoStatus::MATCH:
		return "MATCH";
	case FatoStatus::NOMATCH:
		return "NOMATCH";
	case FatoStatus::NOCLUSTER:
		return "NOCLUSTER";
	case FatoStatus::TRACK:
		return "TRACK";
	default:
		return "LOST";
	}
}

string toString(const Point3f& point)
{
	stringstream ss;

	ss << "[" << point.x << "," << point.y << "," << point.z << "] ";

	return ss.str();
}

string toString(const Vec3f& point)
{
	stringstream ss;

	ss << "[" << point[0] << "," << point[1] << "," << point[2] << "] ";

	return ss.str();
}

std::string toString(const Matrix3d& rotation) {
  stringstream ss;
  ss << "[";
  for (int i = 0; i < rotation.rows(); ++i) {
    ss << "[";
    for (int j = 0; j < rotation.cols(); j++) {
      ss << rotation(i, j);
      if (j < rotation.cols() - 1) ss << ",";
    }
    ss << "]";
    if (i < rotation.rows() - 1) ss << ",";
  }
  ss << "]";

  return ss.str();
}

std::string toString(const Quaterniond& quaternion) {
  stringstream ss;

  ss << "[" << quaternion.w() << "," << quaternion.x() << "," << quaternion.y()
     << "," << quaternion.z() << "]";

  return ss.str();
}

string toPythonString(const Mat& rotation) {
  if (rotation.cols != 3 || rotation.rows != 3) return "";

  stringstream ss;
  ss << "[";
  for (size_t i = 0; i < 3; i++) {
    ss << "[";
    for (size_t j = 0; j < 3; j++) {
      ss << rotation.at<float>(i, j);
      if (j < 2) ss << ",";
    }
    ss << " ]";
    if (i < 2) ss << ",";
  }
  ss << "]";

  return ss.str();
}

string toPythonArray(const Mat& rotation) {
  if (rotation.cols != 3 || rotation.rows != 3) return "";

  stringstream ss;
  ss << "[";
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      ss << rotation.at<float>(i, j);
      if (i != 2 || j != 2) ss << ",";
    }
  }
  ss << "]";

  return ss.str();
}

template <typename T>
std::string toString(const cv::Mat& mat, int precision) {
  std::stringstream ss;
  ss.precision(precision);
  for (size_t i = 0; i < mat.rows; i++) {
    for (size_t j = 0; j < mat.cols; j++) {
      ss << " | " << std::fixed << mat.at<T>(i, j);
    }
    ss << " |\n";
  }

  return ss.str();
}

std::string toPythonString(const std::vector<cv::Point3f>& cloud) {
  stringstream ss;
  ss << "[";
  for (size_t j = 0; j < cloud.size(); j++) {
    if (cloud[j].z != 0) ss << toString(cloud[j]);
    if (cloud[j].z != 0 && j < cloud.size() - 1) ss << ",";
  }
  ss << "]";

  return ss.str();
}


} // end namesapce
