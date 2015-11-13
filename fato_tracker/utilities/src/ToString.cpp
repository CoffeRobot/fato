#include "../include/ToString.h"

#include <iomanip>

using namespace std;
using namespace cv;
using namespace Eigen;

namespace pinot_tracker{

string toString(const Point2f& p)
{
	stringstream ss;
	ss << fixed << setprecision(2) <<"[" << p.x << "," << p.y << "] ";
	return ss.str();
}


string toString(const Status& s)
{
	switch (s)
	{

	case Status::BACKGROUND:
		return "BACKGROUND";
	case Status::INIT:
		return "INIT";
	case Status::MATCH:
		return "MATCH";
	case Status::NOMATCH:
		return "NOMATCH";
	case Status::NOCLUSTER:
		return "NOCLUSTER";
	case Status::TRACK:
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
