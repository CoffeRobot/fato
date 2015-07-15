#include "../include/pose_estimation.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <boost/math/constants/constants.hpp>
#include "../../utilities/include/utilities.h"


using namespace std;
using namespace cv;

namespace pinot_tracker {

void getPose2D(const std::vector<cv::Point3f>& model_points,
               const std::vector<cv::Point2f>& tracked_points,
               const cv::Mat& camera_model, int iterations, float distance,
               vector<int>& inliers, Mat& rotation, Mat& translation) {
  if (model_points.size() > 4) {
    solvePnPRansac(model_points, tracked_points, camera_model,
                   Mat::zeros(1, 8, CV_32F), rotation, translation, false,
                   iterations, distance, model_points.size(), inliers, CV_P3P);
    //    double T[] = {tvec.at<double>(0, 0), tvec.at<double>(0, 1),
    //                  tvec.at<double>(0, 2)};
    //    double R[] = {rvec.at<double>(0, 0), rvec.at<double>(0, 1),
    //                  rvec.at<double>(0, 2)};
  }
}

void getPose2D(const std::vector<cv::Point2f*>& model_points,
               const std::vector<cv::Point2f*>& tracked_points,
               float& scale,
               float& angle) {
  vector<double> angles;
  vector<float> scales;

  angles.reserve(model_points.size() * 2);
  scales.reserve(model_points.size() * 2);

  const double pi = boost::math::constants::pi<double>();

  for (size_t i = 0; i < model_points.size(); ++i) {
    for (size_t j = 0; j < model_points.size(); j++) {
      // computing angle
      Point2f a = *tracked_points.at(i) - *tracked_points.at(j);
      Point2f b = *model_points.at(i) - *model_points.at(j);
      double val = atan2(a.y, a.x) - atan2(b.y, b.x);

      if (abs(val) > pi) {
        int sign = (val < 0) ? -1 : 1;
        val = val - sign * 2 * pi;
      }
      angles.push_back(val);

      if (i == j) continue;

      // computing scale
      auto tracked_dis =
          getDistance(tracked_points.at(i), tracked_points.at(j));
      auto model_dis =
          getDistance(model_points.at(i), model_points.at(j));
      if (model_dis != 0) scales.push_back(tracked_dis / model_dis);
    }
  }

  sort(angles.begin(), angles.end());
  sort(scales.begin(), scales.end());

  auto angles_size = angles.size();
  auto scale_size = scales.size();

  if (angles_size == 0)
    angle = 0;

  else if (angles_size % 2 == 0)
    angle = (angles[angles_size / 2 - 1] + angles[angles_size / 2]) / 2;
  else
    angle = angles[angles_size / 2];

  if (scale_size == 0)
    scale = 1;
  else if (scale_size % 2 == 0)
    scale = (scales[scale_size / 2 - 1] + scales[scale_size / 2]) / 2;
  else
    scale = scales[scale_size / 2];
}

}  // end namespace
