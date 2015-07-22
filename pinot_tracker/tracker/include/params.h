#ifndef PARAMS_H
#define PARAMS_H

#include <image_geometry/pinhole_camera_model.h>
#include <string>
#include <sstream>
#include <ros/ros.h>
#include <opencv2/core/core.hpp>


namespace pinot_tracker
{

struct TrackerParams {
  float eps;
  float min_points;
  int threshold;
  int octaves;
  float pattern_scale;
  bool filter_border;
  bool update_votes;
  int ransac_iterations;
  float ransac_distance;
  image_geometry::PinholeCameraModel camera_model;
  cv::Mat camera_matrix;
  std::string debug_path;


  TrackerParams()
      : eps(10),
        min_points(5),
        threshold(30),
        octaves(3),
        pattern_scale(1.0f),
        ransac_iterations(100),
        ransac_distance(8.0f),
        camera_model()
  {}

  TrackerParams(const TrackerParams& other)
    : eps(other.eps),
      min_points(other.min_points),
      threshold(other.threshold),
      octaves(other.octaves),
      pattern_scale(other.pattern_scale),
      ransac_distance(other.ransac_distance),
      ransac_iterations(other.ransac_iterations),
      camera_model(other.camera_model)
  {
      camera_matrix = other.camera_matrix.clone();
  }


  void readRosConfigFile()
  {
    std::stringstream ss;

    ss << "Reading config parameters... \n";

    ss << "filter_border: ";
    if (!ros::param::get("pinot/tracker_2d/filter_border", filter_border))
    {
        filter_border = false;
        ss << "failed \n";
    }
    else
      ss << filter_border << "\n";

    ss << "update_votes: ";
    if (!ros::param::get("pinot/tracker_2d/update_votes", update_votes))
    {
        ss << "failed \n";
        update_votes = false;
    }
    else
      ss << update_votes << "\n";

    ss << "eps: ";
    if (!ros::param::get("pinot/clustering/eps", eps))
    {
      ss << "failed \n";
      eps = 5;
    }
    else
      ss << eps << "\n";

    ss << "min_points: ";
    if (!ros::param::get("pinot/clustering/min_points", min_points))
    {
      ss << "failed \n";
      min_points = 5;
    }
    else
      ss << min_points << "\n";

    ss << "ransac_iterations: ";
    if (!ros::param::get("pinot/pose_estimation/ransac_iterations",
                         ransac_iterations))
    {
      ss << "failed \n";
      ransac_iterations = 100;
    }
    else
      ss << ransac_iterations << "\n";

    ss << "ransac_distance: ";
    if (!ros::param::get("pinot/pose_estimation/ransac_distance",
                         ransac_distance))
    {
      ss << "failed \n";
      ransac_distance = 8.0f;
    }
    else
      ss << ransac_distance << "\n";

    ROS_INFO(ss.str().c_str());
  }
};

}

#endif // PARAMS_H

