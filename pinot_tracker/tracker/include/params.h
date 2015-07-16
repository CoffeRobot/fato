#ifndef PARAMS_H
#define PARAMS_H

#include <opencv2/core/core.hpp>
#include <image_geometry/pinhole_camera_model.h>

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
  {}
};

}

#endif // PARAMS_H

