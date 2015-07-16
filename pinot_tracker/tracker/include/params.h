#ifndef PARAMS_H
#define PARAMS_H

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

  TrackerParams()
      : eps(10),
        min_points(5),
        threshold(30),
        octaves(3),
        pattern_scale(1.0f),
        ransac_iterations(100),
        ransac_distance(8.0f)
  {}

  TrackerParams(const TrackerParams& other)
    : eps(other.eps),
      min_points(other.min_points),
      threshold(other.threshold),
      octaves(other.octaves),
      pattern_scale(other.pattern_scale),
      ransac_distance(other.ransac_distance),
      ransac_iterations(other.ransac_iterations)
  {}
};

}

#endif // PARAMS_H

