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

#ifndef PARAMS_H
#define PARAMS_H

#include <image_geometry/pinhole_camera_model.h>
#include <string>
#include <sstream>
#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>


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
  bool use_ransac;
  int ransac_iterations;
  float ransac_distance;
  int ransac_method;
  image_geometry::PinholeCameraModel camera_model;
  cv::Mat camera_matrix;
  std::string debug_path;


  TrackerParams()
      : eps(10),
        min_points(5),
        threshold(30),
        octaves(3),
        pattern_scale(1.0f),
        use_ransac(false),
        ransac_iterations(100),
        ransac_distance(2.0f),
        ransac_method(CV_P3P),
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

    ss << "use_ransac: ";
    if (!ros::param::get("pinot/pose_estimation/use_ransac",
                         use_ransac))
    {
      ss << "failed \n";
      use_ransac = false;
    }
    else
      ss << use_ransac << "\n";

    ss << "ransac_method: ";
    int method;
    if (!ros::param::get("pinot/pose_estimation/ransac_method",
                         method))
    {
      ss << "failed \n";
      ransac_method = CV_P3P;
    }
    else
    {
        if(method == 1) ransac_method = CV_ITERATIVE;
        else if(method == 3) ransac_method = CV_EPNP;
        else ransac_method = CV_P3P;
        ss << ransac_method << "\n";
    }

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
      ransac_distance = 2.0f;
    }
    else
      ss << ransac_distance << "\n";

    ROS_INFO(ss.str().c_str());
  }
};

}

#endif // PARAMS_H

