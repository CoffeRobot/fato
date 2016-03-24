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

#include "../include/tracker_node_2d.h"
#include <boost/thread.hpp>
#include <mutex>
#include <opencv2/highgui/highgui.hpp>
#include <sstream>
#include <chrono>
#include <profiler.h>
#include <draw_functions.h>
#include <opencv2/calib3d/calib3d.hpp>

#include "../../utilities/include/hdf5_file.h"
#include "../include/tracker_model_based.h"
#include "../../utilities/include/utilities.h"
#include "../../tracker/include/tracker_model.h"
#include "../../tracker/include/pose_estimation.h"
#include "../../utilities/include/draw_functions.h"

using namespace cv;
using namespace std;

namespace fato {

TrackerModel::TrackerModel(string model_file)
    : nh_(),
      rgb_topic_("/image_raw"),
      camera_info_topic_("/camera_info"),
      queue_size(5),
      is_mouse_dragging_(false),
      img_updated_(false),
      init_requested_(false),
      tracker_initialized_(false),
      mouse_start_(0, 0),
      mouse_end_(0, 0),
      spinner_(0),
      camera_matrix_initialized(false) {
  cvStartWindowThread();
  namedWindow("Image Viewer");

  publisher_ = nh_.advertise<sensor_msgs::Image>("fato_tracker/output", 1);

  getTrackerParameters();

  initRGB();

  run(model_file);
}

void TrackerModel::initRGB() {
  rgb_it_.reset(new image_transport::ImageTransport(nh_));
  sub_camera_info_.subscribe(nh_, camera_info_topic_, 1);
  /** kinect node settings */
  sub_rgb_.subscribe(*rgb_it_, rgb_topic_, 1,
                     image_transport::TransportHints("raw"));

  sync_rgb_.reset(new SynchronizerRGB(SyncPolicyRGB(queue_size), sub_rgb_,
                                      sub_camera_info_));

  sync_rgb_->registerCallback(
      boost::bind(&TrackerModel::rgbCallback, this, _1, _2));
}

void TrackerModel::rgbCallback(
    const sensor_msgs::ImageConstPtr &rgb_msg,
    const sensor_msgs::CameraInfoConstPtr &camera_info_msg) {
  if (!camera_matrix_initialized) {
    camera_matrix_ =
        cv::Mat(3, 4, CV_64F, (void *)camera_info_msg->P.data()).clone();
    camera_matrix_initialized = true;
  }

  Mat rgb;
  readImage(rgb_msg, rgb);
  cvtColor(rgb, rgb_image_, CV_RGB2BGR);
  img_updated_ = true;
}

void TrackerModel::getTrackerParameters() {
  stringstream ss;

  ss << "Tracker Input: \n";

  ROS_INFO(ss.str().c_str());
}

void TrackerModel::rgbdCallback(
    const sensor_msgs::ImageConstPtr &depth_msg,
    const sensor_msgs::ImageConstPtr &rgb_msg,
    const sensor_msgs::CameraInfoConstPtr &camera_info_msg) {
  ROS_INFO("Tracker2D: rgbd usupported");
}

void TrackerModel::run(string model_file) {
  ROS_INFO("TrackerMB: init...");

  spinner_.start();

  Config params;

  std::unique_ptr<FeatureMatcher> derived =
      std::unique_ptr<BriskMatcher>(new BriskMatcher);

  util::HDF5File in_file(model_file);

  //  std::vector<uchar> model_descriptors;
  //  std::vector<int> descriptors_size;
  //  //int test_feature_size;
  //  in_file.readArray<uchar>("descriptors", model_descriptors,
  //  descriptors_size);
  //  cv::Mat mat_descriptors;
  //  vectorToMat(model_descriptors, descriptors_size, mat_descriptors);
  //  derived->setTarget(mat_descriptors);

  TrackerMB tracker(params, BRISK, std::move(derived));
  ROS_INFO("TrackerMB: setting model...");
  tracker.addModel(model_file);
  ROS_INFO("TrackerMB: starting...");
  ros::Rate r(100);
  while (ros::ok()) {
    if (img_updated_) {
      cv_bridge::CvImage cv_img;

      //      std::vector<cv::KeyPoint> query_keypoints;
      //      cv::Mat query_descriptors;
      //      std::vector<std::vector<cv::DMatch>> matches;

      //      derived->match(rgb_image_,query_keypoints, query_descriptors,
      //      matches);

      // cout << "Frame" << endl;
      tracker.computeNextSequential(rgb_image_);

      char c = waitKey(5);
      if (c == 'b') {
        std::cout << "Learning background" << std::endl;
        tracker.learnBackground(rgb_image_);
      }

      vector<Point2f> valid_points;
      vector<Point3f> model_points;

      tracker.getActiveModelPoints(model_points, valid_points);

      // getPoseRansac(model_points, valid_points, camera_matrix_, )

      Mat rotation, translation;
      vector<int> inliers;

      //        for(auto pt : model_points)
      //        {
      //            cout << pt << endl;
      //        }

      // cout << camera_matrix_ << endl;

      Mat cam(3, 3, CV_64FC1);
      for (int i = 0; i < cam.rows; ++i) {
        for (int j = 0; j < cam.cols; ++j) {
          cam.at<double>(i, j) = camera_matrix_.at<double>(i, j);
        }
      }
      // cout << cam << endl;
      fato::getPoseRansac(model_points, valid_points, cam, 100, 5.0, inliers,
                          rotation, translation);
      // cout << "frame " << endl;

      vector<bool> is_in(valid_points.size(), false);
      for (int i : inliers) {
        is_in[i] = true;
      }

      cv::Point3f center(0,0,0);
      for (auto i = 0; i < valid_points.size(); ++i) {
        if (is_in.at(i))
        {
          circle(rgb_image_, valid_points.at(i), 3, cv::Scalar(255, 0, 0));
          center += model_points.at(i);
        }
        else
          circle(rgb_image_, valid_points.at(i), 3, cv::Scalar(0, 0, 255));
      }
      center.x = center.x / static_cast<float>(inliers.size());
      center.y = center.y / static_cast<float>(inliers.size());
      center.z = center.z / static_cast<float>(inliers.size());

      drawObjectPose(center, cam, rotation, translation, rgb_image_);

      //      for (auto pt : valid_points)
      //        circle(rgb_image_, pt, 3, cv::Scalar(255, 0, 0));

      //      for(int i = 0; i < matches.size(); ++i)
      //      {
      //        cv::DMatch& d = matches.at(i)[0];
      //        circle(rgb_image_, query_keypoints.at(d.trainIdx).pt, 3,
      //        cv::Scalar(255,0,0));
      //      }

      cv_img.image = rgb_image_;
      cv_img.encoding = sensor_msgs::image_encodings::BGR8;
      publisher_.publish(cv_img.toImageMsg());
      r.sleep();
    }
  }

  cv::destroyAllWindows();
}

}  // end namespace

int main(int argc, char *argv[]) {
  ROS_INFO("Starting tracker input");
  ros::init(argc, argv, "fato_tracker_node");

  if (argc < 2) throw std::runtime_error("Usage: ./track_model model.h5");

  string model_name(argv[1]);

  fato::TrackerModel manager(model_name);

  ros::shutdown();

  return 0;
}
