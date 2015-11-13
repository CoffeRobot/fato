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
#include "../../tracker/include/tracker_2d_v2.h"
#include "../../tracker/include/tracker_2d.h"

using namespace cv;
using namespace std;

namespace pinot_tracker {

TrackerNode2D::TrackerNode2D()
    : nh_(),
      rgb_topic_("/tracker_input/rgb"),
      camera_info_topic_("/tracker_input/camera_info"),
      queue_size(5),
      is_mouse_dragging_(false),
      img_updated_(false),
      init_requested_(false),
      tracker_initialized_(false),
      mouse_start_(0, 0),
      mouse_end_(0, 0),
      spinner_(0),
      params_(),
      camera_matrix_initialized(false) {
  cvStartWindowThread();
  namedWindow("Image Viewer");
  setMouseCallback("Image Viewer", TrackerNode2D::mouseCallback, this);

  publisher_ = nh_.advertise<sensor_msgs::Image>("pinot_tracker/output", 1);

  getTrackerParameters();

  initRGB();

  run();
}

void TrackerNode2D::initRGB() {
  rgb_it_.reset(new image_transport::ImageTransport(nh_));
  sub_camera_info_.subscribe(nh_, camera_info_topic_, 1);
  /** kinect node settings */
  sub_rgb_.subscribe(*rgb_it_, rgb_topic_, 1,
                     image_transport::TransportHints("raw"));

  sync_rgb_.reset(new SynchronizerRGB(SyncPolicyRGB(queue_size), sub_rgb_,
                                      sub_camera_info_));
  sync_rgb_->registerCallback(
      boost::bind(&TrackerNode2D::rgbCallback, this, _1, _2));
}

void TrackerNode2D::rgbCallback(
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

void TrackerNode2D::getTrackerParameters() {
  stringstream ss;

  ss << "Tracker Input: \n";

  ss << "filter_border: ";
  if (!ros::param::get("pinot/tracker_2d/filter_border", params_.filter_border))
    ss << "failed \n";
  else
    ss << params_.filter_border << "\n";

  ss << "update_votes: ";
  if (!ros::param::get("pinot/tracker_2d/update_votes", params_.update_votes))
    ss << "failed \n";
  else
    ss << params_.update_votes << "\n";

  ss << "eps: ";
  if (!ros::param::get("pinot/tracker_2d/eps", params_.eps))
    ss << "failed \n";
  else
    ss << params_.eps << "\n";

  ss << "min_points: ";
  if (!ros::param::get("pinot/tracker_2d/min_points", params_.min_points))
    ss << "failed \n";
  else
    ss << params_.min_points << "\n";

  ss << "ransac_iterations: ";
  if (!ros::param::get("pinot/tracker_2d/ransac_iterations",
                       params_.ransac_iterations))
    ss << "failed \n";
  else
    ss << params_.ransac_iterations << "\n";

  ss << "ransac_distance: ";
  if (!ros::param::get("pinot/tracker_2d/ransac_distance",
                       params_.ransac_distance))
    ss << "failed \n";
  else
    ss << params_.ransac_distance << "\n";

  ROS_INFO(ss.str().c_str());
}

void TrackerNode2D::rgbdCallback(
    const sensor_msgs::ImageConstPtr &depth_msg,
    const sensor_msgs::ImageConstPtr &rgb_msg,
    const sensor_msgs::CameraInfoConstPtr &camera_info_msg) {
  ROS_INFO("Tracker2D: rgbd usupported");
}

void TrackerNode2D::mouseCallback(int event, int x, int y) {
  auto set_point = [this](int x, int y) {
    if (x < mouse_start_.x) {
      mouse_end_.x = mouse_start_.x;
      mouse_start_.x = x;
    } else
      mouse_end_.x = x;

    if (y < mouse_start_.y) {
      mouse_end_.y = mouse_start_.y;
      mouse_start_.y = y;
    } else
      mouse_end_.y = y;
  };

  if (event == EVENT_LBUTTONDOWN) {
    mouse_start_.x = x;
    mouse_start_.y = y;
    mouse_end_ = mouse_start_;
    is_mouse_dragging_ = true;
  } else if (event == EVENT_MOUSEMOVE && is_mouse_dragging_) {
    set_point(x, y);
  } else if (event == EVENT_LBUTTONUP) {
    set_point(x, y);
    is_mouse_dragging_ = false;
    init_requested_ = true;
  }
}

void TrackerNode2D::mouseCallback(int event, int x, int y, int flags,
                                  void *userdata) {
  auto manager = reinterpret_cast<TrackerNode2D *>(userdata);
  manager->mouseCallback(event, x, y);
}

void TrackerNode2D::run() {
  ROS_INFO("INPUT: init tracker");

  spinner_.start();

  cout << params_.threshold << " " << params_.octaves << " "
       << params_.pattern_scale << endl;

  // Tracker2D tracker(params_);
  TrackerV2 tracker(params_, camera_matrix_);

  auto &profiler = Profiler::getInstance();

  ros::Rate r(100);
  while (ros::ok()) {
    // ROS_INFO_STREAM("Main thread [" << boost::this_thread::get_id() << "].");

    if (img_updated_) {
      if (mouse_start_.x != mouse_end_.x && !tracker_initialized_) {
        rectangle(rgb_image_, mouse_start_, mouse_end_, Scalar(255, 0, 0), 3);
        img_updated_ = false;
      }
      if (!tracker_initialized_) {
        imshow("Image Viewer", rgb_image_);
        waitKey(1);
      }
      if (init_requested_) {
        ROS_INFO("INPUT: tracker intialization requested");
        tracker.init(rgb_image_, mouse_start_, mouse_end_);
        init_requested_ = false;
        tracker_initialized_ = true;
        ROS_INFO("Tracker initialized!");
        destroyWindow("Image Viewer");
        waitKey(1);
      }

      if (tracker_initialized_) {
        //        auto begin = chrono::high_resolution_clock::now();
        profiler->start("total");
        Mat out;
        tracker.computeNext(rgb_image_);
        profiler->stop("total");
        //        auto end = chrono::high_resolution_clock::now();
        //        auto time_span =
        //            chrono::duration_cast<chrono::milliseconds>(end -
        //            begin).count();
        stringstream ss;
        Point2f p = tracker.getCentroid();
        circle(rgb_image_, p, 5, Scalar(255, 0, 0), -1);
        vector<Point2f> bbox = tracker.getBoundingBox();
        drawBoundingBox(bbox, Scalar(255, 0, 0), 2, rgb_image_);
        //        ss << "Tracker run in ms: " << time_span << "";
        ss << "Profiler " << profiler->getProfile().c_str();
        ROS_INFO(ss.str().c_str());

        cv_bridge::CvImage cv_img;
        cv_img.image = rgb_image_;
        cv_img.encoding = sensor_msgs::image_encodings::BGR8;
        publisher_.publish(cv_img.toImageMsg());
        r.sleep();
      }
    }
  }
}

}  // end namespace

int main(int argc, char *argv[]) {
  ROS_INFO("Starting tracker input");
  ros::init(argc, argv, "pinot_tracker_node");

  pinot_tracker::TrackerNode2D manager;

  ros::shutdown();

  return 0;
}
