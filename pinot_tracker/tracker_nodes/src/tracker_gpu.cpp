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


#include "../include/tracker_gpu.h"
#include <boost/thread.hpp>
#include <mutex>
#include <opencv2/highgui/highgui.hpp>
#include <Tracker.h>
#include <DebugFunctions.h>
#include <chrono>

using namespace std;
using namespace cv;

namespace pinot_tracker {

TrackerGpu2D::TrackerGpu2D()
    : nh_(),
      rgb_topic_("/tracker_input/rgb"),
      depth_topic_("/tracker_input/depth"),
      camera_info_topic_("/tracker_input/camera_info"),
      queue_size(5),
      spinner_(0),
      is_mouse_dragging_(false),
      img_updated_(false),
      init_requested_(false),
      tracker_initialized_(false),
      mouse_start_(0, 0),
      mouse_end_(0, 0),
      use_depth_(false)
{

  namedWindow("Image Viewer");
  setMouseCallback("Image Viewer", TrackerGpu2D::mouseCallback, this);

  getTrackerParameters();

  if(use_depth_)
    initRGBD();
  else
    initRGB();

  run();
}

void TrackerGpu2D::start() {

  spinner_.start();

  ROS_INFO("INPUT: init tracker");

  pinot::gpu::Tracker gpu_tracker;

  ros::Rate r(1);
  while (ros::ok()) {
    // ROS_INFO_STREAM("Main thread [" << boost::this_thread::get_id() << "].");

    if (img_updated_) {

      if (mouse_start_.x != mouse_end_.x)
        rectangle(rgb_image_, mouse_start_, mouse_end_, Scalar(255, 0, 0), 3);

      if (init_requested_) {
        ROS_INFO("INPUT: tracker intialization requested");
        gpu_tracker.init(rgb_image_, mouse_start_, mouse_end_);
        init_requested_ = false;
        tracker_initialized_ = true;
        ROS_INFO("Tracker initialized!");
      }

      if (tracker_initialized_) {
        auto begin = chrono::high_resolution_clock::now();
        gpu_tracker.computeNext(rgb_image_);
        auto end = chrono::high_resolution_clock::now();
        auto time_span =
            chrono::duration_cast<chrono::milliseconds>(end - begin).count();
        stringstream ss;
        ss << "Tracker run in ms: " << time_span << "";
        ROS_INFO(ss.str().c_str());

        Point2f p = gpu_tracker.getCentroid();
        circle(rgb_image_, p, 5, Scalar(255, 0, 0), -1);
        Scalar color(255, 0, 0);
        drawBoundingBox(gpu_tracker.getBoundingBox(), color, rgb_image_);
        
      }

      imshow("Image Viewer", rgb_image_);
      img_updated_ = false;
      waitKey(1);
    }

    // r.sleep();
  }
}

void TrackerGpu2D::run() {
  start();
  stop();
}

void TrackerGpu2D::stop() {
  ROS_INFO("Shutting down node ...");
  spinner_.stop();
}

void TrackerGpu2D::rgbdCallback(
    const sensor_msgs::ImageConstPtr &depth_msg,
    const sensor_msgs::ImageConstPtr &rgb_msg,
    const sensor_msgs::CameraInfoConstPtr &camera_info_msg) {

  cv::Mat rgb, depth;

  readImage(rgb_msg, rgb);
  readImage(depth_msg, depth);

  rgb_image_ = rgb;
  depth_image_ = depth;
  img_updated_ = true;
}

void TrackerGpu2D::rgbCallback(
    const sensor_msgs::ImageConstPtr &rgb_msg,
    const sensor_msgs::CameraInfoConstPtr &camera_info_msg) {

  cv::Mat rgb;

  readImage(rgb_msg, rgb);

  rgb_image_ = rgb;
  img_updated_ = true;
}

void TrackerGpu2D::mouseCallback(int event, int x, int y) {

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

void TrackerGpu2D::mouseCallback(int event, int x, int y, int flags,
                                 void *userdata) {
  auto manager = reinterpret_cast<TrackerGpu2D *>(userdata);
  manager->mouseCallback(event, x, y);
}

void TrackerGpu2D::readImage(const sensor_msgs::Image::ConstPtr msgImage,
                             cv::Mat &image) const {
  cv_bridge::CvImageConstPtr pCvImage;
  pCvImage = cv_bridge::toCvShare(msgImage, msgImage->encoding);
  pCvImage->image.copyTo(image);
}

void TrackerGpu2D::getTrackerParameters() {
  int n_feat, levels, edge_thres, patch;
  float scale, confidence, ratio;
  stringstream ss;

  ss << "Tracker Input: \n";
  if(!ros::param::get("pinot/gpu_tracker/use_depth", use_depth_))
  {
    ss << "Failed to parse camera parameter\n";
    use_depth_ = false;
  }

  if(use_depth_)
    ss << "RGBD camera \n";
  else
    ss << "RGB camera \n";

  ss << "Feature Extractor: \n";
  ss << "num_features: ";
  if (!ros::param::get("pinot/gpu_tracker/num_features", n_feat))
    ss << "failed \n";
  else
    ss << n_feat << "\n";

  ss << "scale_factor: ";
  if (!ros::param::get("pinot/gpu_tracker/scale_factor", scale))
    ss << "failed \n";
  else
    ss << scale << "\n";

  ss << "num_levels: ";
  if (!ros::param::get("pinot/gpu_tracker/num_levels", levels))
    ss << "failed \n";
  else
    ss << levels << "\n";

  ss << "edge: ";
  if (!ros::param::get("pinot/gpu_tracker/edge_threshold", edge_thres))
    ss << "failed \n";
  else
    ss << edge_thres << "\n";

  ss << "patch: ";
  if (!ros::param::get("pinot/gpu_tracker/patch_size", patch))
    ss << "failed \n";
  else
    ss << patch << "\n";

  ss << "Matcher: \n";
  ss << "confidence: ";
  if (!ros::param::get("pinot/gpu_tracker/feature_confidence", confidence))
    ss << "failed \n";
  else
    ss << confidence << "\n";

  ss << "ratio: ";
  if (!ros::param::get("pinot/gpu_tracker/second_ratio", ratio))
    ss << "failed \n";
  else
    ss << ratio << "\n";

  ROS_INFO(ss.str().c_str());
}

void TrackerGpu2D::initRGB() {
  rgb_it_.reset(new image_transport::ImageTransport(nh_));
  sub_camera_info_.subscribe(nh_, camera_info_topic_, 1);
  /** kinect node settings */
  sub_rgb_.subscribe(*rgb_it_, rgb_topic_, 1,
                     image_transport::TransportHints("compressed"));

  sync_rgb_.reset(new SynchronizerRGB(SyncPolicyRGB(queue_size), sub_rgb_,
                                      sub_camera_info_));
  sync_rgb_->registerCallback(
      boost::bind(&TrackerGpu2D::rgbCallback, this, _1, _2));
}

void TrackerGpu2D::initRGBD() {
  depth_it_.reset(new image_transport::ImageTransport(nh_));
  rgb_it_.reset(new image_transport::ImageTransport(nh_));
  sub_camera_info_.subscribe(nh_, camera_info_topic_, 1);
  /** kinect node settings */
  sub_depth_.subscribe(*depth_it_, depth_topic_, 1,
                       image_transport::TransportHints("compressed"));
  sub_rgb_.subscribe(*rgb_it_, rgb_topic_, 1,
                     image_transport::TransportHints("compressed"));

  sync_rgbd_.reset(new SynchronizerRGBD(SyncPolicyRGBD(queue_size), sub_depth_,
                                        sub_rgb_, sub_camera_info_));
  sync_rgbd_->registerCallback(
      boost::bind(&TrackerGpu2D::rgbdCallback, this, _1, _2, _3));
}

}  // end namespace

int main(int argc, char *argv[]) {

  ROS_INFO("Starting tracker input");
  ros::init(argc, argv, "pinot_tracker_node");

  pinot_tracker::TrackerGpu2D manager;

  ros::shutdown();

  return 0;
}
