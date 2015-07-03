#include "../include/input_manager.h"
#include <boost/thread.hpp>
#include <mutex>
#include <opencv2/highgui/highgui.hpp>
#include <Tracker.h>
#include <DebugFunctions.h>

using namespace std;
using namespace cv;

namespace pinot_tracker {

InputManager::InputManager()
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
      mouse_end_(0, 0) {

  namedWindow("Image Viewer");
  setMouseCallback("Image Viewer", InputManager::mouseCallback, this);

  run();
}

void InputManager::start() {
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
      boost::bind(&InputManager::rgbdCallback, this, _1, _2, _3));

  spinner_.start();

  ROS_INFO("INPUT: init tracker");

  pinot::gpu::Tracker gpu_tracker;

  ROS_INFO("INPUT: init tracker 2");

  ros::Rate r(1);
  while (ros::ok()) {
    // ROS_INFO_STREAM("Main thread [" << boost::this_thread::get_id() << "].");

    if (img_updated_) {

      if(mouse_start_.x != mouse_end_.x)
        rectangle(rgb_image_, mouse_start_, mouse_end_, Scalar(255,0,0), 3);

      if(init_requested_)
      {
        ROS_INFO("INPUT: tracker intialization requested");
        gpu_tracker.init(rgb_image_, mouse_start_, mouse_end_);
        init_requested_ = false;
        tracker_initialized_ = true;
        ROS_INFO("Tracker initialized!");
      }

      if(tracker_initialized_)
      {
        gpu_tracker.computeNext(rgb_image_);
        Point2f p = gpu_tracker.getCentroid();
        circle(rgb_image_, p, 5, Scalar(255,0,0), 2);
        Scalar color(255,0,0);
        drawBoundingBox(gpu_tracker.getBoundingBox(),color, rgb_image_);
      }

      imshow("Image Viewer", rgb_image_);
      img_updated_ = false;
      waitKey(1);
    }

    // r.sleep();
  }
}

void InputManager::run() {
  start();
  stop();
}

void InputManager::stop() {
  ROS_INFO("Shutting down node ...");
  spinner_.stop();
}

void InputManager::rgbdCallback(
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

void InputManager::mouseCallback(int event, int x, int y) {

  auto set_point = [this](int x, int y)
  {
    if(x < mouse_start_.x)
    {
      mouse_end_.x = mouse_start_.x;
      mouse_start_.x = x;
    }
    else
      mouse_end_.x = x;

    if(y < mouse_start_.y)
    {
      mouse_end_.y = mouse_start_.y;
      mouse_start_.y = y;
    }
    else
      mouse_end_.y = y;
  };


  if (event == EVENT_LBUTTONDOWN) {
    mouse_start_.x = x;
    mouse_start_.y = y;
    mouse_end_ = mouse_start_;
    is_mouse_dragging_ = true;
  } else if (event == EVENT_MOUSEMOVE && is_mouse_dragging_) {
    set_point(x,y);
  } else if (event == EVENT_LBUTTONUP) {
    set_point(x,y);
    is_mouse_dragging_ = false;
    init_requested_ = true;
  }
}

void InputManager::mouseCallback(int event, int x, int y, int flags,
                                 void *userdata) {
  auto manager = reinterpret_cast<InputManager *>(userdata);
  manager->mouseCallback(event, x, y);
}

void InputManager::readImage(const sensor_msgs::Image::ConstPtr msgImage,
                             cv::Mat &image) const {
  cv_bridge::CvImageConstPtr pCvImage;
  pCvImage = cv_bridge::toCvShare(msgImage, msgImage->encoding);
  pCvImage->image.copyTo(image);
}

}  // end namespace
