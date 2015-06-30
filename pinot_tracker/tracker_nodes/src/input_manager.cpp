#include "../include/input_manager.h"
#include <boost/thread.hpp>
#include <mutex>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

namespace pinot_tracker {

InputManager::InputManager()
    : nh_(),
      rgb_topic_("/tracker_input/rgb"),
      depth_topic_("/tracker_input/depth"),
      camera_info_topic_("/tracker_input/camera_info"),
      queue_size(5),
      spinner_(0) {
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

  cv::namedWindow("Image Viewer");

  ros::Rate r(1);
  while (ros::ok()) {
    //ROS_INFO_STREAM("Main thread [" << boost::this_thread::get_id() << "].");

    if(img_updated_)
    {
      imshow("Image Viewer", rgb_image_);
      img_updated_ = false;
      waitKey(1);
    }

    //r.sleep();
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

void InputManager::readImage(const sensor_msgs::Image::ConstPtr msgImage,
                             cv::Mat &image) const {
  cv_bridge::CvImageConstPtr pCvImage;
  pCvImage = cv_bridge::toCvShare(msgImage, msgImage->encoding);
  pCvImage->image.copyTo(image);
}

}  // end namespace
