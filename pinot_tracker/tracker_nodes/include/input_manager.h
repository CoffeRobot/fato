#ifndef INPUTMANAGER_H
#define INPUTMANAGER_H

#include <sensor_msgs/CameraInfo.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <image_geometry/pinhole_camera_model.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <string>

namespace pinot_tracker {

class InputManager {
 public:
  InputManager();

 protected:
  void rgbdCallback(const sensor_msgs::ImageConstPtr& depth_msg,
                    const sensor_msgs::ImageConstPtr& rgb_msg,
                    const sensor_msgs::CameraInfoConstPtr& camera_info_msg);

  static void mouseCallback(int event, int x, int y, int flags, void* userdata);

  void mouseCallback(int event, int x, int y);

 private:
  void start();

  void stop();

  void run();

  void showInput();

  void readImage(const sensor_msgs::Image::ConstPtr msgImage,
                 cv::Mat& image) const;

  ros::NodeHandle nh_;
  // message filter
  boost::shared_ptr<image_transport::ImageTransport> rgb_it_, depth_it_;
  image_transport::SubscriberFilter sub_depth_, sub_rgb_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> sub_camera_info_;
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo>
      SyncPolicyRGBD;
  typedef message_filters::Synchronizer<SyncPolicyRGBD> SynchronizerRGBD;
  boost::shared_ptr<SynchronizerRGBD> sync_rgbd_;

  const std::string rgb_topic_;
  const std::string depth_topic_;
  const std::string camera_info_topic_;
  int queue_size;

  ros::AsyncSpinner spinner_;

  cv::Mat rgb_image_, depth_image_;
  bool img_updated_;

  cv::Point2d mouse_start_, mouse_end_;
  bool is_mouse_dragging_, init_requested_, tracker_initialized_;
};

}  // end namespace

#endif  // INPUTMANAGER_H
