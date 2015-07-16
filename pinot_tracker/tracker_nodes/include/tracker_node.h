#ifndef TRACKER_NODE_H
#define TRACKER_NODE_H

#include <sensor_msgs/CameraInfo.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <sensor_msgs/CameraInfo.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <string>

namespace pinot_tracker {

class TrackerNode {
 public:
  TrackerNode();

 protected:
  virtual void rgbCallback(
      const sensor_msgs::ImageConstPtr& rgb_msg,
      const sensor_msgs::CameraInfoConstPtr& camera_info_msg) = 0;

  virtual void rgbdCallback(
      const sensor_msgs::ImageConstPtr& depth_msg,
      const sensor_msgs::ImageConstPtr& rgb_msg,
      const sensor_msgs::CameraInfoConstPtr& camera_info_msg) = 0;

  void readImage(const sensor_msgs::Image::ConstPtr msgImage,
                 cv::Mat& image) const;

  void getCameraMatrix(const sensor_msgs::CameraInfo::ConstPtr info,
                       cv::Mat& camera_matrix);
};

}  // end namespace

#endif  // TRACKER_NODE_H
