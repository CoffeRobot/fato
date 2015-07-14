#ifndef TRACKER_NODE_2D_H
#define TRACKER_NODE_2D_H

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
#include <tracker_2d.h>

#include "tracker_node.h"


namespace pinot_tracker{

class TrackerNode2D : public TrackerNode{
 public:
  TrackerNode2D();

 protected:
  void rgbCallback(const sensor_msgs::ImageConstPtr& rgb_msg,
                   const sensor_msgs::CameraInfoConstPtr& camera_info_msg);

  void rgbdCallback(
          const sensor_msgs::ImageConstPtr &depth_msg,
          const sensor_msgs::ImageConstPtr &rgb_msg,
          const sensor_msgs::CameraInfoConstPtr &camera_info_msg);

  static void mouseCallback(int event, int x, int y, int flags, void* userdata);

  void mouseCallback(int event, int x, int y);

 private:
  void run();

  void initRGB();

  void getTrackerParameters();

  ros::NodeHandle nh_;
  // message filter
  boost::shared_ptr<image_transport::ImageTransport> rgb_it_;
  image_transport::SubscriberFilter sub_rgb_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> sub_camera_info_;

  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image, sensor_msgs::CameraInfo> SyncPolicyRGB;

  typedef message_filters::Synchronizer<SyncPolicyRGB> SynchronizerRGB;
  boost::shared_ptr<SynchronizerRGB> sync_rgb_;

  int queue_size;

  const std::string rgb_topic_;
  const std::string depth_topic_;
  const std::string camera_info_topic_;

  cv::Mat rgb_image_;
  cv::Mat camera_matrix_;
  bool img_updated_;
  bool camera_matrix_initialized;

  cv::Point2d mouse_start_, mouse_end_;
  bool is_mouse_dragging_, init_requested_, tracker_initialized_;

  TrackerParams params_;

  ros::AsyncSpinner spinner_;
  ros::Publisher publisher_;
};

} // end namespace

#endif  // TRACKER_NODE_2D_H
