#ifndef PROJECTION_H
#define PROJECTION_H

#ifndef TRACKERNODE3D_H
#define TRACKERNODE3D_H

#include <sensor_msgs/CameraInfo.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <image_geometry/pinhole_camera_model.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/image_encodings.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <string>
#include "../../tracker/include/tracker_3d.h"

namespace pinot_tracker {

class Projection {
 public:
  Projection();

 protected:
  void rgbCallback(const sensor_msgs::ImageConstPtr& rgb_msg,
                   const sensor_msgs::CameraInfoConstPtr& camera_info_msg);

  void rgbdCallback(const sensor_msgs::ImageConstPtr& depth_msg,
                    const sensor_msgs::ImageConstPtr& rgb_msg,
                    const sensor_msgs::CameraInfoConstPtr& camera_info_msg);

  static void mouseCallback(int event, int x, int y, int flags, void* userdata);

  void mouseCallback(int event, int x, int y);

  void readImage(const sensor_msgs::Image::ConstPtr msgImage,
                 cv::Mat& image) const;

 private:
  void run();

  void initRGBD();

  void getTrackerParameters();

  void analyzeCube(const cv::Mat& disparity, const cv::Point2d& top_left,
                   const cv::Point2d& bottom_right, cv::Point3f& median_p,
                   cv::Point3f& min_p, cv::Point3f& max_p);

  void publishPose(cv::Point3f& centroid, std::vector<cv::Point3f>& back_points,
                   std::vector<cv::Point3f>& front_points);

  void publishPose(cv::Point3f& mean_point, cv::Point3f& min_point,
                   cv::Point3f& max_point);

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

  tf::TransformBroadcaster transform_broadcaster_;

  int queue_size;

  const std::string rgb_topic_;
  const std::string depth_topic_;
  const std::string camera_info_topic_;

  cv::Mat rgb_image_, depth_image_;
  cv::Mat camera_matrix_;
  bool img_updated_;
  bool camera_matrix_initialized_;

  cv::Point2d mouse_start_, mouse_end_;
  bool is_mouse_dragging_, init_requested_, tracker_initialized_;

  TrackerParams params_;

  ros::AsyncSpinner spinner_;
  ros::Publisher publisher_, markers_publisher_;
};

}  // end namespace

#endif  // TRACKERNODE3D_H

#endif  // PROJECTION_H
