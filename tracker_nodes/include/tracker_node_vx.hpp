/*****************************************************************************/
/*  Copyright (c) 2016, Alessandro Pieropan                                  */
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
#ifndef TRACKER_MODEL_VX_H
#define TRACKER_MODEL_VX_H

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
#include <camera_info_manager/camera_info_manager.h>

#include <memory>

#include "tracker_node.h"
#include "../../tracker/include/tracker_model_vx.h"
#include "fato_tracker_nodes/TrackerService.h"

namespace fato{

class TrackerModelVX : public TrackerNode {
 public:
  TrackerModelVX();

 protected:

  void loadParameters(ros::NodeHandle& nh);

  void rgbCallback(const sensor_msgs::ImageConstPtr& rgb_msg,
                   const sensor_msgs::CameraInfoConstPtr& camera_info_msg);

  void rgbCallbackNoInfo(const sensor_msgs::ImageConstPtr& rgb_msg);

  void rgbdCallback(
          const sensor_msgs::ImageConstPtr &depth_msg,
          const sensor_msgs::ImageConstPtr &rgb_msg,
          const sensor_msgs::CameraInfoConstPtr &camera_info_msg);

  bool serviceCallback(fato_tracker_nodes::TrackerService::Request &req,
                  fato_tracker_nodes::TrackerService::Response &res);

 private:
  void run();

  void initRGBSynch();

  void initRGB();

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

  //TrackerParams params_;

  ros::AsyncSpinner spinner_;
  ros::Publisher publisher_, render_publisher_, flow_publisher_;
  ros::ServiceServer service_server_;

  std::string obj_file_;

  std::unique_ptr<fato::TrackerVX> vx_tracker_;
  TrackerVX::Params params_;

  bool stop_matcher;

  camera_info_manager::CameraInfoManager cam_info_manager_;

};

} // end namespace

#endif // TRACKER_MODEL_BASED_H
