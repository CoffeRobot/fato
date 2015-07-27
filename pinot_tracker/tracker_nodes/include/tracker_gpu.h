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

class TrackerGpu2D {
 public:
  TrackerGpu2D();

 protected:
  void rgbdCallback(const sensor_msgs::ImageConstPtr& depth_msg,
                    const sensor_msgs::ImageConstPtr& rgb_msg,
                    const sensor_msgs::CameraInfoConstPtr& camera_info_msg);

  void rgbCallback(const sensor_msgs::ImageConstPtr& rgb_msg,
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

  void getTrackerParameters();

  void initRGB();

  void initRGBD();

  ros::NodeHandle nh_;
  // message filter
  boost::shared_ptr<image_transport::ImageTransport> rgb_it_, depth_it_;
  image_transport::SubscriberFilter sub_depth_, sub_rgb_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> sub_camera_info_;

  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo>
      SyncPolicyRGBD;
  typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::Image, sensor_msgs::CameraInfo>
    SyncPolicyRGB;


  typedef message_filters::Synchronizer<SyncPolicyRGBD> SynchronizerRGBD;
  boost::shared_ptr<SynchronizerRGBD> sync_rgbd_;
  typedef message_filters::Synchronizer<SyncPolicyRGB> SynchronizerRGB;
  boost::shared_ptr<SynchronizerRGB> sync_rgb_;

  const std::string rgb_topic_;
  const std::string depth_topic_;
  const std::string camera_info_topic_;
  int queue_size;

  ros::AsyncSpinner spinner_;

  cv::Mat rgb_image_, depth_image_;
  bool img_updated_;

  cv::Point2d mouse_start_, mouse_end_;
  bool is_mouse_dragging_, init_requested_, tracker_initialized_;

  bool use_depth_;
};

}  // end namespace

#endif  // INPUTMANAGER_H
