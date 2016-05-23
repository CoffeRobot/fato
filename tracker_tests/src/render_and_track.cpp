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

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core/core.hpp>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <sensor_msgs/CameraInfo.h>
#include <message_filters/subscriber.h>
#include <math.h>
#include <atomic>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

#include "fato_tracker_tests/RenderService.h"

#include <device_1d.h>
#include <hdf5_file.h>
#include <utilities.h>
#include <tracker_model.h>
#include <feature_matcher.hpp>
#include <constants.h>

#include "../../fato_rendering/include/multiple_rigid_models_ogre.h"
#include "../../fato_rendering/include/windowless_gl_context.h"

#include <utility_kernels.h>
#include <utility_kernels_pose.h>

using namespace std;

ros::Publisher pose_publisher, rendering_publisher, disparity_publisher;

cv::Mat camera_image;
string img_topic, info_topic, model_name, obj_file;

std::vector<Eigen::Vector3f> translations;
std::vector<Eigen::Vector3f> rotations;

auto downloadRenderedImg = [](pose::MultipleRigidModelsOgre &model_ogre,
                              std::vector<uchar4> &h_texture) {
  util::Device1D<uchar4> d_texture(480 * 640);
  vision::convertFloatArrayToGrayRGBA(d_texture.data(), model_ogre.getTexture(),
                                      640, 480, 1.0, 2.0);
  h_texture.resize(480 * 640);
  d_texture.copyTo(h_texture);
};

class SyntheticRenderer {
 public:
  SyntheticRenderer() : initialized_(false), next_frame_requested_(false) {}

  void rgbCallback(const sensor_msgs::ImageConstPtr &rgb_msg,
                   const sensor_msgs::CameraInfoConstPtr &camera_info_msg) {
    cv_bridge::CvImageConstPtr pCvImage;
    pCvImage = cv_bridge::toCvShare(rgb_msg, rgb_msg->encoding);
    pCvImage->image.copyTo(camera_image);
    // ROS_INFO("Img received ");

    if (!initialized_) {
      cv::Mat cam =
          cv::Mat(3, 4, CV_64F, (void *)camera_info_msg->P.data()).clone();
      rendering_engine = unique_ptr<pose::MultipleRigidModelsOgre>(
          new pose::MultipleRigidModelsOgre(
              camera_image.cols, camera_image.rows, cam.at<double>(0, 0),
              cam.at<double>(1, 1), cam.at<double>(0, 2), cam.at<double>(1, 2),
              0.01, 10.0));

      camera_matrix_ = cv::Mat(3, 3, CV_64FC1);
      for (int i = 0; i < camera_matrix_.rows; ++i) {
        for (int j = 0; j < camera_matrix_.cols; ++j) {
          camera_matrix_.at<double>(i, j) = cam.at<double>(i, j);
        }
      }

      rendering_engine->addModel(obj_file);

      width_ = camera_image.cols;
      height_ = camera_image.rows;

      initialized_ = true;
    }

    image_updated = true;
  }

  void init(ros::NodeHandle &nh) {
    rgb_it_.reset(new image_transport::ImageTransport(nh));
    info_sub_.subscribe(nh, info_topic, 1);

    img_sub_.subscribe(*rgb_it_, img_topic, 1,
                       image_transport::TransportHints("compressed"));

    sync_rgb_.reset(new SynchronizerRGB(SyncPolicyRGB(5), img_sub_, info_sub_));

    sync_rgb_->registerCallback(
        boost::bind(&SyntheticRenderer::rgbCallback, this, _1, _2));

    service_server_ = nh.advertiseService(
        "render_next", &SyntheticRenderer::serviceCallback, this);
  }

  bool serviceCallback(fato_tracker_tests::RenderService::Request &req,
                       fato_tracker_tests::RenderService::Response &res) {
    res.result = true;

    next_frame_requested_ = true;

    t_req_[0] = req.x;
    t_req_[1] = req.y;
    t_req_[2] = req.z;

    q_req_.x() = req.qx;
    q_req_.y() = req.qy;
    q_req_.z() = req.qz;
    q_req_.w() = req.qw;

    ROS_INFO("received a rendering request");

    return true;
  }

  void zbuffer2Z(vector<float> &depth) {
    util::Device1D<float> d_z(height_ * width_);
    pose::convertZbufferToZ(d_z.data(), rendering_engine->getZBuffer(), width_,
                            height_, camera_matrix_.at<double>(0, 2),
                            camera_matrix_.at<double>(1, 2), 0.01, 10.0);
    depth.resize(height_ * width_, 0);
    d_z.copyTo(depth);
  }

  void depth2Disparity(vector<float> &depth, cv::Mat &disparity) {
    cv::Mat temp(height_, width_, CV_32FC1, 0.0f);

    for (auto j = 0; j < height_; ++j) {
      for (auto i = 0; i < width_; ++i) {
        int id = i + j * width_;
        if (depth.at(id) == depth.at(id)) temp.at<float>(id) = depth.at(id);
      }
    }

    double min = 0;
    double max = 8.0;
    // cv::minMaxIdx(temp, &min, &max);

    cv::Mat adjMap;
    // expand your range to 0..255. Similar to histEq();
    temp.convertTo(disparity, CV_8UC1, 255 / (max - min), -min);

    // this is great. It converts your grayscale image into a tone-mapped one,
    // much more pleasing for the eye
    // function is found in contrib module, so include contrib.hpp
    // and link accordingly
    // cv::applyColorMap(adjMap, disparity, cv::COLORMAP_JET);
  }

  void run() {
    ros::Rate r(100);

    int frame_counter = 0;
    int fst_id = 0;
    int scd_id = 1;

    float t_inc = 0.02 / 30.0;
    Eigen::Vector3f pos(0, 0, 0.5);
    Eigen::Vector3f inc(t_inc, 0, 0);

    Config params;

    std::unique_ptr<fato::FeatureMatcher> derived =
        std::unique_ptr<fato::BriskMatcher>(new fato::BriskMatcher);

    fato::TrackerMB tracker(params, fato::BRISK, std::move(derived));
    tracker.addModel(model_name);

    bool tracker_initialized = false;
    bool keypoints_extracted = false;

    std::vector<cv::Point2f> keypoints, prev_kps, next_kps;
    std::vector<cv::Point3f> points;
    cv::Mat prev, prev_gray;

    while (ros::ok()) {
      Eigen::Vector3f t_inc;

      if (!tracker_initialized && initialized_) {
        tracker.setCameraMatrix(camera_matrix_);
        tracker_initialized = true;
      }

      if (image_updated && initialized_) {
        if (next_frame_requested_) {
          frame_counter++;
          pos += inc;

          if (frame_counter % 240 == 0) {
            inc = inc * -1.0;
          }
        }

        double T[] = {pos[0], pos[1], pos[2]};
        double R[] = {M_PI, 0, 0};

        std::vector<pose::TranslationRotation3D> TR(1);
        TR.at(0) = pose::TranslationRotation3D(T, R);
        rendering_engine->render(TR);

        std::vector<uchar4> h_texture(camera_image.rows * camera_image.cols);
        downloadRenderedImg(*rendering_engine, h_texture);

        cv::Mat img_rgba(camera_image.rows, camera_image.cols, CV_8UC4,
                         h_texture.data());

        cv::Mat disparity;
        vector<float> depth_buffer;
        zbuffer2Z(depth_buffer);
        depth2Disparity(depth_buffer, disparity);

        cv::Mat res;
        cv::cvtColor(img_rgba, res, CV_RGBA2BGR);

        if (!keypoints_extracted) {
          cv::Mat grey;
          cv::cvtColor(res, grey, CV_BGR2GRAY);
          std::vector<cv::KeyPoint> kps;
          cv::ORB orbExtractor;
          orbExtractor.detect(grey, kps);

          for (auto kp : kps) {
            cv::Point2f pt = kp.pt;
            keypoints.push_back(pt);
            float depth = depth_buffer.at(pt.x + pt.y * width_);
            points.push_back(cv::Point3f(pt.x, pt.y, depth));
          }

          prev_kps = keypoints;

          keypoints_extracted = true;

          prev_gray = grey.clone();
        } else {
          cv::Mat gray;
          cv::cvtColor(res, gray, CV_BGR2GRAY);
          vector<uchar> next_status, prev_status;
          vector<float> next_errors, prev_errors;

          cv::calcOpticalFlowPyrLK(prev_gray, gray, prev_kps, next_kps,
                                   next_status, next_errors);

          prev_gray = gray.clone();
        }

        prev = res.clone();

        for (auto i = 0; i < camera_image.rows; ++i) {
          for (auto j = 0; j < camera_image.cols; ++j) {
            cv::Vec3b &pixel = res.at<cv::Vec3b>(i, j);
            if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0) {
              cv::Vec3b &cp = camera_image.at<cv::Vec3b>(i, j);
              pixel[0] = cp[2];
              pixel[1] = cp[1];
              pixel[2] = cp[0];
            }
          }
        }

        for (auto pt : next_kps) {
          cv::circle(res, pt, 3, cv::Scalar(255, 0, 0), 2);
        }

        if (next_frame_requested_) {
          tracker.computeNextSequential(res);

          //          // dump the information to file
          //          util::HDF5File out_file(h5_file_name);
          //          std::vector<int> descriptors_size{valid_point_count,
          //          dsc_size};
          //          std::vector<int> positions_size{valid_point_count, 3};
          //          out_file.writeArray("descriptors",
          //          all_filtered_descriptors,
          //                              descriptors_size, true);
          //          out_file.writeArray("positions", all_filtered_keypoints,
          //                              positions_size, true);

          prev_kps = next_kps;
          next_frame_requested_ = false;
        }

        const fato::Target &target = tracker.getTarget();

        cv::Point3f center(0, 0, 0);
        for (auto i = 0; i < target.active_points.size(); ++i) {
          int id = target.active_to_model_.at(i);

          cv::Scalar color;

          if ((int)target.point_status_.at(id) == 0)
            color = cv::Scalar(255, 0, 0);
          else if ((int)target.point_status_.at(id) == 2)
            color = cv::Scalar(0, 255, 0);

          cv::circle(res, target.active_points.at(i), 3, color);
          // center += model_points.at(i);
        }

        cv_bridge::CvImage cv_img, cv_rend;
        cv_img.image = res;
        cv_img.encoding = sensor_msgs::image_encodings::BGR8;
        cv_rend.image = disparity;
        cv_rend.encoding = sensor_msgs::image_encodings::MONO8;

        rendering_publisher.publish(cv_img.toImageMsg());
        disparity_publisher.publish(cv_rend.toImageMsg());

        image_updated = false;
      }

      r.sleep();
      ros::spinOnce();
    }
  }

 private:
  message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub_;
  image_transport::SubscriberFilter img_sub_;
  boost::shared_ptr<image_transport::ImageTransport> rgb_it_;

  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image, sensor_msgs::CameraInfo> SyncPolicyRGB;

  typedef message_filters::Synchronizer<SyncPolicyRGB> SynchronizerRGB;
  boost::shared_ptr<SynchronizerRGB> sync_rgb_;

  bool initialized_, image_updated;
  unique_ptr<pose::MultipleRigidModelsOgre> rendering_engine;

  ros::ServiceServer service_server_;

  atomic_bool next_frame_requested_;
  Eigen::Vector3d t_req_;
  Eigen::Quaterniond q_req_;

  cv::Mat camera_matrix_;
  int width_, height_;
};

void simpleMovement() {
  translations.push_back(Eigen::Vector3f(0, 0, 0.5));
  translations.push_back(Eigen::Vector3f(0.02, 0, 0.5));
  translations.push_back(Eigen::Vector3f(0.04, 0, 0.5));
  translations.push_back(Eigen::Vector3f(0.06, 0, 0.5));
  translations.push_back(Eigen::Vector3f(0.08, 0, 0.5));
  translations.push_back(Eigen::Vector3f(0.10, 0, 0.5));
  translations.push_back(Eigen::Vector3f(0.08, 0, 0.5));
  translations.push_back(Eigen::Vector3f(0.06, 0, 0.5));
  translations.push_back(Eigen::Vector3f(0.04, 0, 0.5));
  translations.push_back(Eigen::Vector3f(0.02, 0, 0.5));

  float deg2rad = M_PI / 180;

  rotations.resize(10, Eigen::Vector3f(0, 90 * deg2rad, 0));
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "fato_synthetic_scene");

  ros::NodeHandle nh;

  // Create dummy GL context before cudaGL init
  render::WindowLessGLContext dummy(10, 10);

  if (!ros::param::get("fato/camera/image_topic", img_topic)) {
    throw std::runtime_error("cannot read h5 file param");
  }

  if (!ros::param::get("fato_camera/info_topic", info_topic)) {
    throw std::runtime_error("cannot read obj file param");
  }

  if (!ros::param::get("fato/model/h5_file", model_name)) {
    throw std::runtime_error("cannot read h5 file param");
  }

  if (!ros::param::get("fato/model/obj_file", obj_file)) {
    throw std::runtime_error("cannot read obj file param");
  }

  cout << "img_topic " << img_topic << endl;
  cout << "info_topic " << info_topic << endl;

  rendering_publisher =
      nh.advertise<sensor_msgs::Image>("fato_tracker/synthetic", 1);
  disparity_publisher =
      nh.advertise<sensor_msgs::Image>("fato_tracker/synthetic_disparity", 1);

  SyntheticRenderer sr;

  sr.init(nh);

  sr.run();

  ros::shutdown();

  return 0;
}
