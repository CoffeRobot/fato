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
#include <opencv2/highgui/highgui.hpp>
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
#include <fstream>

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
  SyntheticRenderer() : initialized_(false), next_frame_requested_(false)
  {
      t_req_ = Eigen::Vector3d(0,0,0.5);
      r_req_ = Eigen::Vector3d(M_PI,0,0);
  }

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

  void renderObject()
  {
      double T[] = {t_req_[0], t_req_[1], t_req_[2]};
      double R[] = {r_req_[0], r_req_[1], r_req_[2]};
      std::vector<pose::TranslationRotation3D> TR(1);
      TR.at(0) = pose::TranslationRotation3D(T, R);

      Eigen::Matrix3d rot = TR.at(0).eigenRotation();

//      cout << "Eigen rotation " << endl;
//      cout << rot << endl;

      rendering_engine->render(TR);
  }

  void downloadTexture(cv::Mat& img)
  {
      std::vector<uchar4> h_texture(camera_image.rows * camera_image.cols);
      downloadRenderedImg(*rendering_engine, h_texture);

      cv::Mat img_rgba(camera_image.rows, camera_image.cols, CV_8UC4,
                       h_texture.data());

      cv::cvtColor(img_rgba, img, CV_RGBA2BGR);
  }

  void blendWithCamera(cv::Mat& img)
  {
      for (auto i = 0; i < camera_image.rows; ++i) {
        for (auto j = 0; j < camera_image.cols; ++j) {
          cv::Vec3b &pixel = img.at<cv::Vec3b>(i, j);
          if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0) {
            cv::Vec3b &cp = camera_image.at<cv::Vec3b>(i, j);
            pixel[0] = cp[2];
            pixel[1] = cp[1];
            pixel[2] = cp[0];
          }
        }
      }
  }

  void calculateFlow(cv::Mat& next, std::vector<cv::Point2f>& next_pts)
  {

      if(prev_points_.size() == 0)
          return;

      vector<uchar> next_status, proj_status;
      vector<float> next_errors, proj_errors;

      vector<cv::Point2f> proj_pts, tmp_next_pts;

      cv::calcOpticalFlowPyrLK(prev_gray_, next, prev_points_, tmp_next_pts,
                               next_status, next_errors);
      cv::calcOpticalFlowPyrLK(next, prev_gray_, tmp_next_pts, proj_pts, proj_status,
                               proj_errors);


      std::vector<cv::Point2f> tmp_prev;
      std::vector<float> tmp_depth;

      for(int i = 0; i < tmp_next_pts.size(); ++i)
      {
        float error = fato::getDistance(prev_points_.at(i), proj_pts.at(i));

        if(proj_status.at(i) == 1 && error < 20)
        {
            tmp_prev.push_back(prev_points_.at(i));
            tmp_depth.push_back(prev_depth_.at(i));
            next_pts.push_back(tmp_next_pts.at(i));
        }
      }

      std::swap(prev_points_,tmp_prev);
      std::swap(prev_depth_,tmp_depth);
  }

  void saveResults(std::vector<cv::Point2f>& prev_pts, std::vector<float>& prev_depth,
                   std::vector<cv::Point2f>& next_pts, std::vector<float>& next_depth)
  {
    if(prev_pts.size() == 0 || next_pts.size() == 0)
            return;

    string filename = "/home/alessandro/debug/flow.hdf5";


    if (boost::filesystem::exists(boost::filesystem::path(filename))) {
        boost::filesystem::remove(boost::filesystem::path(filename));
    }


    util::HDF5File out_file(filename);
    ofstream file("/home/alessandro/debug/flow.txt");

    vector<float> prev, next;
    for(auto i = 0; i < prev_pts.size(); ++i)
    {
        prev.push_back(prev_pts.at(i).x);
        prev.push_back(prev_pts.at(i).y);
        prev.push_back(prev_depth.at(i));
        next.push_back(next_pts.at(i).x);
        next.push_back(next_pts.at(i).y);
        next.push_back(next_depth.at(i));
        file << "[" << prev_pts.at(i).x  << "," << prev_pts.at(i).y << "," << prev_depth.at(i) << "] "
             << "[" << next_pts.at(i).x  << "," << next_pts.at(i).y << "," << next_depth.at(i) << "]\n";
    }

    std::vector<int> size{prev_pts.size(), 3};

    out_file.writeArray("prev_points", prev, size);
    out_file.writeArray("next_points", next, size);
    out_file.writeScalar<double>("fx", camera_matrix_.at<double>(0, 0));
    out_file.writeScalar<double>("fy", camera_matrix_.at<double>(1, 1));
    out_file.writeScalar<double>("cx", camera_matrix_.at<double>(0, 2));
    out_file.writeScalar<double>("cy", camera_matrix_.at<double>(1, 2));

  }

  bool serviceCallback(fato_tracker_tests::RenderService::Request &req,
                       fato_tracker_tests::RenderService::Response &res) {

    ROS_INFO("received a rendering request");

    t_req_[0] = req.x;
    t_req_[1] = req.y;
    t_req_[2] = req.z;

    r_req_[0] = req.wx;
    r_req_[1] = req.wy;
    r_req_[2] = req.wz;

    res.result = true;
    res.fx = camera_matrix_.at<double>(0,0);
    res.fy = camera_matrix_.at<double>(1,1);
    res.cx = camera_matrix_.at<double>(0,2);
    res.cy = camera_matrix_.at<double>(1,2);

    renderObject();

    cv::Mat rgb, gray, depth;
    vector<float> depth_buffer;

    downloadTexture(rgb);
    zbuffer2Z(depth_buffer);

    if(req.reset_rendering)
    {
        ROS_INFO("reset requested");
        initialize_tracking(rgb, depth_buffer);
        return true;
    }

    blendWithCamera(rgb);
    cv::cvtColor(rgb, gray, CV_BGR2GRAY);
    std::vector<cv::Point2f> next_pts;
    std::vector<float> next_depth;

    calculateFlow(gray, next_pts);

    stringstream ss;
    ss << "zbuffer size " << depth_buffer.size() << " pts size " << next_pts.size() << "\n";
    ROS_INFO(ss.str().c_str());

    for(auto pt : next_pts)
    {
        float depth = 0;
        if(pt.x < 0 || pt.x > width_ || pt.y < 0 || pt.y > height_)
            depth = 0;
        else
            depth = depth_buffer.at(pt.x + pt.y * width_);

        next_depth.push_back(depth);
    }

    cv::imwrite("/home/alessandro/debug/image.png", rgb);
    saveResults(prev_points_, prev_depth_, next_pts, next_depth);

    prev_gray_ = gray.clone();
    std::swap(prev_points_, next_pts);
    std::swap(prev_depth_, next_depth);

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

  void initialize_tracking(cv::Mat& rgb, const std::vector<float>& depth_buffer)
  {
      cv::Mat grey;
      cv::cvtColor(rgb, grey, CV_BGR2GRAY);
      std::vector<cv::KeyPoint> kps;
      cv::ORB orbExtractor;
      orbExtractor.detect(grey, kps);

      prev_points_.clear();
      prev_depth_.clear();

      for (auto kp : kps) {
        cv::Point2f pt = kp.pt;
        prev_points_.push_back(pt);
        float depth = depth_buffer.at(pt.x + pt.y * width_);

        prev_depth_.push_back(depth);
      }

      cout << "points for tracking " << prev_points_.size() << endl;

      keypoints_extracted_ = true;

      prev_gray_ = grey.clone();
  }

  void run() {
    ros::Rate r(100);

    int frame_counter = 0;
    int fst_id = 0;
    int scd_id = 1;

    float t_inc = 0.02 / 30.0;
    Eigen::Vector3f pos(0, 0, 0.5);
    Eigen::Vector3f inc(t_inc, 0, 0);

//    Config params;

//    std::unique_ptr<fato::FeatureMatcher> derived =
//        std::unique_ptr<fato::BriskMatcher>(new fato::BriskMatcher);

//    fato::TrackerMB tracker(params, fato::BRISK, std::move(derived));
//    tracker.addModel(model_name);

    bool tracker_initialized = false;
    keypoints_extracted_ = false;

    std::vector<cv::Point2f> keypoints;
    vector<float> prev_d, next_d;
    std::vector<cv::Point3f> points;
    cv::Mat prev, prev_gray;

    while (ros::ok()) {
      Eigen::Vector3f t_inc;

      if (!tracker_initialized && initialized_) {
//        tracker.setCameraMatrix(camera_matrix_);
        tracker_initialized = true;
      }

      if (image_updated && initialized_) {


        renderObject();

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

        if (!keypoints_extracted_) {

            initialize_tracking(res, depth_buffer);
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

        for (auto pt : prev_points_) {
          cv::circle(res, pt, 3, cv::Scalar(255, 0, 0), 2);
        }

        //const fato::Target &target = tracker.getTarget();

        cv::Point3f center(0, 0, 0);
//        for (auto i = 0; i < target.active_points.size(); ++i) {
//          int id = target.active_to_model_.at(i);

//          cv::Scalar color;

//          if ((int)target.point_status_.at(id) == 0)
//            color = cv::Scalar(255, 0, 0);
//          else if ((int)target.point_status_.at(id) == 2)
//            color = cv::Scalar(0, 255, 0);

//          cv::circle(res, target.active_points.at(i), 3, color);
//          // center += model_points.at(i);
//        }

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
  Eigen::Vector3d r_req_;


  std::vector<cv::Point2f> prev_points_;
  std::vector<float> prev_depth_;

  cv::Mat camera_matrix_, prev_gray_;
  int width_, height_;
  bool keypoints_extracted_;
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
