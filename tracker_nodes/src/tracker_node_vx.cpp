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
#include <boost/thread.hpp>
#include <mutex>
#include <opencv2/highgui/highgui.hpp>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <profiler.h>
#include <draw_functions.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <string>
#include <iostream>

#include <utility_kernels.h>
#include <utility_kernels_pose.h>


#include "../../fato_rendering/include/multiple_rigid_models_ogre.h"
#include "../../fato_rendering/include/windowless_gl_context.h"

#include "../../tracker/include/tracker_model_vx.h"
#include "../../tracker/include/tracker_model.h"
#include "../../tracker/include/pose_estimation.h"
#include "../../tracker/include/constants.h"

#include "../../io/include/VideoWriter.h"

#include "../../utilities/include/device_1d.h"
#include "../../utilities/include/draw_functions.h"
#include "../../utilities/include/utilities.h"
#include "../../utilities/include/hdf5_file.h"

#include "../include/tracker_node_vx.hpp"

using namespace cv;
using namespace std;

namespace fato {

TrackerModelVX::TrackerModelVX(string descriptor_file, string model_file)
    : nh_(),
      rgb_topic_("/image_raw"),
      camera_info_topic_("/camera_info"),
      queue_size(5),
      is_mouse_dragging_(false),
      img_updated_(false),
      init_requested_(false),
      tracker_initialized_(false),
      mouse_start_(0, 0),
      mouse_end_(0, 0),
      spinner_(0),
      camera_matrix_initialized(false),
      obj_file_(descriptor_file),
      params_()
{
  cvStartWindowThread();

  publisher_ = nh_.advertise<sensor_msgs::Image>("fato_tracker/output_pnp", 1);

  render_publisher_ =
      nh_.advertise<sensor_msgs::Image>("fato_tracker/render", 1);

  flow_publisher_ =
      nh_.advertise<sensor_msgs::Image>("fato_tracker/output_flow", 1);

  service_server_ = nh_.advertiseService("tracker_service",
                                         &TrackerModelVX::serviceCallback, this);

  initRGB();

  params_.descriptors_file = descriptor_file;
  params_.model_file = model_file;

  run();
}

void TrackerModelVX::initRGB() {
  rgb_it_.reset(new image_transport::ImageTransport(nh_));
  sub_camera_info_.subscribe(nh_, camera_info_topic_, 1);
  /** kinect node settings */
  sub_rgb_.subscribe(*rgb_it_, rgb_topic_, 1,
                     image_transport::TransportHints("raw"));

  sync_rgb_.reset(new SynchronizerRGB(SyncPolicyRGB(queue_size), sub_rgb_,
                                      sub_camera_info_));

  sync_rgb_->registerCallback(
      boost::bind(&TrackerModelVX::rgbCallback, this, _1, _2));
}

void TrackerModelVX::rgbCallback(
    const sensor_msgs::ImageConstPtr &rgb_msg,
    const sensor_msgs::CameraInfoConstPtr &camera_info_msg) {
  if (!camera_matrix_initialized) {
    camera_matrix_ =
        cv::Mat(3, 4, CV_64F, (void *)camera_info_msg->P.data()).clone();
    camera_matrix_initialized = true;
  }

  Mat rgb;
  readImage(rgb_msg, rgb);
  cvtColor(rgb, rgb_image_, CV_RGB2BGR);
  img_updated_ = true;
}

bool TrackerModelVX::serviceCallback(
    fato_tracker_nodes::TrackerService::Request &req,
    fato_tracker_nodes::TrackerService::Response &res) {
  res.result = true;
  if (req.stop_matcher) {
    stop_matcher = true;
    cout << "Matcher stopped!" << endl;
  } else {
    cout << "Matcher restarted" << endl;
    stop_matcher = false;
  }

  return true;
}

void TrackerModelVX::rgbdCallback(
    const sensor_msgs::ImageConstPtr &depth_msg,
    const sensor_msgs::ImageConstPtr &rgb_msg,
    const sensor_msgs::CameraInfoConstPtr &camera_info_msg) {
  ROS_INFO("Tracker2D: rgbd usupported");
}

void TrackerModelVX::run() {
  ROS_INFO("TrackerMB: init...");

  // Create dummy GL context before cudaGL init
  render::WindowLessGLContext dummy(10, 10);
  // setup the engines
  // unique_ptr<pose::MultipleRigidModelsOgre> rendering_engine;

  spinner_.start();


  ros::Rate r(100);

  bool camera_is_set = false;
  bool background_learned = false;

  auto getPose = [](const cv::Mat &rot, const cv::Mat &tra,
                    Eigen::Transform<double, 3, Eigen::Affine> &t_render) {
    Eigen::Matrix3d rot_view;
    rot_view = Eigen::AngleAxisd(rot.at<double>(2), Eigen::Vector3d::UnitZ()) *
               Eigen::AngleAxisd(rot.at<double>(1), Eigen::Vector3d::UnitY()) *
               Eigen::AngleAxisd(rot.at<double>(0), Eigen::Vector3d::UnitX());

    // rotate around x to 'render' just as Ogre
    Eigen::Matrix3d Rx_180 = Eigen::Matrix<double, 3, 3>::Identity();
    Rx_180(1, 1) = -1.0;
    Rx_180(2, 2) = -1.0;
    rot_view *= Rx_180;

    Eigen::Translation<double, 3> translation(
        tra.at<double>(0), tra.at<double>(1), tra.at<double>(2));

    // apply tra_center -> rot_view to bounding box
    Eigen::Transform<double, 3, Eigen::Affine> t = rot_view * translation;

    // compose render transform (tra_center -> rot_view -> tra_z_shift)
    t_render = rot_view * translation;
  };

  auto renderObject = [](Eigen::Transform<double, 3, Eigen::Affine> &pose,
                         pose::MultipleRigidModelsOgre &model_ogre) {
    double tra_render[3];
    double rot_render[9];
    Eigen::Map<Eigen::Vector3d> tra_render_eig(tra_render);
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> rot_render_eig(
        rot_render);
    tra_render_eig = pose.translation();
    rot_render_eig = pose.rotation();

    std::vector<pose::TranslationRotation3D> TR(1);
    TR.at(0).setT(tra_render);
    TR.at(0).setR_mat(rot_render);

    model_ogre.render(TR);
  };

  auto downloadRenderedImg = [](pose::MultipleRigidModelsOgre &model_ogre,
                                std::vector<uchar4> &h_texture) {
    util::Device1D<uchar4> d_texture(480 * 640);
    vision::convertFloatArrayToGrayRGBA(
        d_texture.data(), model_ogre.getTexture(), 640, 480, 1.0, 2.0);
    h_texture.resize(480 * 640);
    d_texture.copyTo(h_texture);
  };

  Mat flow_output;

  stop_matcher = false;

  vector<cv::Scalar> axis;
  axis.push_back(cv::Scalar(255, 255, 0));
  axis.push_back(cv::Scalar(0, 255, 255));
  axis.push_back(cv::Scalar(255, 0, 255));

  vector<cv::Scalar> axis2;
  axis2.push_back(cv::Scalar(125, 125, 0));
  axis2.push_back(cv::Scalar(0, 125, 125));
  axis2.push_back(cv::Scalar(125, 0, 125));


  float average_time = 0.0f;
  float frame_counter = 0;

  while (ros::ok()) {
    if (img_updated_) {
      if (!camera_matrix_initialized)
        continue;
      else if (camera_matrix_initialized && !camera_is_set) {
        Mat cam(3, 3, CV_64FC1);
        for (int i = 0; i < cam.rows; ++i) {
          for (int j = 0; j < cam.cols; ++j) {
            cam.at<double>(i, j) = camera_matrix_.at<double>(i, j);
          }
        }

        params_.image_width = rgb_image_.cols;
        params_.image_height = rgb_image_.rows;
        params_.fx = cam.at<double>(0, 0);
        params_.fy = cam.at<double>(1, 1);
        params_.cx = cam.at<double>(0, 2);
        params_.cy = cam.at<double>(1, 2);

        std::unique_ptr<FeatureMatcher> derived =
            std::unique_ptr<BriskMatcher>(new BriskMatcher);

        vx_tracker_ = unique_ptr<TrackerVX>(new TrackerVX(params_, std::move(derived)));

        cout << "Tracker initialized!" << endl;

        camera_is_set = true;
      }

      if (!background_learned) {
        //tracker.learnBackground(rgb_image_);
        background_learned = true;
      }

      auto begin = chrono::high_resolution_clock::now();
      vx_tracker_->next(rgb_image_);
      auto end = chrono::high_resolution_clock::now();
      frame_counter++;
      average_time += chrono::duration_cast<chrono::microseconds>(end-begin).count();


      char c = waitKey(1);
      if (c == 'b') {
        std::cout << "Learning background" << std::endl;
        //tracker.learnBackground(rgb_image_);
      }

      vector<int> inliers;

      Mat cam(3, 3, CV_64FC1);
      for (int i = 0; i < cam.rows; ++i) {
        for (int j = 0; j < cam.cols; ++j) {
          cam.at<double>(i, j) = camera_matrix_.at<double>(i, j);
        }
      }

      const Target &target = vx_tracker_->getTarget();

      flow_output = rgb_image_.clone();

      drawObjectPose(target.centroid_, cam, target.rotation, target.translation,
                     rgb_image_);

      Pose p_k = target.kal_pnp_pose;

      pair<Mat, Mat> pcv = p_k.toCV();

      drawObjectPose(target.centroid_, cam, pcv.first, pcv.second, axis,
                     rgb_image_);

      cv::Mat rotation = cv::Mat(3, 3, CV_64FC1, 0.0f);
      cv::Mat translation = cv::Mat(1, 3, CV_32FC1, 0.0f);

      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          rotation.at<double>(i, j) = target.pose_(i, j);
        }
        translation.at<float>(i) = target.pose_(i, 3);
      }


      if (target.target_found_) {


        auto w_pose = target.weighted_pose.toCV();
        drawObjectPose(target.centroid_, cam, w_pose.first, w_pose.second, axis,
                       flow_output);

        for (auto i = 0; i < target.active_points.size(); ++i) {
          int id = target.active_to_model_.at(i);

          Scalar color(0, 0, 255);

          if (target.point_status_.at(id) == fato::KpStatus::MATCH)
            color = Scalar(255, 0, 0);
          else if (target.point_status_.at(id) == fato::KpStatus::TRACK) {
            color = Scalar(0, 255, 0);

            // circle(rgb_image_, target.prev_points_.at(i), 1, color);
            line(flow_output, target.prev_points_.at(i),
                 target.active_points.at(i), Scalar(255, 0, 0), 1);
          }
          else if (target.point_status_.at(id) == fato::KpStatus::PNP) color =
              Scalar(0, 255, 255);

          // circle(rgb_image_, target.prev_points_.at(i), 3, color);
          circle(flow_output, target.active_points.at(i), 1, color);
          // line(rgb_image_, target.prev_points_.at(i),
          // target.active_points.at(i), Scalar(255,0,0), 1);
          // center += model_points.at(i);

          cout << "--- average time: " << (average_time/frame_counter)/1000.0 << " --- \n";
          vx_tracker_->printProfile();

          if(frame_counter > 100 )
          {
              frame_counter = 0;
              average_time = 0;
          }

        }
      }

      //cout << target.active_to_model_.size() << endl;

      if (target.active_to_model_.size()) {
        std::vector<Point3f> model_pts;
        std::vector<Point3f> rel_pts;

        for (auto i = 0; i < target.active_to_model_.size(); ++i) {
          model_pts.push_back(
              target.model_points_.at(target.active_to_model_.at(i)));
          rel_pts.push_back(
              target.rel_distances_.at(target.active_to_model_.at(i)));
        }

        std::vector<Point2f> model_2d, rel_2d;

        projectPoints(model_pts, target.rotation, target.translation, cam,
                      Mat(), model_2d);
        projectPoints(rel_pts, target.rotation, target.translation, cam, Mat(),
                      rel_2d);

        double T[] = {target.pose_(0, 3), target.pose_(1, 3),
                      target.pose_(2, 3)};
        Eigen::Matrix3d rotation_temp;
        Eigen::Vector3d translation_vect;
        for (auto i = 0; i < 3; ++i) {
          for (auto j = 0; j < 3; ++j) {
            rotation_temp(i, j) = target.pose_(i, j);
          }
          translation_vect(i) = target.pose_(i, 3);
        }

        double tra_render[3];
        double rot_render[9];
        Eigen::Map<Eigen::Vector3d> tra_render_eig(tra_render);
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> rot_render_eig(
            rot_render);
        tra_render_eig = translation_vect;
        rot_render_eig = rotation_temp;

        std::vector<pose::TranslationRotation3D> TR(1);
        TR.at(0).setT(tra_render);
        TR.at(0).setR_mat(rot_render);

        Mat rend_mat = vx_tracker_->getRenderedPose();

        cv_bridge::CvImage cv_img, cv_rend, cv_flow;
        cv_img.image = rgb_image_;
        cv_img.encoding = sensor_msgs::image_encodings::BGR8;
        publisher_.publish(cv_img.toImageMsg());

        cv_rend.image = rend_mat;
        cv_rend.encoding = sensor_msgs::image_encodings::MONO8;
        render_publisher_.publish(cv_rend.toImageMsg());

        cv_flow.image = flow_output;
        cv_flow.encoding = sensor_msgs::image_encodings::BGR8;
        flow_publisher_.publish(cv_flow.toImageMsg());

//        Size sz1 = flow_output.size();
//        Size sz2 = rend_mat.size();
//        Mat im3(sz1.height, sz1.width + sz2.width, CV_8UC3);
//        flow_output.copyTo(im3(Rect(0, 0, sz1.width, sz1.height)));
//        rend_mat.copyTo(im3(Rect(sz1.width, 0, sz2.width, sz2.height)));

        img_updated_ = false;
        r.sleep();
      }
    }

    // video_writer.stopRecording();
    // cv::destroyAllWindows();
  }
}

}  // end namespace

int main(int argc, char *argv[]) {
  ROS_INFO("Starting tracker input");
  ros::init(argc, argv, "fato_tracker_model_node");

  string model_name, obj_file;

  if (!ros::param::get("fato/model/h5_file", model_name)) {
    throw std::runtime_error("cannot read h5 file param");
  }

  if (!ros::param::get("fato/model/obj_file", obj_file)) {
    throw std::runtime_error("cannot read obj file param");
  }

  fato::TrackerModelVX manager(model_name, obj_file);

  ros::shutdown();

  return 0;
}
