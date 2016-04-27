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

#include "../include/tracker_node_2d.h"
#include <boost/thread.hpp>
#include <mutex>
#include <opencv2/highgui/highgui.hpp>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <profiler.h>
#include <draw_functions.h>
#include <opencv2/calib3d/calib3d.hpp>

#include <utility_kernels.h>
#include <utility_kernels_pose.h>
#include <hdf5_file.h>
#include <multiple_rigid_models_ogre.h>

#include "../../fato_rendering/include/multiple_rigid_models_ogre.h"
#include "../../fato_rendering/include/windowless_gl_context.h"
#include "../../utilities/include/hdf5_file.h"
#include "../include/tracker_model_based.h"
#include "../../utilities/include/utilities.h"
#include "../../tracker/include/tracker_model.h"
#include "../../tracker/include/pose_estimation.h"
#include "../../tracker/include/constants.h"
#include "../../utilities/include/draw_functions.h"
#include "../../io/include/VideoWriter.h"
#include "../../utilities/include/device_1d.h"

using namespace cv;
using namespace std;

namespace fato {

TrackerModel::TrackerModel(string model_file, string obj_file)
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
      obj_file_(obj_file) {
  cvStartWindowThread();

  publisher_ = nh_.advertise<sensor_msgs::Image>("fato_tracker/output", 1);

  render_publisher_ =
      nh_.advertise<sensor_msgs::Image>("fato_tracker/render", 1);

  getTrackerParameters();

  initRGB();

  run(model_file);
}

void TrackerModel::initRGB() {
  rgb_it_.reset(new image_transport::ImageTransport(nh_));
  sub_camera_info_.subscribe(nh_, camera_info_topic_, 1);
  /** kinect node settings */
  sub_rgb_.subscribe(*rgb_it_, rgb_topic_, 1,
                     image_transport::TransportHints("raw"));

  sync_rgb_.reset(new SynchronizerRGB(SyncPolicyRGB(queue_size), sub_rgb_,
                                      sub_camera_info_));

  sync_rgb_->registerCallback(
      boost::bind(&TrackerModel::rgbCallback, this, _1, _2));
}

void TrackerModel::rgbCallback(
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

void TrackerModel::getTrackerParameters() {
  stringstream ss;

  ss << "Tracker Input: \n";

  ROS_INFO(ss.str().c_str());
}

void TrackerModel::rgbdCallback(
    const sensor_msgs::ImageConstPtr &depth_msg,
    const sensor_msgs::ImageConstPtr &rgb_msg,
    const sensor_msgs::CameraInfoConstPtr &camera_info_msg) {
  ROS_INFO("Tracker2D: rgbd usupported");
}

void TrackerModel::run(string model_file) {
  ROS_INFO("TrackerMB: init...");

  // Create dummy GL context before cudaGL init
  render::WindowLessGLContext dummy(10, 10);
  // setup the engines
  unique_ptr<pose::MultipleRigidModelsOgre> rendering_engine;

  spinner_.start();

  Config params;

  std::unique_ptr<FeatureMatcher> derived =
      std::unique_ptr<BriskMatcher>(new BriskMatcher);

  util::HDF5File in_file(model_file);

  VideoWriter video_writer("/home/alessandro/Downloads/", "pose_estimation.avi",
                           640, 480, 1, 30);

  TrackerMB tracker(params, BRISK, std::move(derived));
  ROS_INFO("TrackerMB: setting model...");
  tracker.addModel(model_file);
  ROS_INFO("TrackerMB: starting...");
  ros::Rate r(100);

  Mat prev_rotation, prev_translation;

  bool target_found = false;
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
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > rot_render_eig(
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
        tracker.setCameraMatrix(cam);

        rendering_engine = unique_ptr<pose::MultipleRigidModelsOgre>(
            new pose::MultipleRigidModelsOgre(
                rgb_image_.cols, rgb_image_.rows, cam.at<double>(0, 0),
                cam.at<double>(1, 1), cam.at<double>(0, 2),
                cam.at<double>(1, 2), 0.01, 10.0));
        rendering_engine->addModel(obj_file_);

        camera_is_set = true;
      }

      if (!background_learned) {
        tracker.learnBackground(rgb_image_);
        background_learned = true;
      }

      // cout << "Frame" << endl;
      tracker.computeNextSequential(rgb_image_);

      char c = waitKey(1);
      if (c == 'b') {
        std::cout << "Learning background" << std::endl;
        tracker.learnBackground(rgb_image_);
      }

      vector<Point2f> valid_points;
      vector<Point3f> model_points;

      tracker.getActiveModelPoints(model_points, valid_points);

      vector<int> inliers;

      Mat cam(3, 3, CV_64FC1);
      for (int i = 0; i < cam.rows; ++i) {
        for (int j = 0; j < cam.cols; ++j) {
          cam.at<double>(i, j) = camera_matrix_.at<double>(i, j);
        }
      }

      const Target &target = tracker.getTarget();

      cv::Point3f center(0, 0, 0);
      for (auto i = 0; i < target.active_points.size(); ++i) {
        int id = target.active_to_model_.at(i);

        Scalar color;

        if (target.point_status_.at(id) == fato::KpStatus::MATCH)
          color = Scalar(255, 0, 0);
        else if (target.point_status_.at(id) == fato::KpStatus::TRACK)
          color = Scalar(0, 255, 0);

        circle(rgb_image_, target.active_points.at(i), 3, color);
        // center += model_points.at(i);
      }

      drawObjectPose(target.centroid_, cam, target.rotation, target.translation,
                     rgb_image_);

      const Mat &tvec = target.translation;
      const Mat &rvec = target.rotation_vec;

      Eigen::Matrix3d b(3, 3);
      Eigen::Matrix3d a(3, 3);
      for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
          a(i, j) = target.rotation.at<double>(i, j);
          b(i, j) = target.rotation_custom.at<double>(i, j);
        }
      }

      double determinant = a.determinant();

      cout << "determint cv " << determinant << " cus " << b.determinant() << endl;

      double err = 0;

      for(auto i = 0; i < 3; ++i)
      {
          for(auto j = 0; j < 3; ++j)
          {
              double dt = a(i,j) - b(i,j);
              err +=  sqrt(err * err);
          }
      }

      if(err > 1)
      {
          for(auto i = 0; i < 3; ++i)
          {
              for(auto j = 0; j < 3; ++j)
              {
                  cout << std::setprecision(2) << a(i,j) << "|" << b(i,j) << " ";
              }
              cout << "\n";
          }
          cout << "\n";
      }

      double T[] = {tvec.at<double>(0, 0), tvec.at<double>(0, 1),
                    tvec.at<double>(0, 2)};
      double R[] = {rvec.at<double>(0, 0), rvec.at<double>(0, 1),
                    rvec.at<double>(0, 2)};

      std::vector<pose::TranslationRotation3D> TR(1);
      TR.at(0) = pose::TranslationRotation3D(T, R);
      rendering_engine->render(TR);

      //      Eigen::Transform<double, 3, Eigen::Affine> t_render;
      //      getPose(target.rotation, target.translation, t_render);
      //      renderObject(t_render, *rendering_engine);
      std::vector<uchar4> h_texture(480 * 640);
      downloadRenderedImg(*rendering_engine, h_texture);

      cv::Mat img_rgba(480, 640, CV_8UC4, h_texture.data());
      cv::Mat img_rgb;
      cv::cvtColor(img_rgba, img_rgb, CV_RGBA2BGR);
      //      cout << "t rot" << target.rotation.rows << " " <<
      //      target.rotation.cols
      //           << "\n";
      //      cout << "t tra" << target.translation.rows << " "
      //           << target.translation.cols << "\n";
      cv_bridge::CvImage cv_img, cv_rend;
      cv_img.image = rgb_image_;
      cv_img.encoding = sensor_msgs::image_encodings::BGR8;
      publisher_.publish(cv_img.toImageMsg());

      cv_rend.image = img_rgb;
      cv_rend.encoding = sensor_msgs::image_encodings::BGR8;
      render_publisher_.publish(cv_rend.toImageMsg());

      video_writer.write(rgb_image_);
      r.sleep();
    }
  }

  video_writer.stopRecording();
  cv::destroyAllWindows();
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

  fato::TrackerModel manager(model_name, obj_file);

  ros::shutdown();

  return 0;
}
