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
#include <opencv2/calib3d/calib3d.hpp>
#include <string>
#include <iostream>
#include <sstream>

#include <utility_kernels.h>
#include <utility_kernels_pose.h>

#include "../../fato_rendering/include/multiple_rigid_models_ogre.h"
#include "../../fato_rendering/include/windowless_gl_context.h"

#include "../../tracker/include/tracker_model_vx.h"
#include "../../tracker/include/tracker_model.h"
#include "../../tracker/include/pose_estimation.h"
#include "../../tracker/include/constants.h"

#include "../../io/include/VideoWriter.h"

#include "../../fato_rendering/include/device_1d.h"
#include "../../utilities/include/draw_functions.h"
#include "../../utilities/include/utilities.h"
#include "../../utilities/include/hdf5_file.h"

#include "../include/tracker_node_vx.hpp"

using namespace cv;
using namespace std;

namespace fato {

void getPose(const cv::Mat &rot, const cv::Mat &tra,
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
}

void renderObject(Eigen::Transform<double, 3, Eigen::Affine> &pose,
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
}

void downloadRenderedImg(pose::MultipleRigidModelsOgre &model_ogre,
                         std::vector<uchar4> &h_texture) {
  util::Device1D<uchar4> d_texture(480 * 640);
  vision::convertFloatArrayToGrayRGBA(d_texture.data(), model_ogre.getTexture(),
                                      640, 480, 1.0, 2.0);
  h_texture.resize(480 * 640);
  d_texture.copyTo(h_texture);
}


void blendRendered(const cv::Mat& rendered, cv::Mat& out)
{
    for(auto i = 0; i < rendered.rows; ++i)
    {
        for(auto j = 0; j < rendered.cols; ++j)
        {
            const uchar& px = rendered.at<uchar>(i,j);
            Vec3b& px_out = out.at<Vec3b>(i,j);
            if(px != 0)
            {
                px_out[0] -= px;
                if(px_out[0] < 0)
                    px_out[0] = 0;
            }
        }
    }
}

TrackerModelVX::TrackerModelVX()
    : nh_(),
      rgb_topic_("tracker/image_raw"),
      camera_info_topic_("tracker/camera_info"),
      queue_size(5),
      is_mouse_dragging_(false),
      img_updated_(false),
      init_requested_(false),
      tracker_initialized_(false),
      mouse_start_(0, 0),
      mouse_end_(0, 0),
      spinner_(0),
      camera_matrix_initialized(false),
      params_(),
      cam_info_manager_(nh_)
{
  cvStartWindowThread();

  publisher_ = nh_.advertise<sensor_msgs::Image>("fato_tracker/output_pnp", 1);

  render_publisher_ =
      nh_.advertise<sensor_msgs::Image>("fato_tracker/render", 1);

  flow_publisher_ =
      nh_.advertise<sensor_msgs::Image>("fato_tracker/output_flow", 1);

  service_server_ = nh_.advertiseService(
      "tracker_service", &TrackerModelVX::serviceCallback, this);


  loadParameters(nh_);

  initRGB();

  run();
}

void TrackerModelVX::loadParameters(ros::NodeHandle &nh)
{
    string camera_info_file,model_name, obj_file;
    if (!ros::param::get("fato/camera_info_url", camera_info_file)) {
      throw std::runtime_error("cannot read camera infor url");
    }
    if (!ros::param::get("fato/model/h5_file", model_name)) {
      throw std::runtime_error("cannot read h5 file param");
    }
    if (!ros::param::get("fato/model/obj_file", obj_file)) {
      throw std::runtime_error("cannot read obj file param");
    }
    if (!ros::param::get("fato/parallel", params_.parallel)) {
      throw std::runtime_error("cannot read parallel param");
    }
    int pyr_level;
    if (!ros::param::get("fato/pyr_levels", pyr_level)) {
      throw std::runtime_error("cannot read pyr_levels param");
    }
    params_.pyr_levels = pyr_level;
    int lk_iters;
    if (!ros::param::get("fato/lk_num_iters",lk_iters )) {
      throw std::runtime_error("cannot read lk_num_iters param");
    }
    params_.lk_num_iters = lk_iters;
    int win_size;
    if (!ros::param::get("fato/lk_win_size",win_size)) {
      throw std::runtime_error("cannot read lk_win_size param");
    }
    win_size =  params_.lk_win_size;
    if (!ros::param::get("fato/lk_epsilon", params_.lk_epsilon)) {
      throw std::runtime_error("cannot read lk_epsilon param");
    }
    if (!ros::param::get("fato/flow_threshold", params_.flow_threshold)) {
      throw std::runtime_error("cannot read flow_threshold param");
    }
    int capacity;
    if (!ros::param::get("fato/array_capacity", capacity)) {
      throw std::runtime_error("cannot read array_capacity param");
    }
    capacity = params_.array_capacity;
    int cell_size;
    if (!ros::param::get("fato/detector_cell_size", cell_size)) {
      throw std::runtime_error("cannot read detector_cell_size param");
    }
    params_.detector_cell_size = cell_size;
    if (!ros::param::get("fato/use_harris", params_.use_harris_detector)) {
      throw std::runtime_error("cannot read use_harris param");
    }
    if (!ros::param::get("fato/harris_k", params_.harris_k)) {
      throw std::runtime_error("cannot read harris_k param");
    }
    if (!ros::param::get("fato/harris_thresh", params_.harris_thresh)) {
      throw std::runtime_error("cannot read harris_thresh param");
    }
    int fast_type;
    if (!ros::param::get("fato/fast_type", fast_type)) {
      throw std::runtime_error("cannot read fast_type param");
    }
    fast_type = params_.fast_type;
    int thresh;
    if (!ros::param::get("fato/fast_thresh", thresh)) {
      throw std::runtime_error("cannot read fast_thresh param");
    }
    thresh = params_.fast_thresh;
    if (!ros::param::get("fato/iterations_cam", params_.iterations_m_real)) {
      throw std::runtime_error("cannot read iterations cam param");
    }
    if (!ros::param::get("fato/iterations_synth", params_.iterations_m_synth)) {
      throw std::runtime_error("cannot read iterations real param");
    }

    cam_info_manager_.loadCameraInfo(camera_info_file);
    const sensor_msgs::CameraInfo& camera_info_msg = cam_info_manager_.getCameraInfo();
    camera_matrix_ =
        cv::Mat(3, 4, CV_64F, (void *)camera_info_msg.P.data()).clone();
    camera_matrix_initialized = true;

    params_.descriptors_file = model_name;
    params_.model_file = obj_file;

    cout << "tracker parameters loaded!" << endl;
}

void TrackerModelVX::initRGBSynch() {
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

void TrackerModelVX::initRGB()
{
    rgb_it_.reset(new image_transport::ImageTransport(nh_));

    sub_rgb_.subscribe(*rgb_it_, rgb_topic_, 1,
                       image_transport::TransportHints("raw"));
    sub_rgb_.registerCallback(
                boost::bind(&TrackerModelVX::rgbCallbackNoInfo, this, _1));
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

void TrackerModelVX::rgbCallbackNoInfo(const sensor_msgs::ImageConstPtr &rgb_msg)
{
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

//  ofstream file("/home/alessandro/debug/debug.txt");
//  cv::VideoWriter video_writer("/home/alessandro/Downloads/pose_estimation.avi",
//                               CV_FOURCC('D', 'I', 'V', 'X'), 30,
//                               cv::Size(640, 480), true);
  spinner_.start();

  ros::Rate r(100);

  bool camera_is_set = false;

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

  Mat cam(3, 3, CV_64FC1);

  // set to true to run concurrent threads
  params_.parallel = true;

  while (ros::ok()) {

    if (img_updated_) {
      if (!camera_matrix_initialized)
        continue;
      else if (camera_matrix_initialized && !camera_is_set) {
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

        vx_tracker_ =
            unique_ptr<TrackerVX>(new TrackerVX(params_, std::move(derived)));

        cout << "Tracker initialized!" << endl;

        camera_is_set = true;
      }

      auto begin = chrono::high_resolution_clock::now();
      if(params_.parallel)
      {
        vx_tracker_->parNext(rgb_image_);
      }
      else
      {
          vx_tracker_->next(rgb_image_);
      }
      auto end = chrono::high_resolution_clock::now();
      frame_counter++;
      average_time +=
          chrono::duration_cast<chrono::microseconds>(end - begin).count();

      const Target &target = vx_tracker_->getTarget();

      Pose p = target.weighted_pose;

      glm::mat4 pose_glm = p.toGL();

      flow_output = rgb_image_.clone();

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

          } else if (target.point_status_.at(id) == fato::KpStatus::PNP)
            color = Scalar(0, 255, 255);


          circle(flow_output, target.active_points.at(i), 1, color);

          

          if (frame_counter > 100) {
            frame_counter = 0;
            average_time = 0;
          }
        }
      }

      vx_tracker_->printProfile();


      //imshow("depth buffer", out_depth);
      //waitKey(1);

      float total = target.real_pts_ + target.synth_pts_;
      float ratio = target.real_pts_ / total;

      pair<float, float> last_vals = target.target_history_.getLastVal();
      stringstream ss, ss1, ss2;
      ss << "average time: " << (average_time / frame_counter) / 1000.0;
      ss << " cam pts " << target.real_pts_ << " synth " << target.synth_pts_ << " r " << ratio;
      ss1 << "vel " << target.target_history_.getAvgVelocity() << " conf "
          << target.target_history_.getConfidence().first << " last "
          << last_vals.first;
      ss2 << "angular " << target.target_history_.getAvgAngular() << " conf "
          << target.target_history_.getConfidence().second << " last "
          << last_vals.second;

      drawInformationHeader(Point2f(0, 0), ss.str(), 0.8, flow_output.cols, 20,
                            flow_output);
      drawInformationHeader(Point2f(0, 20), ss1.str(), 0.8, flow_output.cols,
                            20, flow_output);
      drawInformationHeader(Point2f(0, 40), ss2.str(), 0.8, flow_output.cols,
                            20, flow_output);
      // cout << target.active_to_model_.size() << endl;

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

        Mat rend_mat = vx_tracker_->getRenderedPose();

        if(target.target_found_)
            blendRendered(rend_mat, flow_output);

        cv_bridge::CvImage cv_img, cv_rend, cv_flow;

        //Mat out_depth = vx_tracker_->getDepthBuffer();

        cv_rend.image = rend_mat;
        cv_rend.encoding = sensor_msgs::image_encodings::MONO8;
        render_publisher_.publish(cv_rend.toImageMsg());

        cv_flow.image = flow_output;
        cv_flow.encoding = sensor_msgs::image_encodings::BGR8;
        flow_publisher_.publish(cv_flow.toImageMsg());

        //video_writer.write(flow_output);

        //        Size sz1 = flow_output.size();
        //        Size sz2 = rend_mat.size();
        //        Mat im3(sz1.height, sz1.width + sz2.width, CV_8UC3);
        //        flow_output.copyTo(im3(Rect(0, 0, sz1.width, sz1.height)));
        //        rend_mat.copyTo(im3(Rect(sz1.width, 0, sz2.width,
        //        sz2.height)));

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

  fato::TrackerModelVX manager;

  ros::shutdown();

  return 0;
}
