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
#include <iostream>
#include <visualization_msgs/Marker.h>
#include <memory>
#include <opencv2/highgui/highgui.hpp>

#include "../include/projection.h"
#include "../../utilities/include/draw_functions.h"
#include "../../utilities/include/profiler.h"
#include "../../utilities/include/utilities.h"
#include "../../tracker/include/bounding_cube.h"
#include "../../utilities/include/visualization_ros.h"

using namespace cv;
using namespace std;

namespace pinot_tracker {

Projection::Projection()
    : nh_(),
      rgb_topic_("/tracker_input/rgb"),
      depth_topic_("/tracker_input/depth"),
      camera_info_topic_("/tracker_input/camera_info"),
      queue_size(5),
      spinner_(0),
      is_mouse_dragging_(false),
      img_updated_(false),
      init_requested_(false),
      tracker_initialized_(false),
      camera_matrix_initialized_(false),
      mouse_start_(0, 0),
      mouse_end_(0, 0),
      params_() {
  cvStartWindowThread();
  namedWindow("Tracker");
  namedWindow("Experiments");
  //  namedWindow("debug");
  setMouseCallback("Tracker", Projection::mouseCallback, this);

  publisher_ = nh_.advertise<sensor_msgs::Image>("pinot_tracker/output", 1);
  markers_publisher_ =
      nh_.advertise<visualization_msgs::Marker>("pinot_tracker/pose", 10);

#ifdef VERBOSE_LOGGING
  cout << "Tracker initialized in verbose mode!!!" << endl;
#endif

  initRGBD();

  run();
}

void Projection::readImage(const sensor_msgs::Image::ConstPtr msgImage,
                           cv::Mat &image) const {
  cv_bridge::CvImageConstPtr pCvImage;
  pCvImage = cv_bridge::toCvShare(msgImage, msgImage->encoding);
  pCvImage->image.copyTo(image);
}

void Projection::mouseCallback(int event, int x, int y) {
  auto set_point = [this](int x, int y) {
    if (x < mouse_start_.x) {
      mouse_end_.x = mouse_start_.x;
      mouse_start_.x = x;
    } else
      mouse_end_.x = x;

    if (y < mouse_start_.y) {
      mouse_end_.y = mouse_start_.y;
      mouse_start_.y = y;
    } else
      mouse_end_.y = y;
  };

  if (event == EVENT_LBUTTONDOWN) {
    mouse_start_.x = x;
    mouse_start_.y = y;
    mouse_end_ = mouse_start_;
    is_mouse_dragging_ = true;
  } else if (event == EVENT_MOUSEMOVE && is_mouse_dragging_) {
    set_point(x, y);
  } else if (event == EVENT_LBUTTONUP) {
    set_point(x, y);
    is_mouse_dragging_ = false;
    init_requested_ = true;
  }
}

void Projection::mouseCallback(int event, int x, int y, int flags,
                               void *userdata) {
  auto manager = reinterpret_cast<Projection *>(userdata);
  manager->mouseCallback(event, x, y);
}

void Projection::rgbdCallback(
    const sensor_msgs::ImageConstPtr &depth_msg,
    const sensor_msgs::ImageConstPtr &rgb_msg,
    const sensor_msgs::CameraInfoConstPtr &camera_info_msg) {
  if (!camera_matrix_initialized_) {
    ROS_INFO("Initializing camera parameters...");

    Mat camera_matrix_full =
        cv::Mat(3, 4, CV_64F, (void *)camera_info_msg->P.data()).clone();

    Mat camera_matrix(3, 3, CV_64F);

    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j)
        camera_matrix.at<double>(i, j) = camera_matrix_full.at<double>(i, j);
    }

    //    getCameraMatrix(camera_info_msg, params_.camera_matrix);
    params_.camera_model.fromCameraInfo(camera_info_msg);
    params_.camera_matrix = camera_matrix.clone();
    params_.image_width = camera_info_msg->width;
    params_.image_height = camera_info_msg->height;

    //    waitKey(0);
    camera_matrix_initialized_ = true;
    ROS_INFO("Initialized camera parameters...");
  }

  //  cout << " encoding " << depth_msg->encoding << "\n" << endl;

  cv::Mat rgb, depth;

  readImage(rgb_msg, rgb);
  readImage(depth_msg, depth);

  input_mutex.lock();
  rgb_image_ = rgb.clone();
  depth_image_ = depth.clone();
  img_updated_ = true;
  input_mutex.unlock();
}

void Projection::rgbCallback(
    const sensor_msgs::ImageConstPtr &rgb_msg,
    const sensor_msgs::CameraInfoConstPtr &camera_info_msg) {
  cv::Mat rgb;

  readImage(rgb_msg, rgb);

  rgb_image_ = rgb;
  img_updated_ = true;
}

void Projection::initRGBD() {
  depth_it_.reset(new image_transport::ImageTransport(nh_));
  rgb_it_.reset(new image_transport::ImageTransport(nh_));
  sub_camera_info_.subscribe(nh_, camera_info_topic_, 1);
  /** kinect node settings */
  sub_depth_.subscribe(*depth_it_, depth_topic_, 1,
                       image_transport::TransportHints("raw"));
  sub_rgb_.subscribe(*rgb_it_, rgb_topic_, 1,
                     image_transport::TransportHints("compressed"));

  sync_rgbd_.reset(new SynchronizerRGBD(SyncPolicyRGBD(queue_size), sub_depth_,
                                        sub_rgb_, sub_camera_info_));
  sync_rgbd_->registerCallback(
      boost::bind(&Projection::rgbdCallback, this, _1, _2, _3));
}

void Projection::publishPose(Point3f &center, Eigen::Quaterniond pose,
                             std::vector<Point3f> &back_points,
                             std::vector<Point3f> &front_points) {
  tf::Vector3 centroid(center.x, -center.y, center.z);

  tf::Transform transform;
  transform.setOrigin(centroid);
  tf::Quaternion q;
  q.setX(pose.x());
  q.setY(pose.y());
  q.setZ(pose.z());
  q.setW(pose.w());

  transform.setRotation(q);

  transform_broadcaster_.sendTransform(
      tf::StampedTransform(transform, ros::Time::now(),
                           "camera_rgb_optical_frame", "object_centroid"));

  vector<visualization_msgs::Marker> faces;

  getCubeMarker(front_points, back_points, faces);

  for (auto face : faces) markers_publisher_.publish(face);
}

void Projection::initTracker(Tracker3D &tracker, BoundingCube &cube) {
  Mat3f points(depth_image_.rows, depth_image_.cols, cv::Vec3f(0, 0, 0));
  depthTo3d(depth_image_, params_.camera_model.cx(), params_.camera_model.cy(),
            params_.camera_model.fx(), params_.camera_model.fy(), points);

  Mat1b mask(depth_image_.rows, depth_image_.cols, uchar(0));
  rectangle(mask, mouse_start_, mouse_end_, uchar(255), -1);

  // cloud_image_.copyTo(points);
  ROS_INFO("INPUT: tracker intialization requested");
  if (tracker.init(params_, rgb_image_, points, mouse_start_, mouse_end_) < 0) {
    ROS_WARN("Error initializing the tracker!");
    return;
  } else {
    ROS_INFO("Tracker correctly intialized!");
  }

  init_requested_ = false;
  tracker_initialized_ = true;
  Mat out;
  // tracker.computeNext(rgb_image_, depth_image_, out);
  rgb_image_.copyTo(out);
  Point2f center;
  projectPoint(params_.camera_model.fx(),
               Point2f(params_.camera_model.cx(), params_.camera_model.cy()),
               tracker.getCurrentCentroid(), center);
  circle(out, center, 5, Scalar(255, 0, 0), -1);

  cube.setPerspective(params_.camera_model.fx(), params_.camera_model.fy(),
                      params_.camera_model.cx(), params_.camera_model.cy());
  cube.initCube(points, mouse_start_, mouse_end_);
  cube.setVects(tracker.getFrontVects(), tracker.getBackVects());

  Point2f img_center(params_.camera_model.cx(), params_.camera_model.cy());

  drawBoundingCube(cube.getFrontPoints(), cube.getBackPoints(),
                   params_.camera_model.fx(), img_center, out);

#ifdef VERBOSE_LOGGING
  cout << "cube centroid " << tracker.getCurrentCentroid() << endl;
  cout << cube.str() << endl;
#endif

  imshow("Tracker", out);
  waitKey(0);
}

void Projection::updateTracker(Tracker3D &tracker, const Mat3f &points) {
  auto &profiler = Profiler::getInstance();
  profiler->start("frame_time");
  tracker.next(rgb_image_, points);
  profiler->stop("frame_time");
}

void Projection::estimateCube(Tracker3D &tracker, BoundingCube &cube,
                              const Mat3f &points, Mat &out) {
  auto &profiler = Profiler::getInstance();

  profiler->start("cube");

  vector<float> visibility_ratio = tracker.getVisibilityRatio();

  const auto &center = tracker.getCurrentCentroid();

  if (!is_valid(center.x) || !is_valid(center.y) || !is_valid(center.z)) return;

  cube.estimateDepth(points, center, tracker.getPoseMatrix(), visibility_ratio,
                     out);
  profiler->stop("cube");
}

void Projection::drawTrackerResults(Tracker3D &tracker, Mat &out) {
  Point3f pt = tracker.getCurrentCentroid();
  auto front_points = tracker.getFrontBB();
  auto back_points = tracker.getBackBB();
  Eigen::Quaterniond q = tracker.getPoseQuaternion();

  rgb_image_.copyTo(out);

  try {
    //tracker.drawObjectLocation(out);

    drawObjectPose(
        tracker.getCurrentCentroid(), params_.camera_model.fx(),
        Point2f(params_.camera_model.cx(), params_.camera_model.cy()),
        tracker.getPoseMatrix(), out);

    Point2f center;
    projectPoint(params_.camera_model.fx(),
                 Point2f(params_.camera_model.cx(), params_.camera_model.cy()),
                 tracker.getCurrentCentroid(), center);
    circle(out, center, 5, Scalar(255, 0, 0), -1);

    vector<Point3f *> pts, votes;
    tracker.getActivePoints(pts, votes);

    drawCentroidVotes(pts, votes, Point2f(params_.camera_model.cx(),
                                          params_.camera_model.cy()),
                      true, params_.camera_model.fx(), out);


  }
  catch (cv::Exception &e) {
    cout << "error drawing results: " << e.what() << endl;
    return;
  }

  try {
    publishPose(pt, q, back_points, front_points);
  }
  catch (cv::Exception &e) {
    cout << "error publishing the pose: " << e.what() << endl;
    return;
  }

  imshow("Tracker", out);
}

void Projection::run() {
  spinner_.start();

  ROS_INFO("INPUT: init tracker");
  params_.readRosConfigFile();

  Tracker3D tracker;

  auto &profiler = Profiler::getInstance();

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = chrono::system_clock::now();

  unique_ptr<VideoWriter> video_recorder;

  if (params_.save_output) {
    video_recorder = unique_ptr<VideoWriter>(new VideoWriter(
        params_.output_path + "output.avi", CV_FOURCC('X', 'V', 'I', 'D'), 30,
        Size(params_.image_width, params_.image_height), true));
  }

  BoundingCube cube;

  ros::Rate r(100);
  while (ros::ok()) {
    // ROS_INFO_STREAM("Main thread [" << boost::this_thread::get_id() << "].");

    Mat rgb_out;
    Mat experiments_out;

    if (img_updated_) {
      Mat tmp;
      rgb_image_.copyTo(tmp);
      Mat depth_mapped;
      applyColorMap(depth_image_, depth_mapped);

      if (mouse_start_.x != mouse_end_.x) {
        rectangle(tmp, mouse_start_, mouse_end_, Scalar(255, 0, 0), 3);
      }

      if (!tracker_initialized_) {
        rgb_out = tmp.clone();
        experiments_out = depth_mapped.clone();
      }

      if (init_requested_ && !tracker_initialized_) {
        initTracker(tracker, cube);
        rgb_out = tmp.clone();
        experiments_out = depth_mapped.clone();
      } else if (tracker_initialized_) {
        // getting 3d points from disparity

        input_mutex.lock();
        Mat3f points(depth_image_.rows, depth_image_.cols, cv::Vec3f(0, 0, 0));
        depthTo3d(depth_image_, params_.camera_model.cx(),
                  params_.camera_model.cy(), params_.camera_model.fx(),
                  params_.camera_model.fy(), points);
        input_mutex.unlock();

        Mat tmp;
        rgb_image_.copyTo(tmp);
        // experiments_out = depth_mapped.clone();
        experiments_out = rgb_image_.clone();
#ifdef TRACKER_VERBOSE_LOGGING
        cout << "updating tracker " << endl;
#endif
        updateTracker(tracker, points);
#ifdef TRACKER_VERBOSE_LOGGING
        cout << "drawing tracker " << endl;
#endif
        drawTrackerResults(tracker, tmp);
#ifdef TRACKER_VERBOSE_LOGGING
        cout << "estimating cube depth " << endl;
#endif
        estimateCube(tracker, cube, points, experiments_out);
#ifdef TRACKER_VERBOSE_LOGGING
        cout << "estimated cube depth " << endl;
#endif
        if (params_.save_output) {
          //video_recorder->write(experiments_out);
          video_recorder->write(tmp);
        }

        rgb_out = tmp.clone();

        end = chrono::system_clock::now();
        float elapsed =
            chrono::duration_cast<chrono::seconds>(end - start).count();

        stringstream ss;
        ss << "Tracker profile in ms: \n";
        if (elapsed > 3.0) {
          start = end;
          ss << "\n" << profiler->getProfile() << "\n";
          ROS_INFO(ss.str().c_str());
        }
        char c = waitKey(30);

        if (c == 's') cout << "save" << endl;
      }

      imshow("Tracker", rgb_out);
      imshow("Experiments", experiments_out);

      img_updated_ = false;
    }
  }

  // r.sleep();
}

}  // end namespace

int main(int argc, char *argv[]) {
  ROS_INFO("Starting tracker input");
  ros::init(argc, argv, "pinot_tracker_node_3d");

  pinot_tracker::Projection tracker_node;

  return 0;
}
