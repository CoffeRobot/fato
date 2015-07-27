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
      rgb_topic_("/camera/rgb/image_color"),
      depth_topic_("/camera/depth_registered/hw_registered/image_rect_raw"),
      camera_info_topic_("/camera/rgb/camera_info"),
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
  namedWindow("Tracker_d");
  //  namedWindow("debug");
  setMouseCallback("Tracker", Projection::mouseCallback, this);

  publisher_ = nh_.advertise<sensor_msgs::Image>("pinot_tracker/output", 1);
  markers_publisher_ =
      nh_.advertise<visualization_msgs::Marker>("pinot_tracker/pose", 10);

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
    ROS_INFO("Init camera parameters");

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

    cout << "camera model: " << params_.camera_model.fx() << "\n";

    cout << camera_matrix << endl;
    cout << params_.camera_model.projectionMatrix() << endl;

    //    waitKey(0);
    camera_matrix_initialized_ = true;
  }

  //  cout << " encoding " << depth_msg->encoding << "\n" << endl;

  cv::Mat rgb, depth;

  readImage(rgb_msg, rgb);
  readImage(depth_msg, depth);

  rgb_image_ = rgb;
  depth_image_ = depth;
  img_updated_ = true;
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

void Projection::analyzeCube(const cv::Mat &disparity, const Point2d &top_left,
                             const Point2d &bottom_right, Point3f &median_p,
                             Point3f &min_p, Point3f &max_p) {
  Mat1b mask(disparity.rows, disparity.cols, uchar(0));
  rectangle(mask, top_left, bottom_right, 255, -1);

  vector<float> depth_x, depth_y, depth_z;
  Point3f min_depth(numeric_limits<float>::max(), numeric_limits<float>::max(),
                    numeric_limits<float>::max());
  Point3f max_depth(-numeric_limits<float>::max(),
                    -numeric_limits<float>::max(),
                    -numeric_limits<float>::max());

  float average = 0;
  float counter = 0;
  for (int i = 0; i < disparity.rows; ++i) {
    for (int j = 0; j < disparity.cols; ++j) {
      if (mask.at<uchar>(i, j) == 255 && disparity.at<Vec3f>(i, j)[2] != 0 &&
          is_valid<float>(disparity.at<Vec3f>(i, j)[2])) {
        float x = disparity.at<Vec3f>(i, j)[0];
        float y = disparity.at<Vec3f>(i, j)[1];
        float z = disparity.at<Vec3f>(i, j)[2];

        depth_z.push_back(z);
        depth_x.push_back(x);
        depth_y.push_back(y);
        average += z;

        min_depth.x = std::min(min_depth.x, x);
        min_depth.y = std::min(min_depth.y, y);
        min_depth.z = std::min(min_depth.z, z);

        max_depth.x = std::max(max_depth.x, x);
        max_depth.y = std::max(max_depth.y, y);
        max_depth.z = std::max(max_depth.z, z);

        counter++;
      }
    }
  }

  sort(depth_x.begin(), depth_x.end());
  sort(depth_y.begin(), depth_y.end());
  sort(depth_z.begin(), depth_z.end());

  auto size = depth_x.size();

  if (size == 0) {
    cout << "No point to calculate median \n";
    return;
  }
  float median_x, median_y, median_z;

  if (size % 2 == 0) {
    median_x = (depth_x.at(size / 2) + depth_x.at(size / 2 + 1));
    median_y = (depth_y.at(size / 2) + depth_y.at(size / 2 + 1));
    median_z = (depth_z.at(size / 2) + depth_z.at(size / 2 + 1));
  } else {
    median_x = depth_x.at(size / 2);
    median_y = depth_y.at(size / 2);
    median_z = depth_z.at(size / 2);
  }

  median_p.x = median_x;
  median_p.y = median_y;
  median_p.z = median_z;

  min_p = min_depth;
  max_p = max_depth;

  cout << "depth: avg " << average / counter << " median x " << median_x
       << " median y " << median_y << " median z " << median_z
       << "\n min point " << min_depth << " max point " << max_depth << endl;
}

void Projection::publishPose(Point3f &center, Eigen::Quaterniond pose,
                             std::vector<Point3f> &back_points,
                             std::vector<Point3f> &front_points) {
  tf::Vector3 centroid(center.x, center.y, center.z);

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

void Projection::run() {
  spinner_.start();

  ROS_INFO("INPUT: init tracker");

  params_.debug_path = "/home/alessandro/Debug";
  params_.readRosConfigFile();

  Tracker3D tracker;

  auto &profiler = Profiler::getInstance();
  Point3f mean_pt, min_pt, max_pt;

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = chrono::system_clock::now();

  BoundingCube cube;

  ros::Rate r(100);
  while (ros::ok()) {
    // ROS_INFO_STREAM("Main thread [" << boost::this_thread::get_id() << "].");

    if (img_updated_) {
      Mat tmp;
      rgb_image_.copyTo(tmp);

      if (mouse_start_.x != mouse_end_.x) {
        rectangle(tmp, mouse_start_, mouse_end_, Scalar(255, 0, 0), 3);
      }

      Mat depth_mapped;
      applyColorMap(depth_image_, depth_mapped);
      if (!tracker_initialized_) {
        imshow("Tracker", tmp);
        imshow("Tracker_d", depth_mapped);
      }

      if (init_requested_ && !tracker_initialized_) {
        Mat3f points(depth_image_.rows, depth_image_.cols, cv::Vec3f(0, 0, 0));
        depthTo3d(depth_image_, params_.camera_model.cx(),
                  params_.camera_model.cy(), params_.camera_model.fx(),
                  params_.camera_model.fy(), points);

        Mat1b mask(depth_image_.rows, depth_image_.cols, uchar(0));
        rectangle(mask, mouse_start_, mouse_end_, uchar(255), -1);

        // cloud_image_.copyTo(points);
        ROS_INFO("INPUT: tracker intialization requested");
        if (tracker.init(params_, rgb_image_, points, mouse_start_,
                         mouse_end_) < 0) {
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
        projectPoint(
            params_.camera_model.fx(),
            Point2f(params_.camera_model.cx(), params_.camera_model.cy()),
            tracker.getCurrentCentroid(), center);
        circle(out, center, 5, Scalar(255, 0, 0), -1);

        cube.initCube(points, mouse_start_, mouse_end_);
        cube.setPerspective(params_.camera_model.fx(),
                            params_.camera_model.cx(),
                            params_.camera_model.cy());

        analyzeCube(points, mouse_start_, mouse_end_, mean_pt, min_pt, max_pt);

        Point2f img_center(params_.camera_model.cx(),
                           params_.camera_model.cy());

        drawBoundingCube(cube.getFrontPoints(), cube.getBackPoints(),
                         params_.camera_model.fx(), img_center, out);

        cout << "cube centroid " << tracker.getCurrentCentroid() << endl;

        imshow("Tracker", out);
        waitKey(30);

      } else if (tracker_initialized_) {

        Mat3f points(depth_image_.rows, depth_image_.cols, cv::Vec3f(0, 0, 0));
        depthTo3d(depth_image_, params_.camera_model.cx(),
                  params_.camera_model.cy(), params_.camera_model.fx(),
                  params_.camera_model.fy(), points);

        // analyzeCube(points, mouse_start_, mouse_end_, mean_pt, min_pt,
        // max_pt);
        Mat out;

        // publishPose(mean_pt, min_pt, max_pt);
        profiler->start("frame_time");
        tracker.next(rgb_image_, points);
        profiler->stop("frame_time");

        Point3f pt = tracker.getCurrentCentroid();
        auto front_points = tracker.getFrontBB();
        auto back_points = tracker.getBackBB();
        Eigen::Quaterniond q = tracker.getPoseQuaternion();

        publishPose(pt, q, back_points, front_points);

        end = chrono::system_clock::now();
        float elapsed =
            chrono::duration_cast<chrono::seconds>(end - start).count();

        stringstream ss;
        ss << "Tracker run in ms: ";
        if (elapsed > 3.0) {
          start = end;
          ss << "\n" << profiler->getProfile() << "\n";
        } else
          ss << profiler->getTime("frame_time") << "\n";

        ROS_INFO(ss.str().c_str());

        rgb_image_.copyTo(out);

        tracker.drawObjectLocation(out);
        Point2f center;
        projectPoint(
            params_.camera_model.fx(),
            Point2f(params_.camera_model.cx(), params_.camera_model.cy()),
            tracker.getCurrentCentroid(), center);
        circle(out, center, 5, Scalar(255, 0, 0), -1);

        vector<Point3f *> pts, votes;
        tracker.getActivePoints(pts, votes);

        drawCentroidVotes(pts, votes, Point2f(params_.camera_model.cx(),
                                              params_.camera_model.cy()),
                          true, params_.camera_model.fx(), out);

        imshow("Tracker", out);

        Mat ransac;
        depth_mapped.copyTo(ransac);

        // tracker.drawRansacEstimation(ransac);

        vector<Point3f> front, back;
        cube.rotate(tracker.getCurrentCentroid(), tracker.getPoseMatrix(),
                    front, back);
        drawBoundingCube(
            front, back, params_.camera_model.fx(),
            Point2f(params_.camera_model.cx(), params_.camera_model.cy()),
            ransac);


        char c = waitKey(30);

        if (c == 'e') {
           cube.estimateDepth(points, tracker.getCurrentCentroid(),
                              tracker.getPoseMatrix(), ransac);
           cout << "Estimating depth..." << endl;
           imshow("Tracker_d", ransac);
           waitKey(0);

        }

        imshow("Tracker_d", ransac);
        if (c == 's') cout << "save" << endl;
      }





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
