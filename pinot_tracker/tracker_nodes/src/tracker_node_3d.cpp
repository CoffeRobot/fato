#include "../include/tracker_node_3d.h"
#include "../../utilities/include/draw_functions.h"
#include "../../utilities/include/profiler.h"
#include "../../utilities/include/utilities.h"
#include <iostream>
#include <visualization_msgs/Marker.h>

using namespace cv;
using namespace std;

namespace pinot_tracker {

TrackerNode3D::TrackerNode3D()
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
  namedWindow("Tracker_d");
  //  namedWindow("debug");
  setMouseCallback("Tracker", TrackerNode3D::mouseCallback, this);

  publisher_ = nh_.advertise<sensor_msgs::Image>("pinot_tracker/output", 1);

  initRGBD();

  run();
}

void TrackerNode3D::mouseCallback(int event, int x, int y) {
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

void TrackerNode3D::mouseCallback(int event, int x, int y, int flags,
                                  void *userdata) {
  auto manager = reinterpret_cast<TrackerNode3D *>(userdata);
  manager->mouseCallback(event, x, y);
}

void TrackerNode3D::rgbdCallback(
    const sensor_msgs::ImageConstPtr &depth_msg,
    const sensor_msgs::ImageConstPtr &rgb_msg,
    const sensor_msgs::CameraInfoConstPtr &camera_info_msg) {
  if (!camera_matrix_initialized_) {
    ROS_INFO("Init camera parameters");
    //    getCameraMatrix(camera_info_msg, params_.camera_matrix);
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

    camera_matrix_initialized_ = true;
  }

  cout << " encoding " << depth_msg->encoding << "\n" << endl;

  cv::Mat rgb, depth;

  readImage(rgb_msg, rgb);
  readImage(depth_msg, depth);

  Mat3f points(depth.rows, depth.cols, cv::Vec3f(0, 0, 0));
  depthTo3d(depth, params_.camera_model.cx(), params_.camera_model.cy(),
            params_.camera_model.fx(), params_.camera_model.fy(), points);

  rgb_image_ = rgb.clone();
  depth_image_ = depth.clone();
  image_points_ = points.clone();
  img_updated_ = true;
}

void TrackerNode3D::rgbCallback(
    const sensor_msgs::ImageConstPtr &rgb_msg,
    const sensor_msgs::CameraInfoConstPtr &camera_info_msg) {
  cv::Mat rgb;

  readImage(rgb_msg, rgb);

  rgb_image_ = rgb;
  img_updated_ = true;
}

void TrackerNode3D::initRGBD() {
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
      boost::bind(&TrackerNode3D::rgbdCallback, this, _1, _2, _3));
}

void TrackerNode3D::run() {
  spinner_.start();

  ROS_INFO("INPUT: init tracker");

  Tracker3D tracker;

  auto &profiler = Profiler::getInstance();

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = chrono::system_clock::now();

  ros::Rate r(100);
  while (ros::ok()) {
    // ROS_INFO_STREAM("Main thread [" << boost::this_thread::get_id() << "].");

    if (img_updated_) {
      Mat tmp;
      rgb_image_.copyTo(tmp);
      if (mouse_start_.x != mouse_end_.x && !tracker_initialized_) {
        rectangle(tmp, mouse_start_, mouse_end_, Scalar(255, 0, 0), 3);
        img_updated_ = false;
      }
      if (!tracker_initialized_) {
        Mat depth_mapped;
        applyColorMap(depth_image_, depth_mapped);
        imshow("Tracker", tmp);
        imshow("Tracker_d", depth_mapped);
        waitKey(1);
      }

      if (init_requested_) {
        ROS_INFO("INPUT: tracker intialization requested");
        tracker.init(params_, rgb_image_, image_points_, mouse_start_,
                     mouse_end_);
        init_requested_ = false;
        tracker_initialized_ = true;
        ROS_INFO("Tracker initialized!");
        destroyWindow("Tracker");
        destroyWindow("Tracker_d");
        waitKey(1);
      }

      if (tracker_initialized_) {
        profiler->start("total");
        Mat out;
        tracker.next(rgb_image_, image_points_);
        profiler->stop("total");

        rgb_image_.copyTo(out);
        tracker.drawObjectLocation(out);
        vector<Point3f *> pts, votes;
        tracker.getActivePoints(pts, votes);

        drawCentroidVotes(pts, votes, Point2f(params_.camera_model.cx(),
                                              params_.camera_model.cy()),
                          true, params_.camera_model.fx(), out);

        publishPose(tracker.getCurrentCentroid(), tracker.getRotation());

        end = chrono::system_clock::now();
        float elapsed =
            chrono::duration_cast<chrono::seconds>(end - start).count();

        stringstream ss;
        ss << "Tracker run in ms: ";
        if (elapsed > 3.0) {
          start = end;
          ss << profiler->getProfile() << "\n";
        } else
          ss << profiler->getTime("total") << "\n";

        ROS_INFO(ss.str().c_str());

        cv_bridge::CvImage cv_img;
        cv_img.image = out;
        cv_img.encoding = sensor_msgs::image_encodings::BGR8;
        publisher_.publish(cv_img.toImageMsg());
        r.sleep();
        img_updated_ = false;
      }
    }

    // r.sleep();
  }
}

void TrackerNode3D::publishPose(Point3f mean_point,
                                Eigen::Quaterniond rotation) {
  tf::Vector3 centroid(mean_point.x, -mean_point.y, mean_point.z);

  tf::Transform transform;
  transform.setOrigin(centroid);
  tf::Quaternion q(rotation.x(), rotation.y(), rotation.z(),rotation.w());

  transform.setRotation(q);

  transform_broadcaster_.sendTransform(tf::StampedTransform(
      transform, ros::Time::now(), "camera_rgb_optical_frame", "object_centroid"));

}

}  // end namespace

int main(int argc, char *argv[]) {
  ROS_INFO("Starting tracker input");
  ros::init(argc, argv, "pinot_tracker_node_3d");

  pinot_tracker::TrackerNode3D tracker_node;

  return 0;
}
