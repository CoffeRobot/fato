#include "../include/tracker_node_3d.h"
#include "../../utilities/include/draw_functions.h"
#include "../../utilities/include/profiler.h"
#include <iostream>


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
    params_.camera_model.fromCameraInfo(camera_info_msg);

    cout << "camera model: " << params_.camera_model.fx() << "\n";

    //    cout << params_.camera_matrix << endl;
    //    cout << params_.camera_model.projectionMatrix() << endl;

    //    waitKey(0);
    camera_matrix_initialized_ = true;
  }

  cout << " encoding " << depth_msg->encoding << "\n" << endl;

  cv::Mat rgb, depth;

  readImage(rgb_msg, rgb);
  readImage(depth_msg, depth);

  rgb_image_ = rgb;
  depth_image_ = depth;
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
        tracker.init(params_, rgb_image_, depth_image_, mouse_start_,
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

        cout << "Depth image type " << depth_image_.channels() << " "
             << depth_image_.type();

        Mat out;
        tracker.computeNext(rgb_image_, depth_image_, out);
        profiler->stop("total");

        stringstream ss;
        ss << "Tracker run in ms: " << profiler->getTime("total") << "\n";
        ss << "Centroid " << tracker.getCurrentCentroid() << "\n";
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

}  // end namespace

int main(int argc, char *argv[]) {
  ROS_INFO("Starting tracker input");
  ros::init(argc, argv, "pinot_tracker_node_3d");

  pinot_tracker::TrackerNode3D tracker_node;

  return 0;
}
