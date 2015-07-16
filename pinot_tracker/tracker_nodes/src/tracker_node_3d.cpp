#include "../include/tracker_node_3d.h"
#include "../../utilities/include/draw_functions.h"
#include <iostream>


namespace pinot_tracker{


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
      params_()
{
    cvStartWindowThread();
    namedWindow("Tracker");
    namedWindow("Tracker_d");
    setMouseCallback("Tracker", TrackerNode3D::mouseCallback, this);

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


  if(!camera_matrix_initialized_)
  {
    ROS_INFO("Init camera parameters");
//    getCameraMatrix(camera_info_msg, params_.camera_matrix);
    params_.camera_model.fromCameraInfo(camera_info_msg);

//    cout << params_.camera_matrix << endl;
//    cout << params_.camera_model.projectionMatrix() << endl;

//    waitKey(0);
    camera_matrix_initialized_ = true;
  }

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

void TrackerNode3D::run()
{
    spinner_.start();

    ROS_INFO("INPUT: init tracker");

    Tracker3D tracker(params_);

    ros::Rate r(1);
    while (ros::ok()) {
      // ROS_INFO_STREAM("Main thread [" << boost::this_thread::get_id() << "].");

      if (img_updated_) {

        if (mouse_start_.x != mouse_end_.x)
          rectangle(rgb_image_, mouse_start_, mouse_end_, Scalar(255, 0, 0), 3);

        if (init_requested_) {
          ROS_INFO("INPUT: tracker intialization requested");

          init_requested_ = false;
          tracker_initialized_ = true;
          ROS_INFO("Tracker initialized!");
        }

//        if (tracker_initialized_) {
//          auto begin = chrono::high_resolution_clock::now();
//          gpu_tracker.computeNext(rgb_image_);
//          auto end = chrono::high_resolution_clock::now();
//          auto time_span =
//              chrono::duration_cast<chrono::milliseconds>(end - begin).count();
//          stringstream ss;
//          ss << "Tracker run in ms: " << time_span << "";
//          ROS_INFO(ss.str().c_str());

//          Point2f p = gpu_tracker.getCentroid();
//          circle(rgb_image_, p, 5, Scalar(255, 0, 0), -1);
//          Scalar color(255, 0, 0);
//          drawBoundingBox(gpu_tracker.getBoundingBox(), color, rgb_image_);

//        }

        Mat depth_mapped;
        applyColorMap(depth_image_, depth_mapped);

        cout << "depth img " << depth_image_.rows << " " <<depth_image_.cols
             << "\n";
        cout << "map img " << depth_mapped.rows << " " <<depth_mapped.cols
             << "\n";

        imshow("Tracker", rgb_image_);
        imshow("Tracker_d", depth_mapped);

        img_updated_ = false;
        waitKey(1);
      }

      // r.sleep();
    }
}

} // end namespace


int main(int argc, char* argv[])
{
    ROS_INFO("Starting tracker input");
    ros::init(argc, argv, "pinot_tracker_node_3d");

    pinot_tracker::TrackerNode3D tracker_node;

    return 0;
}
