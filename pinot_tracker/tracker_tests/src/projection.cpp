#include "../include/projection.h"
#include "../../utilities/include/draw_functions.h"
#include "../../utilities/include/profiler.h"
#include "../../utilities/include/utilities.h"
#include <iostream>
#include <visualization_msgs/Marker.h>

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
    //    getCameraMatrix(camera_info_msg, params_.camera_matrix);
    params_.camera_model.fromCameraInfo(camera_info_msg);

    cout << "camera model: " << params_.camera_model.fx() << "\n";

    //    cout << params_.camera_matrix << endl;
    //    cout << params_.camera_model.projectionMatrix() << endl;

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

void Projection::publishPose(cv::Point3f &mean_point, cv::Point3f &min_point,
                             cv::Point3f &max_point) {
  tf::Vector3 centroid(mean_point.z, -mean_point.x, mean_point.y);

  tf::Transform transform;
  transform.setOrigin(centroid);
  transform.setRotation(tf::createIdentityQuaternion());

  transform_broadcaster_.sendTransform(tf::StampedTransform(
      transform, ros::Time::now(), "camera_rgb_frame", "mean_centroid"));

  visualization_msgs::Marker marker;
  marker.header.frame_id = "camera_rgb_frame";

  marker.header.stamp = ros::Time::now();
  marker.type = visualization_msgs::Marker::CUBE;
  marker.ns = "mean_cuboid";
  marker.id = 1;
  marker.action = visualization_msgs::Marker::ADD;

  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;

  marker.color.r = 1.0f;
  marker.color.g = 0.0f;
  marker.color.b = 0.0f;
  marker.color.a = 0.3;

  auto scale_x = max_point.x - min_point.x;
  auto scale_y = max_point.y - min_point.y;
  auto scale_z = std::min(scale_x, scale_y);

  marker.pose.position.x = mean_point.z;
  marker.pose.position.y = mean_point.x;
  marker.pose.position.z = mean_point.y;
  marker.scale.x = scale_z;
  marker.scale.y = scale_x;
  marker.scale.z = scale_y;

  markers_publisher_.publish(marker);
}

void Projection::publishPose(Point3f &center, std::vector<Point3f> &back_points,
                             std::vector<Point3f> &front_points) {
  tf::Vector3 centroid(center.z, -center.x, center.y);

  tf::Transform transform;
  transform.setOrigin(centroid);
  transform.setRotation(tf::createIdentityQuaternion());

  transform_broadcaster_.sendTransform(tf::StampedTransform(
      transform, ros::Time::now(), "camera_rgb_frame", "object_centroid"));

  visualization_msgs::Marker marker;
  marker.header.frame_id = "camera_rgb_frame";

  marker.header.stamp = ros::Time::now();
  marker.type = visualization_msgs::Marker::CUBE;
  marker.ns = "bounding_cuboid";
  marker.id = 1;
  marker.action = visualization_msgs::Marker::ADD;

  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;

  marker.color.r = 0.0f;
  marker.color.g = 1.0f;
  marker.color.b = 0.0f;
  marker.color.a = 0.3;

  auto scale_x = front_points.at(1).x - front_points.at(0).x;

  marker.pose.position.x = center.z;
  marker.pose.position.y = center.x + scale_x / 2.0f;
  marker.pose.position.z = center.y;
  marker.scale.x = back_points.at(0).z - front_points.at(0).z;
  marker.scale.y = front_points.at(1).x - front_points.at(0).x;
  marker.scale.z = front_points.at(2).y - front_points.at(0).y;

  markers_publisher_.publish(marker);
}

void Projection::run() {
  spinner_.start();

  ROS_INFO("INPUT: init tracker");

  params_.debug_path = "/home/alessandro/Debug";
  Tracker3D tracker;

  auto &profiler = Profiler::getInstance();
  Point3f mean_pt, min_pt, max_pt;

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
      if (!tracker_initialized_) imshow("Tracker", tmp);
      imshow("Tracker_d", depth_mapped);

      if (init_requested_ && !tracker_initialized_) {
        Mat3f points(depth_image_.rows, depth_image_.cols, cv::Vec3f(0, 0, 0));
        depthTo3d(depth_image_, params_.camera_model.cx(),
                  params_.camera_model.cy(), params_.camera_model.fx(),
                  params_.camera_model.fy(), points);

        Mat1b mask(depth_image_.rows, depth_image_.cols, uchar(0));
        rectangle(mask, mouse_start_, mouse_end_, uchar(255), -1);
        Mat3b debug_cloud(mask.rows, mask.cols, Vec3b(0, 0, 0));
        Mat3b debug_proj(mask.rows, mask.cols, Vec3b(0, 0, 0));
        for (auto i = 0; i < mask.rows; ++i) {
          for (auto j = 0; j < mask.cols; ++j) {
            Vec3f &val = points.at<Vec3f>(i, j);
            if (mask.at<uchar>(i, j) != 0) {
              if (val[2] != 0)
                debug_cloud.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
              else
                debug_cloud.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
            }

            if (val[2] != 0) debug_proj.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
          }
        }

        // points = cloud_image_.clone();

        imwrite("/home/alessandro/Debug/mask_debug.png", debug_cloud);
        imwrite("/home/alessandro/Debug/proj_debug.png", debug_proj);

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
        imshow("Tracker", out);

        analyzeCube(points, mouse_start_, mouse_end_, mean_pt, min_pt, max_pt);

        cout << "cube centroid " << tracker.getCurrentCentroid() << endl;

        waitKey(30);
      }

      else if (tracker_initialized_) {
        Point3f pt = tracker.getCurrentCentroid();
        auto front_points = tracker.getFrontBB();
        auto back_points = tracker.getBackBB();

        Mat3f points(depth_image_.rows, depth_image_.cols, cv::Vec3f(0, 0, 0));
        depthTo3d(depth_image_, params_.camera_model.cx(),
                  params_.camera_model.cy(), params_.camera_model.fx(),
                  params_.camera_model.fy(), points);

        // analyzeCube(points, mouse_start_, mouse_end_, mean_pt, min_pt,
        // max_pt);
        Mat out;
        publishPose(pt, back_points, front_points);
        // publishPose(mean_pt, min_pt, max_pt);
        profiler->start("frame_time");
        tracker.computeNext(rgb_image_, points, out);
        profiler->stop("frame_time");
        float elapsed = profiler->getTime("frame_time");
        cout << "Average time: " << elapsed << " ms \n";

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

        cout << "Pts size " << pts.size() << " votes " << votes.size() << endl;

        drawCentroidVotes(
            pts, votes,
            Point2f(params_.camera_model.cx(), params_.camera_model.cy()),
            true, params_.camera_model.fx(), out);

        imshow("Tracker", out);

        waitKey(30);
      }

      //      else if (tracker_initialized_) {
      //        //        cout << "Next Frame " << endl;
      //        Mat out;
      //        points =
      //            cv::Mat3f(depth_image_.rows, depth_image_.cols, cv::Vec3f(0,
      //            0, 0));
      //        disparityToDepth(depth_image_, params_.camera_model.cx(),
      //                         params_.camera_model.cy(),
      //                         params_.camera_model.fx(),
      //                         params_.camera_model.fy(), points);
      //        tracker.computeNext(rgb_image_, points, out);
      //        rgb_image_.copyTo(out);
      //        Point2f center;
      //        projectPoint(
      //            params_.camera_model.fx(),
      //            Point2f(params_.camera_model.cx(),
      //            params_.camera_model.cy()),
      //            tracker.getCurrentCentroid(), center);
      //        tracker.drawObjectLocation(out);
      //        circle(out, center, 5, Scalar(255, 0, 0), -1);
      //        imshow("Tracker", out);
      //      }

      char c = waitKey(30);

      if (c == 'r') {
        // tracker.clear(), init_requested_ = false;
        tracker_initialized_ = false;
      }
      if (c == 's') cout << "save" << endl;

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
