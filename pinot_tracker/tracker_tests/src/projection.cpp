#include "../include/projection.h"
#include "../../utilities/include/draw_functions.h"
#include "../../utilities/include/profiler.h"
#include "../../utilities/include/utilities.h"
#include <iostream>

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
  Mat3f cloud(depth_msg->width, depth_msg->height, Vec3f(0, 0, 0));
  if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
    disparityToDepth<uint16_t>(depth_msg, params_.camera_model, cloud);
  } else if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
    disparityToDepth<float>(depth_msg, params_.camera_model, cloud);
  } else {
    ROS_WARN("Warning: unsupported depth image format!");
    // ROS_INFO("Mutex unlocked by cloud message");
    return;
  }

  cloud_image_ = cloud.clone();

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

void Projection::analyzeCube(cv::Mat &disparity, Point2d &top_left,
                             Point2d &bottom_right) {
  Mat1b mask(disparity.rows, disparity.cols, uchar(0));
  rectangle(mask, top_left, bottom_right, 255, -1);

  vector<float> depth_x, depth_y, depth_z;
  float average = 0;
  float counter = 0;
  for (int i = 0; i < disparity.rows; ++i) {
    for (int j = 0; j < disparity.cols; ++j) {
      if (mask.at<uchar>(i, j) == 255 && disparity.at<Vec3f>(i, j)[2] != 0) {
        depth_z.push_back(disparity.at<Vec3f>(i, j)[2]);
        depth_x.push_back(disparity.at<Vec3f>(i, j)[0]);
        depth_y.push_back(disparity.at<Vec3f>(i, j)[1]);
        average += disparity.at<Vec3f>(i, j)[2];
        counter++;
      }
    }
  }

  sort(depth_x.begin(), depth_x.end());
  sort(depth_y.begin(), depth_y.end());
  sort(depth_z.begin(), depth_z.end());

  auto size = depth_x.size();

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

  cout << "depth: avg " << average / counter << " median x " << median_x
       << " median y " << median_y << " median z " << median_z << endl;
}

void Projection::run() {
  spinner_.start();

  ROS_INFO("INPUT: init tracker");

  params_.debug_path = "/home/alessandro/Debug";
  Tracker3D tracker;

  auto &profiler = Profiler::getInstance();

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
        ROS_INFO("before disparity to cloud depth");
        Mat3f points(depth_image_.rows, depth_image_.cols, cv::Vec3f(0, 0, 0));
        ROS_INFO("before disparity to cloud depth");
        disparityToDepth(depth_image_, params_.camera_model.cx(),
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
          ROS_WARN("Tracker correctly intialized!");
        }
        init_requested_ = false;
        tracker_initialized_ = true;
        ROS_INFO("Tracker initialized!");
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

//        ofstream file("/home/alessandro/Debug/disparity.txt");
//        for (auto i = 0; i < points.rows; ++i) {
//          for (auto j = 0; j < points.cols; ++j) {
//            file << points.at<Vec3f>(i, j) << " "
//                 << cloud_image_.at<Vec3f>(i, j) << "\n";
//          }
//        }
//        file.close();

//        ROS_INFO("before getting mean depth");
//        // analyzeCube(points, mouse_start_, mouse_end_);
//        analyzeCube(cloud_image_, mouse_start_, mouse_end_);
//        ROS_INFO("after getting mean depth");

        waitKey(30);
      }

      else if (tracker_initialized_) {


        Point3f pt = tracker.getCurrentCentroid();
        tf::Vector3 centroid(pt.z, -pt.x, pt.y);

        tf::Transform transform;
        transform.setOrigin(centroid);
        transform.setRotation(tf::createIdentityQuaternion());

        transform_broadcaster_.sendTransform(tf::StampedTransform(
            transform, ros::Time::now(), "camera_rgb_frame", "object_centroid"));

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
      //        //        projectPoint(
      //        //            params_.camera_model.fx(),
      //        //            Point2f(params_.camera_model.cx(),
      //        //            params_.camera_model.cy()),
      //        //            tracker.getCurrentCentroid(), center);
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

      //      stringstream ss;
      //      cout << "cloud " << cloud.rows << " " << cloud.cols << endl;

      //      for(auto i = 0; i < cloud.rows; ++i)
      //      {
      //          for(auto j = 0; j < cloud.cols; ++j)
      //          {
      //              ss << cloud.at<Vec3f>(i,j) << " ";
      //          }
      //          ss << "\n";
      //      }

      //      ofstream file("/home/alessandro/Downloads/debug.txt");
      //      file << ss.str();
      //      file.close();
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
