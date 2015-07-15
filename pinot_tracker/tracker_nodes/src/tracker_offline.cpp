#include "../include/tracker_offline.h"
#include "../../utilities/include/profiler.h"
#include "../../utilities/include/draw_functions.h"

using namespace std;
using namespace cv;

namespace pinot_tracker {

TrackerOffline::TrackerOffline()
    : nh_(),
      rgb_video_path_(),
      depth_video_path_(),
      top_left_pt_(),
      bottom_righ_pt_(),
      use_depth_(false),
      rgb_video_(),
      depth_video_(),
      spinner_(0),
      tracker_() {
  publisher_ = nh_.advertise<sensor_msgs::Image>("pinot_tracker/output", 1);

  init();
}

void TrackerOffline::init() {
  getParameters();
  if (rgb_video_path_.size() == 0) {
    ROS_INFO("No input video set in parameters!");
    return;
  }

  rgb_video_.open(rgb_video_path_);

  if (use_depth_) {
    depth_video_.open(depth_video_path_);
  }

  if (top_left_pt_ == bottom_righ_pt_) {
    ROS_INFO("Initial bounding box not defined!");
    return;
  }

  cvStartWindowThread();
  namedWindow("Init");
  Mat rgb_image, tmp;
  rgb_video_.read(rgb_image);
  rgb_image.copyTo(tmp);
  rectangle(tmp, top_left_pt_, bottom_righ_pt_, Scalar(255, 0, 0), 1);
  imshow("Init", tmp);
  waitKey(0);
  destroyWindow("Init");
  waitKey(1);

  tracker_.init(rgb_image, top_left_pt_, bottom_righ_pt_);

  start();
}

void TrackerOffline::start() {
  ros::Rate r(30);

  Mat rgb_image;

  auto& profiler = Profiler::getInstance();

  while (ros::ok()) {
    if (!rgb_video_.read(rgb_image)) break;

    profiler->start("frame");
    tracker_.computeNext(rgb_image);
    profiler->stop("frame");

    Point2f p = tracker_.getCentroid();
    circle(rgb_image, p, 5, Scalar(255, 0, 0), -1);
    vector<Point2f> bbox = tracker_.getBoundingBox();
    drawBoundingBox(bbox, Scalar(255, 0, 0), 2, rgb_image);

    cv_bridge::CvImage cv_img;
    cv_img.image = rgb_image;
    cv_img.encoding = sensor_msgs::image_encodings::BGR8;
    publisher_.publish(cv_img.toImageMsg());

    r.sleep();
  }

  stringstream ss;
  ss << "Average tracking time: " << profiler->getTime("frame");
  ROS_INFO(ss.str().c_str());
}

void TrackerOffline::getParameters() {
  stringstream ss;

  ss << "Tracker Input: \n";

  ss << "filter_border: ";
  if (!ros::param::get("pinot/tracker_2d/filter_border", params_.filter_border))
    ss << "failed \n";
  else
    ss << params_.filter_border << "\n";

  ss << "update_votes: ";
  if (!ros::param::get("pinot/tracker_2d/update_votes", params_.update_votes))
    ss << "failed \n";
  else
    ss << params_.update_votes << "\n";

  ss << "eps: ";
  if (!ros::param::get("pinot/tracker_2d/eps", params_.eps))
    ss << "failed \n";
  else
    ss << params_.eps << "\n";

  ss << "min_points: ";
  if (!ros::param::get("pinot/tracker_2d/min_points", params_.min_points))
    ss << "failed \n";
  else
    ss << params_.min_points << "\n";

  ss << "ransac_iterations: ";
  if (!ros::param::get("pinot/tracker_2d/ransac_iterations",
                       params_.ransac_iterations))
    ss << "failed \n";
  else
    ss << params_.ransac_iterations << "\n";

  int x, y, w, h;
  x = y = w = h = 0;

  ss << "box_x: ";
  if (!ros::param::get("pinot/offline/box_x", x))
    ss << "failed \n";
  else
    ss << x << "\n";

  ss << "box_y: ";
  if (!ros::param::get("pinot/offline/box_y", y))
    ss << "failed \n";
  else
    ss << y << "\n";

  ss << "box_w: ";
  if (!ros::param::get("pinot/offline/box_w", w))
    ss << "failed \n";
  else
    ss << w << "\n";

  ss << "box_h: ";
  if (!ros::param::get("pinot/offline/box_h", h))
    ss << "failed \n";
  else
    ss << h << "\n";

  top_left_pt_ = Point2f(x, y);
  bottom_righ_pt_ = Point2f(x + w, y + h);

  ss << "rgb_input: ";
  if (!ros::param::get("pinot/offline/rgb_input", rgb_video_path_))
    ss << "failed \n";
  else
    ss << rgb_video_path_ << "\n";

  ss << "depth_input: ";
  if (!ros::param::get("pinot/offline/depth_input", depth_video_path_))
    ss << "failed \n";
  else
    ss << depth_video_path_ << "\n";

  ROS_INFO(ss.str().c_str());
}

}  // end namespace

int main(int argc, char* argv[]) {
  ROS_INFO("Starting tracker in offline mode");
  ros::init(argc, argv, "pinot_tracker_offline_node");

  pinot_tracker::TrackerOffline offline;

  ros::shutdown();

  return 0;
}
