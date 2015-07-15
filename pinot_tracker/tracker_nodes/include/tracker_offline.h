#ifndef TRACKEROFFLINE_H
#define TRACKEROFFLINE_H

#include <string>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "../../tracker/include/params.h"
#include "../../tracker/include/tracker_2d_v2.h"

namespace pinot_tracker{

class TrackerOffline
{
public:
    TrackerOffline();

private:

    void init();

    void start();

    void getParameters();

    ros::NodeHandle nh_;
    ros::AsyncSpinner spinner_;
    ros::Publisher publisher_;

    cv::VideoCapture rgb_video_;
    cv::VideoCapture depth_video_;

    std::string rgb_video_path_;
    std::string depth_video_path_;

    TrackerParams params_;
    cv::Point2f top_left_pt_, bottom_righ_pt_;

    bool use_depth_;

    TrackerV2 tracker_;

};

}

#endif // TRACKEROFFLINE_H
