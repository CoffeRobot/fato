#include "../include/tracker_node.h"  // ugly include for qtcreator

using namespace std;

namespace pinot_tracker {

TrackerNode::TrackerNode() {}

void TrackerNode::readImage(const sensor_msgs::Image::ConstPtr msgImage,
                            cv::Mat &image) const {
 cv_bridge::CvImageConstPtr pCvImage;
 pCvImage = cv_bridge::toCvShare(msgImage, msgImage->encoding);
 pCvImage->image.copyTo(image);
}

}
