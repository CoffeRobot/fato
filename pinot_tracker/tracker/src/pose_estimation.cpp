#include "../include/pose_estimation.h"

namespace pinot_tracker
{

//    void estimate_pose()
//    {

//        vector<Point2f> objectPoints;
//        vector<Point2f> imagePoints;

//        Mat rvec, tvec;
//        vector<int> inliers_cpu;
//        if (objectPoints.size() > 4) {
//          solvePnPRansac(objectPoints, imagePoints, _camera_mat,
//                         Mat::zeros(1, 8, CV_32F), rvec, tvec, false,
//                         _num_iter_ransac, max_dist, objectPoints.size(),
//                         inliers_cpu, CV_P3P);
//          double T[] = { tvec.at<double>(0, 0), tvec.at<double>(0, 1),
//                         tvec.at<double>(0, 2) };
//          double R[] = { rvec.at<double>(0, 0), rvec.at<double>(0, 1),
//                         rvec.at<double>(0, 2) };
//          currPose = TranslationRotation3D(T, R);
//    }


} // end namespace
