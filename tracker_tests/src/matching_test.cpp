#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../../tracker/include/feature_matcher.hpp"
#include "../../tracker/include/matcher.h"
#include "../../utilities/include/utilities.h"
#include "../../tracker/include/config.h"

#include <hdf5_file.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <image_geometry/pinhole_camera_model.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <iostream>
#include <chrono>

#include "AKAZE.h"
#include "AKAZEConfig.h"

// ORB settings
int ORB_MAX_KPTS = 1500;
float ORB_SCALE_FACTOR = 1.2;
int ORB_PYRAMID_LEVELS = 4;
float ORB_EDGE_THRESHOLD = 31.0;
int ORB_FIRST_PYRAMID_LEVEL = 0;
int ORB_WTA_K = 2;
int ORB_PATCH_SIZE = 31;

// BRISK settings
int BRISK_HTHRES = 10;
int BRISK_NOCTAVES = 4;

// Some image matching options
float MIN_H_ERROR = 2.50f;  // Maximum error in pixels to accept an inlier
float DRATIO = 0.80f;       // NNDR Matching value

using namespace cv;
using namespace std;
using namespace fato;
using namespace std::chrono;
using namespace libAKAZECU;

typedef high_resolution_clock my_clock;

ros::Publisher image_publisher;
bool image_updated = false;
Mat rgb_in;

void cudaAkaze() {
  // Variables
  AKAZEOptions options;
  cv::Mat img1, img1_32, img2, img2_32;
  string img_path1, img_path2, homography_path;
  double t1 = 0.0, t2 = 0.0;

  // ORB variables
  vector<cv::KeyPoint> kpts1_orb, kpts2_orb;
  vector<cv::Point2f> matches_orb, inliers_orb;
  vector<vector<cv::DMatch>> dmatches_orb;
  cv::Mat desc1_orb, desc2_orb;
  int nmatches_orb = 0, ninliers_orb = 0, noutliers_orb = 0;
  int nkpts1_orb = 0, nkpts2_orb = 0;
  float ratio_orb = 0.0;
  double torb = 0.0;

  // BRISK variables
  vector<cv::KeyPoint> kpts1_brisk, kpts2_brisk;
  vector<cv::Point2f> matches_brisk, inliers_brisk;
  vector<vector<cv::DMatch>> dmatches_brisk;
  cv::Mat desc1_brisk, desc2_brisk;
  int nmatches_brisk = 0, ninliers_brisk = 0, noutliers_brisk = 0;
  int nkpts1_brisk = 0, nkpts2_brisk = 0;
  float ratio_brisk = 0.0;
  double tbrisk = 0.0;

  // AKAZE variables
  vector<cv::KeyPoint> kpts1_akaze, kpts2_akaze;
  vector<cv::Point2f> matches_akaze, inliers_akaze, cuda_matches_akaze,
      cuda_inliers;
  vector<vector<cv::DMatch>> dmatches_akaze;
  cv::Mat desc1_akaze, desc2_akaze;
  int nmatches_akaze = 0, ninliers_akaze = 0, noutliers_akaze = 0;
  int nkpts1_akaze = 0, nkpts2_akaze = 0;
  float ratio_akaze = 0.0;
  double takaze = 0.0;

  // Create the L2 and L1 matchers
  cv::Ptr<cv::DescriptorMatcher> matcher_l2 =
      cv::DescriptorMatcher::create("BruteForce");
  cv::Ptr<cv::DescriptorMatcher> matcher_l1 =
      cv::DescriptorMatcher::create("BruteForce-Hamming");
  cv::Mat HG;

  // Read the image, force to be grey scale
  img1 = cv::imread(
      "/home/alessandro/projects/drone_ws/src/fato_tracker/data/imgs/hydro.png",
      0);

  if (img1.data == NULL) {
    cerr << "Error loading image: " << img_path1 << endl;
    return;
  }

  // Read the image, force to be grey scale
  img2 = cv::imread(
      "/home/alessandro/projects/drone_ws/src/fato_tracker/data/imgs/cam.png",
      0);

  if (img2.data == NULL) {
    cerr << "Error loading image: " << img_path2 << endl;
    return;
  }

  // Convert the images to float
  img1.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);
  img2.convertTo(img2_32, CV_32F, 1.0 / 255.0, 0);

  // Color images for results visualization
  cv::Mat img1_rgb_orb = cv::Mat(cv::Size(img1.cols, img1.rows), CV_8UC3);
  cv::Mat img2_rgb_orb = cv::Mat(cv::Size(img2.cols, img1.rows), CV_8UC3);
  cv::Mat img_com_orb = cv::Mat(cv::Size(img1.cols * 2, img1.rows), CV_8UC3);

  cv::Mat img1_rgb_brisk = cv::Mat(cv::Size(img1.cols, img1.rows), CV_8UC3);
  cv::Mat img2_rgb_brisk = cv::Mat(cv::Size(img2.cols, img1.rows), CV_8UC3);
  cv::Mat img_com_brisk = cv::Mat(cv::Size(img1.cols * 2, img1.rows), CV_8UC3);

  cv::Mat img1_rgb_akaze = cv::Mat(cv::Size(img1.cols, img1.rows), CV_8UC3);
  cv::Mat img2_rgb_akaze = cv::Mat(cv::Size(img2.cols, img1.rows), CV_8UC3);
  cv::Mat img_com_akaze = cv::Mat(cv::Size(img1.cols * 2, img1.rows), CV_8UC3);

  // Read the homography file
  bool use_ransac = false;
  if (read_homography(homography_path, HG) == false) use_ransac = true;

/* ************************************************************************* */

// ORB Features
//*****************
#if CV_VERSION_EPOCH == 2
  cv::ORB orb(ORB_MAX_KPTS, ORB_SCALE_FACTOR, ORB_PYRAMID_LEVELS,
              ORB_EDGE_THRESHOLD, ORB_FIRST_PYRAMID_LEVEL, ORB_WTA_K,
              ORB_PATCH_SIZE);
#else
  cv::Ptr<cv::ORB> orb = cv::ORB::create(
      ORB_MAX_KPTS, ORB_SCALE_FACTOR, ORB_PYRAMID_LEVELS, ORB_EDGE_THRESHOLD,
      ORB_FIRST_PYRAMID_LEVEL, ORB_WTA_K, ORB_PATCH_SIZE);
#endif

  t1 = cv::getTickCount();

#if CV_VERSION_EPOCH == 2
  orb(img1, cv::noArray(), kpts1_orb, desc1_orb, false);
  orb(img2, cv::noArray(), kpts2_orb, desc2_orb, false);
#else
  orb->detectAndCompute(img1, cv::noArray(), kpts1_orb, desc1_orb, false);
  orb->detectAndCompute(img2, cv::noArray(), kpts2_orb, desc2_orb, false);
#endif

  matcher_l1->knnMatch(desc1_orb, desc2_orb, dmatches_orb, 2);
  matches2points_nndr(kpts1_orb, kpts2_orb, dmatches_orb, matches_orb, DRATIO);

  if (use_ransac == false)
    compute_inliers_homography(matches_orb, inliers_orb, HG, MIN_H_ERROR);
  else
    compute_inliers_ransac(matches_orb, inliers_orb, MIN_H_ERROR, false);

  nkpts1_orb = kpts1_orb.size();
  nkpts2_orb = kpts2_orb.size();
  nmatches_orb = matches_orb.size() / 2;
  ninliers_orb = inliers_orb.size() / 2;
  noutliers_orb = nmatches_orb - ninliers_orb;
  ratio_orb = 100.0 * (float)(ninliers_orb) / (float)(nmatches_orb);

  t2 = cv::getTickCount();
  torb = 1000.0 * (t2 - t1) / cv::getTickFrequency();

  cvtColor(img1, img1_rgb_orb, cv::COLOR_GRAY2BGR);
  cvtColor(img2, img2_rgb_orb, cv::COLOR_GRAY2BGR);

  draw_keypoints(img1_rgb_orb, kpts1_orb);
  draw_keypoints(img2_rgb_orb, kpts2_orb);
  draw_inliers(img1_rgb_orb, img2_rgb_orb, img_com_orb, inliers_orb, 0);

  cout << "ORB Results" << endl;
  cout << "**************************************" << endl;
  cout << "Number of Keypoints Image 1: " << nkpts1_orb << endl;
  cout << "Number of Keypoints Image 2: " << nkpts2_orb << endl;
  cout << "Number of Matches: " << nmatches_orb << endl;
  cout << "Number of Inliers: " << ninliers_orb << endl;
  cout << "Number of Outliers: " << noutliers_orb << endl;
  cout << "Inliers Ratio: " << ratio_orb << endl;
  cout << "ORB Features Extraction Time (ms): " << torb << endl;
  cout << endl;

/* ************************************************************************* */

// BRISK Features
//*****************
#if CV_VERSION_EPOCH == 2
  cv::BRISK brisk(BRISK_HTHRES, BRISK_NOCTAVES, 1.0f);
#else
  cv::Ptr<cv::BRISK> brisk =
      cv::BRISK::create(BRISK_HTHRES, BRISK_NOCTAVES, 1.0f);
#endif

  t1 = cv::getTickCount();

#if CV_VERSION_EPOCH == 2
  brisk(img1, cv::noArray(), kpts1_brisk, desc1_brisk, false);
  brisk(img2, cv::noArray(), kpts2_brisk, desc2_brisk, false);
#else
  brisk->detectAndCompute(img1, cv::noArray(), kpts1_brisk, desc1_brisk, false);
  brisk->detectAndCompute(img2, cv::noArray(), kpts2_brisk, desc2_brisk, false);
#endif

  matcher_l1->knnMatch(desc1_brisk, desc2_brisk, dmatches_brisk, 2);
  matches2points_nndr(kpts1_brisk, kpts2_brisk, dmatches_brisk, matches_brisk,
                      DRATIO);

  if (use_ransac == false)
    compute_inliers_homography(matches_brisk, inliers_brisk, HG, MIN_H_ERROR);
  else
    compute_inliers_ransac(matches_brisk, inliers_brisk, MIN_H_ERROR, false);

  nkpts1_brisk = kpts1_brisk.size();
  nkpts2_brisk = kpts2_brisk.size();
  nmatches_brisk = matches_brisk.size() / 2;
  ninliers_brisk = inliers_brisk.size() / 2;
  noutliers_brisk = nmatches_brisk - ninliers_brisk;
  ratio_brisk = 100.0 * (float)(ninliers_brisk) / (float)(nmatches_brisk);

  t2 = cv::getTickCount();
  tbrisk = 1000.0 * (t2 - t1) / cv::getTickFrequency();

  cvtColor(img1, img1_rgb_brisk, cv::COLOR_GRAY2BGR);
  cvtColor(img2, img2_rgb_brisk, cv::COLOR_GRAY2BGR);

  draw_keypoints(img1_rgb_brisk, kpts1_brisk);
  draw_keypoints(img2_rgb_brisk, kpts2_brisk);
  draw_inliers(img1_rgb_brisk, img2_rgb_brisk, img_com_brisk, inliers_brisk, 1);

  cout << "BRISK Results" << endl;
  cout << "**************************************" << endl;
  cout << "Number of Keypoints Image 1: " << nkpts1_brisk << endl;
  cout << "Number of Keypoints Image 2: " << nkpts2_brisk << endl;
  cout << "Number of Matches: " << nmatches_brisk << endl;
  cout << "Number of Inliers: " << ninliers_brisk << endl;
  cout << "Number of Outliers: " << noutliers_brisk << endl;
  cout << "Inliers Ratio: " << ratio_brisk << endl;
  cout << "BRISK Features Extraction Time (ms): " << tbrisk << endl;
  cout << endl;

  /* *************************************************************************
   */
  // A-KAZE Features
  //*******************
  options.img_width = img1.cols;
  options.img_height = img1.rows;
  libAKAZECU::AKAZE evolution1(options);

  options.img_width = img2.cols;
  options.img_height = img2.rows;
  libAKAZECU::AKAZE evolution2(options);

  t1 = cv::getTickCount();

  evolution1.Create_Nonlinear_Scale_Space(img1_32);
  evolution1.Feature_Detection(kpts1_akaze);
  evolution1.Compute_Descriptors(kpts1_akaze, desc1_akaze);

  evolution2.Create_Nonlinear_Scale_Space(img2_32);
  evolution2.Feature_Detection(kpts2_akaze);
  evolution2.Compute_Descriptors(kpts2_akaze, desc2_akaze);

  if (options.descriptor < MLDB_UPRIGHT)
    matcher_l2->knnMatch(desc1_akaze, desc2_akaze, dmatches_akaze, 2);

  // Binary descriptor, use Hamming distance
  else
    matcher_l1->knnMatch(desc1_akaze, desc2_akaze, dmatches_akaze, 2);

  matches2points_nndr(kpts1_akaze, kpts2_akaze, dmatches_akaze, matches_akaze,
                      DRATIO);

  if (use_ransac == false)
    compute_inliers_homography(matches_akaze, inliers_akaze, HG, MIN_H_ERROR);
  else
    compute_inliers_ransac(matches_akaze, inliers_akaze, MIN_H_ERROR, false);

  t2 = cv::getTickCount();
  takaze = 1000.0 * (t2 - t1) / cv::getTickFrequency();

  nkpts1_akaze = kpts1_akaze.size();
  nkpts2_akaze = kpts2_akaze.size();
  nmatches_akaze = matches_akaze.size() / 2;
  ninliers_akaze = inliers_akaze.size() / 2;
  noutliers_akaze = nmatches_akaze - ninliers_akaze;
  ratio_akaze = 100.0 * ((float)ninliers_akaze / (float)nmatches_akaze);

  cvtColor(img1, img1_rgb_akaze, cv::COLOR_GRAY2BGR);
  cvtColor(img2, img2_rgb_akaze, cv::COLOR_GRAY2BGR);

  draw_keypoints(img1_rgb_akaze, kpts1_akaze);
  draw_keypoints(img2_rgb_akaze, kpts2_akaze);
  draw_inliers(img1_rgb_akaze, img2_rgb_akaze, img_com_akaze, inliers_akaze, 2);

  // AKAZE CUDA RESULTS
  std::vector<std::vector<cv::DMatch>> cuda_dmatches;
  MatchDescriptors(desc1_akaze, desc2_akaze, kpts1_akaze.size(), cuda_dmatches);

  matches2points_nndr(kpts1_akaze, kpts2_akaze, cuda_dmatches,
                      cuda_matches_akaze, DRATIO);

  if (use_ransac == false)
    compute_inliers_homography(cuda_matches_akaze, cuda_inliers, HG,
                               MIN_H_ERROR);
  else
    compute_inliers_ransac(cuda_matches_akaze, cuda_inliers, MIN_H_ERROR,
                           false);

  float cuda_nmatches_akaze = cuda_matches_akaze.size() / 2;
  float cuda_ninliers_akaze = cuda_inliers.size() / 2;
  float cuda_noutliers_akaze = cuda_nmatches_akaze - cuda_ninliers_akaze;
  float cuda_ratio_akaze =
      100.0 * ((float)cuda_ninliers_akaze / (float)cuda_nmatches_akaze);

  cout << "A-KAZE Results" << endl;
  cout << "**************************************" << endl;
  cout << "Number of Keypoints Image 1: " << nkpts1_akaze << endl;
  cout << "Number of Keypoints Image 2: " << nkpts2_akaze << endl;
  cout << "Number of Matches: " << nmatches_akaze << " cuda "
       << cuda_nmatches_akaze << endl;
  cout << "Number of Inliers: " << ninliers_akaze << " cuda "
       << cuda_ninliers_akaze << endl;
  cout << "Number of Outliers: " << noutliers_akaze << " cuda "
       << cuda_noutliers_akaze << endl;
  cout << "Inliers Ratio: " << ratio_akaze << " cuda " << cuda_ratio_akaze
       << endl;
  cout << "A-KAZE Features Extraction Time (ms): " << takaze << endl;
  cout << endl;  // Show the images with the inliers

  Mat img1_rgb_akazec, img2_rgb_akazec;
  cvtColor(img1, img1_rgb_akazec, cv::COLOR_GRAY2BGR);
  cvtColor(img2, img2_rgb_akazec, cv::COLOR_GRAY2BGR);

  std::vector<KeyPoint> fst_kps, scd_kps;
  Mat fst_dsc, scd_dsc;
  std::vector<std::vector<cv::DMatch>> akazec_matches;

  AkazeMatcher akaze_matcher;
  akaze_matcher.init(img1_rgb_akazec.cols, img1_rgb_akazec.rows);
  akaze_matcher.extractTarget(img1);
  fst_kps = akaze_matcher.getTargetPoints();
  akaze_matcher.match(img2, scd_kps, scd_dsc, akazec_matches);

  vector<cv::Point2f> matches_akazec, inliers_akaze_c;

  matches2points_nndr(fst_kps, scd_kps, akazec_matches, matches_akazec, DRATIO);

  if (use_ransac == false)
    compute_inliers_homography(matches_akazec, inliers_akaze_c, HG,
                               MIN_H_ERROR);
  else
    compute_inliers_ransac(matches_akazec, inliers_akaze_c, MIN_H_ERROR, false);

  nkpts1_akaze = fst_kps.size();
  nkpts2_akaze = scd_kps.size();
  nmatches_akaze = matches_akazec.size() / 2;
  ninliers_akaze = inliers_akaze_c.size() / 2;
  noutliers_akaze = nmatches_akaze - ninliers_akaze;
  ratio_akaze = 100.0 * ((float)ninliers_akaze / (float)nmatches_akaze);

  cv::Mat out = cv::Mat(cv::Size(img1.cols * 2, img1.rows), CV_8UC3);
  draw_keypoints(img1_rgb_akazec, fst_kps);
  draw_keypoints(img2_rgb_akazec, scd_kps);
  draw_inliers(img1_rgb_akazec, img2_rgb_akazec, out, inliers_akaze_c, 2);

  cout << "A-KAZE Matcher Results" << endl;
  cout << "**************************************" << endl;
  cout << "Number of Keypoints Image 1: " << nkpts1_akaze << endl;
  cout << "Number of Keypoints Image 2: " << nkpts2_akaze << endl;
  cout << "Number of Matches: " << nmatches_akaze << " cuda "
       << cuda_nmatches_akaze << endl;
  cout << "Number of Inliers: " << ninliers_akaze << " cuda "
       << cuda_ninliers_akaze << endl;
  cout << "Number of Outliers: " << noutliers_akaze << " cuda "
       << cuda_noutliers_akaze << endl;
  cout << "Inliers Ratio: " << ratio_akaze << " cuda " << cuda_ratio_akaze
       << endl;
  cout << "A-KAZE Features Extraction Time (ms): " << takaze << endl;
  cout << endl;  // Show the images with the inliers

  cv::imshow("ORB", img_com_orb);
  cv::imshow("BRISK", img_com_brisk);
  cv::imshow("A-KAZE", img_com_akaze);
  cv::imshow("A-KAZEM", out);
  cv::waitKey(0);
}

void compareAkazeImp() {
  Mat fst = imread(
      "/home/alessandro/projects/drone_ws/src/fato_tracker/data/imgs/hydro.png",
      0);
  Mat scd = imread(
      "/home/alessandro/projects/drone_ws/src/fato_tracker/data/imgs/cam.png",
      0);

  cv::Mat img_com_lib = cv::Mat(cv::Size(fst.cols * 2, fst.rows), CV_8UC3);
  cv::Mat img_com_match = cv::Mat(cv::Size(fst.cols * 2, fst.rows), CV_8UC3);
  cv::Mat img_com_match_cuda =
      cv::Mat(cv::Size(fst.cols * 2, fst.rows), CV_8UC3);
  cv::Mat img_com_match_cuda2 =
      cv::Mat(cv::Size(fst.cols * 2, fst.rows), CV_8UC3);

  Size sz1 = fst.size();
  Size sz2 = scd.size();
  Mat akaze_rgb1, akaze_rgb2, matcher_rgb1, matcher_rgb2, cuda_matcher_rgb1,
      cuda_matcher_rgb2, cuda2_matcher_rgb1, cuda2_matcher_rgb2, out_rgb1,
      out_rgb2;

  cvtColor(fst, akaze_rgb1, CV_GRAY2BGR);
  cvtColor(scd, akaze_rgb2, CV_GRAY2BGR);
  akaze_rgb1.copyTo(matcher_rgb1);
  akaze_rgb1.copyTo(cuda_matcher_rgb1);
  akaze_rgb1.copyTo(cuda2_matcher_rgb1);
  akaze_rgb1.copyTo(out_rgb1);
  akaze_rgb2.copyTo(matcher_rgb2);
  akaze_rgb2.copyTo(cuda_matcher_rgb2);
  akaze_rgb2.copyTo(cuda2_matcher_rgb2);
  akaze_rgb2.copyTo(out_rgb2);

  cv::Ptr<cv::DescriptorMatcher> matcher_l1 =
      cv::DescriptorMatcher::create("BruteForce-Hamming");

  /*************************************************************************/
  /*   AKAZE LIB INTERFACE                                                 */
  /*************************************************************************/
  vector<cv::KeyPoint> lib_kps1, lib_kps2;
  Mat lib_dsc1, lib_dsc2;
  vector<vector<DMatch>> lib_matches;
  vector<cv::Point2f> lib_matched_points, lib_inliers;

  AKAZEOptions options;
  options.img_width = akaze_rgb1.cols;
  options.img_height = akaze_rgb1.rows;
  libAKAZECU::AKAZE evolution1(options);

  options.img_width = akaze_rgb1.cols;
  options.img_height = akaze_rgb1.rows;
  libAKAZECU::AKAZE evolution2(options);

  Mat img1_32, img2_32;
  fst.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);
  scd.convertTo(img2_32, CV_32F, 1.0 / 255.0, 0);

  evolution1.Create_Nonlinear_Scale_Space(img1_32);
  evolution1.Feature_Detection(lib_kps1);
  evolution1.Compute_Descriptors(lib_kps1, lib_dsc1);

  evolution2.Create_Nonlinear_Scale_Space(img2_32);
  evolution2.Feature_Detection(lib_kps2);
  evolution2.Compute_Descriptors(lib_kps2, lib_dsc2);

  matcher_l1->knnMatch(lib_dsc1, lib_dsc2, lib_matches, 2);

  matches2points_nndr(lib_kps1, lib_kps2, lib_matches, lib_matched_points,
                      DRATIO);

  compute_inliers_ransac(lib_matched_points, lib_inliers, MIN_H_ERROR, false);

  draw_keypoints(akaze_rgb1, lib_kps1);
  draw_keypoints(akaze_rgb2, lib_kps2);
  draw_inliers(akaze_rgb1, akaze_rgb2, img_com_lib, lib_inliers, 1);
  /*************************************************************************/
  /*   TRACKER INTERFACE - CPU MATCHER                                      */
  /*************************************************************************/
  vector<cv::KeyPoint> tr_kps1, tr_kps2;
  Mat tr_dsc1, tr_dsc2;
  vector<vector<DMatch>> tr_matches;
  vector<cv::Point2f> tr_matched_points, tr_inliers;

  AkazeMatcher akaze_matcher;
  akaze_matcher.init(matcher_rgb1.cols, matcher_rgb2.rows);
  akaze_matcher.extract(fst, tr_kps1, tr_dsc1);
  akaze_matcher.extract(scd, tr_kps2, tr_dsc2);
  matcher_l1->knnMatch(tr_dsc1, tr_dsc2, tr_matches, 2);
  matches2points_nndr(tr_kps1, tr_kps2, tr_matches, tr_matched_points, DRATIO);
  compute_inliers_ransac(tr_matched_points, tr_inliers, MIN_H_ERROR, false);

  draw_keypoints(matcher_rgb1, tr_kps1);
  draw_keypoints(matcher_rgb2, tr_kps2);
  draw_inliers(matcher_rgb1, matcher_rgb2, img_com_match, tr_inliers, 1);
  /*************************************************************************/
  /*   TRACKER INTERFACE - GPU MATCHER                                      */
  /*************************************************************************/
  vector<cv::KeyPoint> gpu_kps1, gpu_kps2;
  Mat gpu_dsc1, gpu_dsc2;
  vector<vector<DMatch>> gpu_matches;
  vector<cv::Point2f> gpu_matched_points, gpu_inliers;

  akaze_matcher.extractTarget(fst);
  gpu_kps1 = akaze_matcher.getTargetPoints();
  gpu_dsc1 = akaze_matcher.getTargetDescriptors();
  akaze_matcher.match(scd, gpu_kps2, gpu_dsc2, gpu_matches);

  matches2points_nndr(gpu_kps2, gpu_kps1, gpu_matches, gpu_matched_points,
                      DRATIO);
  compute_inliers_ransac(gpu_matched_points, gpu_inliers, MIN_H_ERROR, false);

  draw_keypoints(cuda_matcher_rgb1, gpu_kps1);
  draw_keypoints(cuda_matcher_rgb2, gpu_kps2);
  draw_inliers(cuda_matcher_rgb2, cuda_matcher_rgb1, img_com_match_cuda,
               gpu_inliers, 1);
  /*************************************************************************/
  /*   TRACKER INTERFACE - GPU MATCHER  - UPLOADING TARGET                 */
  /*************************************************************************/
  vector<cv::KeyPoint> gpu2_kps1, gpu2_kps2;
  Mat gpu2_dsc1, gpu2_dsc2;
  vector<vector<DMatch>> gpu2_matches;
  vector<cv::Point2f> gpu2_matched_points, gpu2_inliers;

  akaze_matcher.setTarget(gpu_dsc1);
  gpu2_dsc1 = akaze_matcher.getTargetDescriptors();
  gpu2_kps1 = gpu_kps1;

  akaze_matcher.match(scd, gpu2_kps2, gpu2_dsc2, gpu2_matches);

  matches2points_nndr(gpu2_kps2, gpu2_kps1, gpu2_matches, gpu2_matched_points,
                      DRATIO);
  compute_inliers_ransac(gpu2_matched_points, gpu2_inliers, MIN_H_ERROR, false);

  draw_keypoints(cuda2_matcher_rgb1, gpu2_kps1);
  draw_keypoints(cuda2_matcher_rgb2, gpu2_kps2);
  draw_inliers(cuda2_matcher_rgb2, cuda2_matcher_rgb1, img_com_match_cuda2,
               gpu2_inliers, 1);

  /*************************************************************************/
  /*   STATS                                                               */
  /*************************************************************************/
  Mat out(out_rgb1.rows, 2 * out_rgb1.cols, CV_8UC3);
  {
    out_rgb1.copyTo(out(Rect(0, 0, sz1.width, sz1.height)));
    out_rgb2.copyTo(out(Rect(sz1.width, 0, sz1.width, sz1.height)));

    Scalar color;

    int ratio_count = 0;
    int confidence_count = 0;
    int valid_count = 0;
    for (auto i = 0; i < gpu2_matches.size(); ++i) {
      DMatch best = gpu2_matches.at(i)[0];
      DMatch second = gpu2_matches.at(i)[1];

      Point2f tr = gpu2_kps1[best.trainIdx].pt;
      Point2f qr = gpu2_kps2[best.queryIdx].pt;
      qr.x += out_rgb1.cols;
      float confidence = 1 - (best.distance / akaze_matcher.maxDistance());
      float ratio = best.distance / second.distance;

      if (confidence < 0.8) {
        confidence_count++;
      } else if (ratio > 0.8) {
        ratio_count++;
      } else {
        valid_count++;

        line(out, tr, qr, Scalar(0, 255, 0), 1);
      }
      circle(out, tr, 3, Scalar(0, 255, 0), 1);
      circle(out, qr, 3, Scalar(0, 255, 0), 1);
    }

    cout << "valid " << valid_count << " confidence " << confidence_count
         << " ratio " << ratio_count << endl;
  }
  /*******************************************************************/
  /*          LOADING DESCRIPTORS                                    */
  /*******************************************************************/
  util::HDF5File out_file(
      "/home/alessandro/projects/drone_ws/src/fato_tracker/data/models/"
      "ros_hydro/ros_hydro_features.h5");
  std::vector<uchar> descriptors;
  std::vector<int> dscs_size, point_size;
  std::vector<float> points;
  // int test_feature_size;
  out_file.readArray<uchar>("descriptors", descriptors, dscs_size);
  out_file.readArray<float>("positions", points, point_size);

  cout << "extracted descriptor size " << endl;
  cout << gpu2_dsc2.rows << " " << gpu2_dsc2.cols << endl;

  cout << "descriptors size " << endl;
  for (auto val : dscs_size) cout << val << " ";
  cout << '\n';

  Mat init_descriptors;
  vectorToMat(descriptors, dscs_size, init_descriptors);

  akaze_matcher.setTarget(init_descriptors);
  gpu2_dsc1 = akaze_matcher.getTargetDescriptors();
  gpu2_kps1 = akaze_matcher.getTargetPoints();
  akaze_matcher.match(scd, gpu2_kps2, gpu2_dsc2, gpu2_matches);

  /*************************************************************************/
  /*   STATS                                                               */
  /*************************************************************************/
  // Size sz1 = out_rgb1.size;
  Mat out2(out_rgb1.rows, 2 * out_rgb1.cols, CV_8UC3);
  out_rgb1.copyTo(out2(Rect(0, 0, sz1.width, sz1.height)));
  out_rgb2.copyTo(out2(Rect(sz1.width, 0, sz1.width, sz1.height)));

  Scalar color;

  int ratio_count = 0;
  int confidence_count = 0;
  int valid_count = 0;
  for (auto i = 0; i < gpu2_matches.size(); ++i) {
    DMatch best = gpu2_matches.at(i)[0];
    DMatch second = gpu2_matches.at(i)[1];

    Point2f tr = gpu2_kps1[best.trainIdx].pt;
    Point2f qr = gpu2_kps2[best.queryIdx].pt;
    qr.x += out_rgb1.cols;
    float confidence = 1 - (best.distance / akaze_matcher.maxDistance());
    float ratio = best.distance / second.distance;

    if (confidence < 0.8) {
      confidence_count++;
    } else if (ratio > 0.8) {
      ratio_count++;
    } else {
      valid_count++;

      line(out2, tr, qr, Scalar(0, 255, 0), 1);
    }
    circle(out2, tr, 3, Scalar(0, 255, 0), 1);
    circle(out2, qr, 3, Scalar(0, 255, 0), 1);
  }

  cout << "valid " << valid_count << " confidence " << confidence_count
       << " ratio " << ratio_count << endl;

  imshow("akaze library", img_com_lib);
  imshow("matcher cpu", img_com_match);
  imshow("matcher gpu", img_com_match_cuda);
  imshow("matcher cuda2", img_com_match_cuda2);
  imshow("matching", out);
  imshow("matching h5", out2);
  waitKey(0);
}

void testWritingReadingDescriptors() {
  Mat fst = imread(
      "/home/alessandro/projects/drone_ws/src/fato_tracker/data/imgs/hydro.png",
      0);
  Mat scd = imread(
      "/home/alessandro/projects/drone_ws/src/fato_tracker/data/imgs/cam.png",
      0);

  //  Mat fst_gray, scd_gray;
  //  cvtColor(fst, fst_gray, CV_BGR2GRAY);

  AkazeMatcher akaze_matcher;
  akaze_matcher.init(fst.cols, fst.rows);

  akaze_matcher.extractTarget(fst);
  std::vector<cv::KeyPoint>& points = akaze_matcher.getTargetPoints();
  cv::Mat& dscs = akaze_matcher.getTargetDescriptors();

  std::vector<uchar> all_descriptors;
  for (int i = 0; i < dscs.rows; ++i) {
    all_descriptors.insert(all_descriptors.end(), dscs.ptr<uchar>(i),
                           dscs.ptr<uchar>(i) + dscs.cols);
  }

  util::HDF5File out_file("/home/alessandro/debug/test.hdf5");
  std::vector<int> descriptors_size{dscs.rows, dscs.cols};
  out_file.writeArray("descriptors", all_descriptors, descriptors_size, true);

  cout << "input descriptor size " << dscs.rows << " " << dscs.cols << endl;

  std::vector<uchar> out_descriptors;
  std::vector<int> out_dscs_size;
  // int test_feature_size;
  out_file.readArray<uchar>("descriptors", out_descriptors, out_dscs_size);

  Mat out_mat;
  vectorToMat(out_descriptors, out_dscs_size, out_mat);

  cout << "out mat " << out_mat.cols << " " << out_mat.rows << endl;
}

void testReading() {
  util::HDF5File out_file(
      "/home/alessandro/projects/drone_ws/src/fato_tracker/data/models/"
      "ros_hydro/ros_hydro_features.h5");

  std::vector<uchar> out_descriptors;
  std::vector<int> out_dscs_size;
  // int test_feature_size;
  out_file.readArray<uchar>("descriptors", out_descriptors, out_dscs_size);

  Mat out_mat;
  vectorToMat(out_descriptors, out_dscs_size, out_mat);

  cout << "out mat " << out_mat.cols << " " << out_mat.rows << endl;
}

void imagetest() {
  Mat fst = imread(
      "/home/alessandro/projects/drone_ws/src/fato_tracker/data/imgs/hydro.png",
      0);
  Mat scd = imread(
      "/home/alessandro/projects/drone_ws/src/fato_tracker/data/imgs/cam.png",
      0);

  BriskMatcher custom;
  Mat fst_dsc, scd_dsc;
  vector<KeyPoint> fst_kps, scd_kps;
  custom.extract(fst, fst_kps, fst_dsc);
  custom.extract(scd, scd_kps, scd_dsc);

  std::vector<std::vector<cv::DMatch>> custom_matches, cv_matches;

  CustomMatcher matcher_custom;
  auto begin_custom = my_clock::now();
  matcher_custom.matchV2(fst_dsc, scd_dsc, custom_matches);
  auto end_custom = my_clock::now();

  cv::BFMatcher matcher(NORM_HAMMING);
  auto begin_cv = my_clock::now();
  matcher.knnMatch(fst_dsc, scd_dsc, cv_matches, 2);
  auto end_cv = my_clock::now();

  cout << "kps " << fst_kps.size() << " " << scd_kps.size() << endl;
  cout << "kps " << fst_dsc.cols << " " << scd_dsc.cols << endl;
  cout << "matches " << custom_matches.size() << " " << cv_matches.size()
       << endl;

  for (auto i = 0; i < custom_matches.size(); ++i) {
    DMatch mine = custom_matches.at(i)[0];
    DMatch alt = cv_matches.at(i)[0];

    cout << mine.trainIdx << " " << mine.queryIdx << " " << mine.distance
         << endl;
    cout << alt.trainIdx << " " << alt.queryIdx << " " << alt.distance << endl;
    cout << "\n";
  }

  float ms_custom =
      duration_cast<nanoseconds>(end_custom - begin_custom).count();
  float ms_cv = duration_cast<nanoseconds>(end_cv - begin_cv).count();

  cout << fixed << setprecision(3) << "opencv " << ms_cv / 1000.0 << " mine "
       << ms_custom / 1000.0 << endl;

  imshow("debug", fst);
  waitKey(0);

  // testing akaze matcher
  fst_kps.clear();
  scd_kps.clear();
  fst_dsc = Mat();
  scd_dsc = Mat();
  cv_matches.clear();
  ;

  Size sz1 = fst.size();
  Size sz2 = scd.size();

  Mat fst_rgb, scd_rgb;

  Mat out(sz1.height, sz1.width + sz2.width, CV_8UC3);
  cvtColor(fst, fst_rgb, CV_GRAY2BGR);
  cvtColor(scd, scd_rgb, CV_GRAY2BGR);
  fst_rgb.copyTo(out(Rect(0, 0, sz1.width, sz1.height)));
  scd_rgb.copyTo(out(Rect(sz1.width, 0, sz2.width, sz2.height)));
  Mat out2;
  out.copyTo(out2);

  AkazeMatcher akaze_matcher;
  akaze_matcher.extract(fst, fst_kps, fst_dsc);
  akaze_matcher.extract(scd, scd_kps, scd_dsc);
  matcher.knnMatch(fst_dsc, scd_dsc, cv_matches, 2);

  for (auto i = 0; i < cv_matches.size(); ++i) {
    DMatch best = cv_matches.at(i)[0];
    DMatch second = cv_matches.at(i)[1];

    float confidence = 1 - (best.distance / akaze_matcher.maxDistance());
    float ratio = best.distance / second.distance;

    // if (confidence < 0.8 || ratio > 0.8) continue;

    Point2f fst_pt = fst_kps[best.trainIdx].pt;
    Point2f scd_pt = scd_kps[best.queryIdx].pt;
    scd_pt.x += sz1.width;

    circle(out, fst_pt, 2, Scalar(0, 255, 0), -1);
    circle(out, scd_pt, 2, Scalar(0, 255, 0), -1);
    line(out, fst_pt, scd_pt, Scalar(0, 255, 0), 1);
  }

  imshow("Akaze match", out);
  waitKey(0);

  fst_kps.clear();
  scd_kps.clear();
  custom_matches.clear();

  akaze_matcher.extractTarget(fst);
  fst_kps = akaze_matcher.getTargetPoints();

  akaze_matcher.match(scd, scd_kps, scd_dsc, custom_matches);

  for (auto i = 0; i < custom_matches.size(); ++i) {
    DMatch best = custom_matches.at(i)[0];
    DMatch second = custom_matches.at(i)[1];

    float confidence = 1 - (best.distance / akaze_matcher.maxDistance());
    float ratio = best.distance / second.distance;

    if (confidence < 0.8 || ratio > 0.8) continue;

    Point2f fst_pt = fst_kps[best.trainIdx].pt;
    Point2f scd_pt = scd_kps[best.queryIdx].pt;
    scd_pt.x += sz1.width;

    circle(out2, fst_pt, 2, Scalar(0, 255, 0), -1);
    circle(out2, scd_pt, 2, Scalar(0, 255, 0), -1);
    line(out2, fst_pt, scd_pt, Scalar(0, 255, 0), 1);
  }

  imshow("Custom Akaze match", out2);
  waitKey(0);
}

void rgbCallback(const sensor_msgs::ImageConstPtr& rgb_msg) {
  Mat rgb;
  cv_bridge::CvImageConstPtr pCvImage;
  pCvImage = cv_bridge::toCvShare(rgb_msg, rgb_msg->encoding);
  pCvImage->image.copyTo(rgb);

  rgb.copyTo(rgb_in);

  // cvtColor(rgb, rgb_in, CV_RGB2BGR);

  image_updated = true;
}

void livetest() {
  AkazeMatcher matcher;
  matcher.init(640,480);

  util::HDF5File out_file(
      "/home/alessandro/projects/drone_ws/src/fato_tracker/data/models/"
      "ros_hydro/ros_hydro_features.h5");
  std::vector<uchar> descriptors;
  std::vector<int> dscs_size, point_size;
  std::vector<float> points;
  // int test_feature_size;
  out_file.readArray<uchar>("descriptors", descriptors, dscs_size);
  out_file.readArray<float>("positions", points, point_size);

  Mat init_descriptors;
  vectorToMat(descriptors, dscs_size, init_descriptors);

  matcher.setTarget(init_descriptors);

  ros::NodeHandle nh;
  boost::shared_ptr<image_transport::ImageTransport> rgb_it;
  image_transport::SubscriberFilter sub_rgb;

  rgb_it.reset(new image_transport::ImageTransport(nh));

  sub_rgb.subscribe(*rgb_it, "/image_raw", 1,
                    image_transport::TransportHints("raw"));

  image_publisher = nh.advertise<sensor_msgs::Image>("fato_matching/image", 1);

  sub_rgb.registerCallback(boost::bind(rgbCallback, _1));

  ros::AsyncSpinner spinner(0);
  spinner.start();

  ros::Rate r(100);

  image_updated = false;

  while (ros::ok()) {
    if (image_updated == false) continue;

    std::vector<cv::KeyPoint> keypoints;

    Mat gray;
    cvtColor(rgb_in, gray, CV_BGR2GRAY);
    cout << rgb_in.channels() << endl;

    cv::Mat descriptors;
    // matcher.extract(gray, keypoints, descriptors);

    vector<vector<DMatch>> matches;

    matcher.match(gray, keypoints, descriptors, matches);

    cout << "number of matches: " << matches.size() << endl;
    cout << "descriptors: " << descriptors.cols << " " << descriptors.rows
         << endl;

    float max_distance = matcher.maxDistance();
    cout << "confidence " << matcher.maxDistance() << endl;

    for (auto i = 0; i < matches.size(); ++i) {
      const int& match_idx = matches.at(i)[0].queryIdx;

      float confidence = 1 - (matches[i][0].distance / max_distance);
      auto ratio = (matches[i][0].distance / matches[i][1].distance);

      Point2f pt = keypoints[match_idx].pt;

      if (confidence < 0.7) {
        uchar val = (uchar)confidence * 255;
        circle(rgb_in, pt, 3, Scalar(255, 0, 0), 1);
      } else if (ratio > 0.8) {
        uchar val = (uchar)confidence * 255;
        circle(rgb_in, pt, 3, Scalar(255, 102, 0), 1);
      } else {
        uchar val = (uchar)confidence * 255;
        circle(rgb_in, pt, 3, Scalar(0, 255, 0), 1);
      }
    }

    cv_bridge::CvImage cv_rend;
    cv_rend.image = rgb_in;
    cv_rend.encoding = sensor_msgs::image_encodings::RGB8;
    image_publisher.publish(cv_rend.toImageMsg());

    image_updated = false;

    r.sleep();
  }
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "fato_matching_test");

  // testReading();

  // testWritingReadingDescriptors();

  //compareAkazeImp();

  // cudaAkaze();

  // imagetest();

   livetest();

  return 0;
}
