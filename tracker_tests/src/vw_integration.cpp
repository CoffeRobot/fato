/*****************************************************************************/
/*  Copyright (c) 2016, Alessandro Pieropan                                  */
/*  All rights reserved.                                                     */
/*                                                                           */
/*  Redistribution and use in source and binary forms, with or without       */
/*  modification, are permitted provided that the following conditions       */
/*  are met:                                                                 */
/*                                                                           */
/*  1. Redistributions of source code must retain the above copyright        */
/*  notice, this list of conditions and the following disclaimer.            */
/*                                                                           */
/*  2. Redistributions in binary form must reproduce the above copyright     */
/*  notice, this list of conditions and the following disclaimer in the      */
/*  documentation and/or other materials provided with the distribution.     */
/*                                                                           */
/*  3. Neither the name of the copyright holder nor the names of its         */
/*  contributors may be used to endorse or promote products derived from     */
/*  this software without specific prior written permission.                 */
/*                                                                           */
/*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      */
/*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        */
/*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    */
/*  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     */
/*  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   */
/*  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         */
/*  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    */
/*  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    */
/*  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      */
/*  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    */
/*  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     */
/*****************************************************************************/

#include "../../fato_rendering/include/multiple_rigid_models_ogre.h"
#include "../../fato_rendering/include/windowless_gl_context.h"
#include "../../fato_rendering/include/device_2d.h"
#include "../../fato_rendering/include/device_1d.h"
#include "../../tracker/include/flow_graph.hpp"
#include "../../tracker/include/nvx_utilities.hpp"
#include "../../cuda/include/utility_kernels.h"
#include <NVX/nvxcu.h>
#include <NVX/nvx_opencv_interop.hpp>
#include <NVX/nvx_timer.hpp>
#include <string>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <chrono>
#include <random>
#include <iostream>

#include "../../tracker/include/target.hpp"
#include "../../tracker/include/pose_estimation.h"


using namespace std;
using namespace cv;


void testAverage()
{
    random_device rd;
    default_random_engine dre(rd());
    uniform_int_distribution<int> dist(0,6);

    fato::TrackingHistory history(5,3);

    vector<double> b{0,0,0,0,0,0};

    fato::Pose p(b);
    history.init(p);

    for(int i = 0; i < 100; ++i)
    {
       int val = dist(dre);

       cout << "creating pose with " << val << endl;

       vector<double> beta{val,0,0,val,0,0};

       fato::Pose tmp(beta);

       history.update(tmp);

       auto vec = history.getHistory();

       for(auto el : vec)
           cout << el << " ";
       cout << "\n";

       cout << "average " << history.getAvgVelocity() << endl;
       cout << "confidence " << history.getConfidence().first << endl;

       cout << "average " << history.getAvgAngular() << endl;

       cin.get();
    }
}


void transferData(vx_context& context) {
  vector<nvx_keypointf_t> kps;

  nvx_keypointf_t kp;
  kp.error = -1;
  kp.orientation = 0;
  kp.scale = 1;
  kp.strength = 2;
  kp.strength = 3;
  kp.tracking_status = 4;
  kp.x = 5;
  kp.y = 6;

  nvx_keypointf_t kp2;
  kp2.error = 7;
  kp2.orientation = 8;
  kp2.scale = 9;
  kp2.strength = 10;
  kp2.strength = 11;
  kp2.tracking_status = 12;
  kp2.x = 13;
  kp2.y = 14;

  kps.push_back(kp);
  kps.push_back(kp);

  vx_array kp_list = vxCreateArray(context, NVX_TYPE_KEYPOINTF, 20);
  NVXIO_CHECK_REFERENCE(kp_list);

  NVXIO_SAFE_CALL(vxAddArrayItems(kp_list, kps.size(), kps.data(),
                                  sizeof(nvx_keypointf_t)));
  vx_size stride = 0;
  void* data = NULL;
  vxAccessArrayRange(kp_list, 0, kps.size(), &stride, &data, VX_READ_ONLY);
  for (auto i = 0; i < kps.size(); ++i) {
    nvx_keypointf_t point = vxArrayItem(nvx_keypointf_t, data, i, stride);
    cout << point.error << " " << point.orientation << " " << point.scale << " "
         << point.strength << " " << point.tracking_status << " " << point.x
         << " " << point.y << endl;
  }
  cout << "\n" << endl;
  vxCommitArrayRange(kp_list, 0, kps.size(), data);

  int old_size = kps.size();

  kps.push_back(kp2);
  kps.push_back(kp2);
  kps.push_back(kp2);

  int to_add = kps.size() - old_size;
  nvx_keypointf_t* p = kps.data() + old_size;

  NVXIO_SAFE_CALL(vxAddArrayItems(kp_list, to_add, p, sizeof(nvx_keypointf_t)));

  vxAccessArrayRange(kp_list, 0, kps.size(), &stride, &data, VX_READ_ONLY);
  for (auto i = 0; i < kps.size(); ++i) {
    nvx_keypointf_t point = vxArrayItem(nvx_keypointf_t, data, i, stride);
    cout << point.error << " " << point.orientation << " " << point.scale << " "
         << point.strength << " " << point.tracking_status << " " << point.x
         << " " << point.y << endl;
  }
  vxCommitArrayRange(kp_list, 0, kps.size(), data);

  vx_size size, max_capacity;
  vxQueryArray(kp_list, VX_ARRAY_ATTRIBUTE_NUMITEMS, &size, sizeof(size));
  vxQueryArray(kp_list, VX_ARRAY_ATTRIBUTE_CAPACITY, &max_capacity,
               sizeof(size));

  cout << "size " << size << " max " << max_capacity << endl;

  vxReleaseArray(&kp_list);
}

void uploadPoints(vx_context& context) {
  vx_array kp_list = vxCreateArray(context, NVX_TYPE_POINT2F, 20);
  NVXIO_CHECK_REFERENCE(kp_list);

  vx_size size, max_capacity = 0;
  vxQueryArray(kp_list, VX_ARRAY_ATTRIBUTE_NUMITEMS, &size, sizeof(size));
  vxQueryArray(kp_list, VX_ARRAY_ATTRIBUTE_CAPACITY, &max_capacity,
               sizeof(size));
  cout << "Num of points: " << size << " capacity " << max_capacity << endl;

  vx_size stride = 0;
  void* data = NULL;
  /* vxAccessArrayRange(kp_list, 0, 20, &stride, &data,
                           VX_READ_AND_WRITE)*/;
  vx_size new_size = 20;
  vxAddArrayItems(kp_list, new_size, &data, stride);

  vxQueryArray(kp_list, VX_ARRAY_ATTRIBUTE_NUMITEMS, &size, sizeof(size));
  vxQueryArray(kp_list, VX_ARRAY_ATTRIBUTE_CAPACITY, &max_capacity,
               sizeof(size));
  cout << "Num of points: " << size << " capacity " << max_capacity << endl;

  for (auto i = 0; i < size; ++i) {
    nvx_point2f_t* myptr = NULL;
    vxAccessArrayRange(kp_list, i, i + 1, &stride, (void**)&myptr,
                       VX_READ_AND_WRITE);
    myptr->x = vx_float32(i);
    myptr->y = vx_float32(i);
    vxCommitArrayRange(kp_list, i, i + 1, (void*)myptr);
  }

  vxAccessArrayRange(kp_list, 0, size, &stride, &data, VX_READ_ONLY);
  for (auto i = 0; i < 20; ++i) {
    nvx_point2f_t point = vxArrayItem(nvx_point2f_t, data, i, stride);
    cout << point.x << " " << point.y << endl;
  }
  vxCommitArrayRange(kp_list, 0, 20, data);

  vxTruncateArray(kp_list, 10);
  vxQueryArray(kp_list, VX_ARRAY_ATTRIBUTE_NUMITEMS, &size, sizeof(size));
  vxQueryArray(kp_list, VX_ARRAY_ATTRIBUTE_CAPACITY, &max_capacity,
               sizeof(size));
  cout << "Num of points: " << size << " capacity " << max_capacity << endl;

  vxAccessArrayRange(kp_list, 0, size, &stride, &data, VX_READ_ONLY);
  for (auto i = 0; i < size; ++i) {
    nvx_point2f_t point = vxArrayItem(nvx_point2f_t, data, i, stride);
    cout << point.x << " " << point.y << endl;
  }
  vxCommitArrayRange(kp_list, 0, 20, data);

  vxReleaseArray(&kp_list);
}

void realgraph_test(vx_context& context) {
  VideoCapture camera(0);
  Mat img;
  fato::vx::FeatureTracker::Params params;
  fato::vx::FeatureTrackerReal tracker(context, params);

  // return 0;

  vx_image mask = NULL;
  double proc_ms = 0;
  bool is_initialized = false;
  std::vector<nvx_keypointf_t> next_points, prev_points, back_points;
  while (camera.isOpened()) {
    camera >> img;

    if (img.empty()) break;

    vx_image vxiSrc;
    vxiSrc = nvx_cv::createVXImageFromCVMat(context, img);
    vx_image frameGray =
        vxCreateImage(context, img.cols, img.rows, VX_DF_IMAGE_U8);
    NVXIO_CHECK_REFERENCE(frameGray);
    NVXIO_SAFE_CALL(vxuColorConvert(context, vxiSrc, frameGray));

    if (!is_initialized) {
      is_initialized = true;

      Mat gray;
      cv::cvtColor(img, gray, CV_BGR2GRAY);

      vector<cv::KeyPoint> kps;
      cv::FAST(img, kps, 7, true);

      vector<nvx_keypointf_t> nvx_kps(kps.size(), nvx_keypointf_t());

      for (auto i = 0; i < kps.size(); ++i) {
        nvx_keypointf_t& nvx_kp = nvx_kps[i];

        nvx_kp.x = kps[i].pt.x;
        nvx_kp.y = kps[i].pt.y;
        nvx_kp.error = 0;
        nvx_kp.orientation = kps[i].angle;
        nvx_kp.tracking_status = -1;
        nvx_kp.strength = kps[i].response;
        nvx_kp.scale = 0;
        next_points.push_back(nvx_kp);
      }

      tracker.init(frameGray);

      for (auto i = 0; i < 10; ++i)
        cout << (int)next_points[i].tracking_status << endl;
      waitKey(0);

    } else {
      nvx::Timer procTimer;
      procTimer.tic();
      tracker.uploadPoints(next_points);
      tracker.track(frameGray, mask);
      proc_ms = procTimer.toc();
      next_points.clear();
      // tracker.downloadPoints(prev_points,next_points, back_points);
      // tracker.getValidPoints(20, prev_points, next_points);
      tracker.printPerfs();

      for (auto pt : next_points)
        cv::circle(img, Point2f(pt.x, pt.y), 3, cv::Scalar(255, 0, 0), 1);

      //        for(auto i = 0; i < 10; ++i)
      //            cout << (int)next_points[i].tracking_status << endl;

      cout << "features tracked " << next_points.size() << endl;
    }

    imshow("Debug", img);
    cv::waitKey(10);

    vxReleaseImage(&frameGray);
    vxReleaseImage(&vxiSrc);
  }
}

void trackCorners(const cv::Mat& rendered_image, const cv::Mat& next_img,
                  std::vector<cv::Point2f>& prev_pts,
                  std::vector<cv::Point2f>& next_pts, float distance,
                  Mat& out) {
  distance = distance * distance;

  vector<KeyPoint> kps;
  FAST(rendered_image, kps, 7, true);

  if (kps.empty()) return;

  vector<Point2f> tmp_next_pts, tmp_prev_pts, backflow_pts;

  for (auto kp : kps) tmp_prev_pts.push_back(kp.pt);

  vector<uchar> next_status, prev_status;
  vector<float> next_errors, prev_errors;

  calcOpticalFlowPyrLK(rendered_image, next_img, tmp_prev_pts, tmp_next_pts,
                       next_status, next_errors);
  calcOpticalFlowPyrLK(next_img, rendered_image, tmp_next_pts, backflow_pts,
                       prev_status, prev_errors);

  int lost_track = 0;
  int dist_err = 0;
  int correct = 0;

  for (int i = 0; i < tmp_prev_pts.size(); ++i) {
    float dx = backflow_pts[i].x - tmp_prev_pts[i].x;
    float dy = backflow_pts[i].y - tmp_prev_pts[i].y;

    float error = dx * dx + dy * dy;

    Scalar color(0, 255, 0);

    if (prev_status[i] != 1) {
      color = cv::Scalar(0, 0, 255);
      lost_track++;
    } else if (error > distance) {
      color = cv::Scalar(255, 0, 0);
      dist_err++;
    } else {
      correct++;
    }

    cv::Point2f p_pt = tmp_prev_pts[i];
    cv::Point2f n_pt = tmp_next_pts[i];
    n_pt.x += 640;
    prev_pts.push_back(p_pt);
    next_pts.push_back(n_pt);

    cv::circle(out, p_pt, 3, color, 1);
    cv::circle(out, n_pt, 3, color, 1);
    cv::line(out, p_pt, n_pt, color, 1);
  }

  cout << "CV: total features " << tmp_prev_pts.size() << endl;
  cout << "\t correct " << correct << " ratio "
       << correct / (float)tmp_prev_pts.size() << endl;
  cout << "\t lost " << lost_track << endl;
  cout << "\t dist " << dist_err << endl;
}

void synthgraph_test2(vx_context& context) {
  fato::vx::FeatureTrackerSynth::Params params;
  params.img_w = 640;
  params.img_h = 480;
  params.use_harris_detector = false;
  params.fast_type = 9;
  params.fast_thresh = 7;
  params.detector_cell_size = 7;
  params.lk_num_iters = 5;

  fato::vx::FeatureTrackerSynth tracker(context, params);

  Mat rendered = imread("/home/alessandro/debug/rendered.jpg", 0);
  Mat real = imread("/home/alessandro/debug/real.jpg", 0);

  vx_image vx_rend, vx_real;
  vx_rend = nvx_cv::createVXImageFromCVMat(context, rendered);
  vx_real = nvx_cv::createVXImageFromCVMat(context, real);

  tracker.track(vx_rend, vx_real);

  Mat rend_c, cam_c;
  Size sz1 = rendered.size();
  Size sz2 = real.size();
  Mat im3(sz1.height, sz1.width + sz2.width, CV_8UC3);

  cvtColor(rendered, rend_c, CV_GRAY2BGR);
  cvtColor(real, cam_c, CV_GRAY2BGR);

  rend_c.copyTo(im3(Rect(0, 0, sz1.width, sz1.height)));
  cam_c.copyTo(im3(Rect(sz1.width, 0, sz2.width, sz2.height)));
  Mat im4;
  im3.copyTo(im4);
  std::vector<cv::Point2f> prev_cv, next_cv;
  trackCorners(rendered, real, prev_cv, next_cv, 5, im4);

  std::vector<cv::Point2f> prev, next;
  tracker.debugPoints(25, prev, next, im3);

  tracker.printPerfs();

  imshow("Debug_vx", im3);
  imshow("Debug_cv", im4);
  waitKey(0);
}

void synthgraph_test(vx_context& context) {
  VideoCapture camera(0);
  Mat img;
  fato::vx::FeatureTrackerSynth::Params params;

  fato::vx::FeatureTrackerSynth tracker(context, params);

  // return 0;

  vx_delay gray_img_delay_;

  vx_image mask = NULL;
  double proc_ms = 0;
  bool is_initialized = false;
  std::vector<nvx_keypointf_t> next_points;
  Mat gray_out;

  while (camera.isOpened()) {
    camera >> img;

    if (img.empty()) break;

    vx_image vxiSrc;
    vxiSrc = nvx_cv::createVXImageFromCVMat(context, img);

    if (!is_initialized) {
      vx_image frameGray =
          vxCreateImage(context, img.cols, img.rows, VX_DF_IMAGE_U8);
      NVXIO_CHECK_REFERENCE(frameGray);
      NVXIO_SAFE_CALL(vxuColorConvert(context, vxiSrc, frameGray));

      tracker.init(frameGray);

      gray_img_delay_ = vxCreateDelay(context, (vx_reference)frameGray, 2);

      NVXIO_SAFE_CALL(vxuColorConvert(
          context, vxiSrc,
          (vx_image)vxGetReferenceFromDelay(gray_img_delay_, -1)));

      vxReleaseImage(&frameGray);

      is_initialized = true;

      // imshow("debug", img);
      // waitKey(10);
    } else {
      auto begin = chrono::high_resolution_clock::now();
      NVXIO_SAFE_CALL(vxuColorConvert(
          context, vxiSrc,
          (vx_image)vxGetReferenceFromDelay(gray_img_delay_, 0)));
      auto end = chrono::high_resolution_clock::now();

      float gray_download =
          chrono::duration_cast<chrono::nanoseconds>(end - begin).count();

      // do my stuff here
      tracker.track((vx_image)vxGetReferenceFromDelay(gray_img_delay_, -1),
                    (vx_image)vxGetReferenceFromDelay(gray_img_delay_, 0));

      begin = chrono::high_resolution_clock::now();
      nvx_cv::VXImageToCVMatMapper mapper(
          (vx_image)vxGetReferenceFromDelay(gray_img_delay_, 0));
      gray_out = mapper.getMat();
      end = chrono::high_resolution_clock::now();
      // gray_download += chrono::duration_cast<chrono::nanoseconds>(end -
      // begin).count();

      begin = chrono::high_resolution_clock::now();
      Mat gray;
      cvtColor(img, gray, CV_BGR2GRAY);
      end = chrono::high_resolution_clock::now();
      float cv_gray =
          chrono::duration_cast<chrono::nanoseconds>(end - begin).count();

      std::vector<nvx_keypointf_t> prev, next, proj;
      tracker.downloadPoints(prev, next, proj);

      vxAgeDelay(gray_img_delay_);

      for (auto i = 0; i < prev.size(); ++i) {
        Point2f pt1(prev[i].x, prev[i].y);
        Point2f pt2(next[i].x, next[i].y);

        circle(img, pt1, 3, Scalar(0, 255, 0), 1);
        line(img, pt1, pt2, Scalar(255, 0, 0), 1);
      }

      tracker.getValidPoints(5, prev, next);

      tracker.printPerfs();
      std::cout << "\t VX Gray : " << gray_download / 1000000.0 << " ms"
                << std::endl;
      std::cout << "\t CV Gray : " << cv_gray / 1000000.0 << " ms" << std::endl;
    }

    imshow("debug_gray", img);
    waitKey(10);
    vxReleaseImage(&vxiSrc);
  }
}

void gl2Vx(vx_context context) {
  render::WindowLessGLContext dummy(10, 10);

  int image_w = 640;
  int image_h = 480;
  double focal_length_x = 632.7361080533549;
  double focal_length_y = 634.2327075892116;
  double nodal_point_x = 321.9474832561696;
  double nodal_point_y = 223.9353111003978;

  string filename =
      "/home/alessandro/projects/drone_ws/src/fato_tracker/data/ros_hydro/"
      "ros_hydro.obj";

  pose::MultipleRigidModelsOgre rendering_engine(
      image_w, image_h, focal_length_x, focal_length_y, nodal_point_x,
      nodal_point_y, 0.01, 10.0);

  rendering_engine.addModel(filename);

  Eigen::Vector3d translation(0, 0, 0.7);
  Eigen::Vector3d rotation(M_PI, 1.3962634015954636, 0);

  double T[] = {translation[0], translation[1], translation[2]};
  double R[] = {rotation[0], rotation[1], rotation[2]};
  std::vector<pose::TranslationRotation3D> TR(1);
  TR.at(0) = pose::TranslationRotation3D(T, R);

  rendering_engine.render(TR);

  vx_df_image format = VX_DF_IMAGE_VIRT;
  vx_uint32 width = 0;
  vx_uint32 height = 0;

  vx_image src = vxCreateImage(context, 640, 480, VX_DF_IMAGE_U8);
  vx_rectangle_t rect;
  vxGetValidRegionImage(src, &rect);
  vx_uint8* dst_ptr = NULL;  // should be NULL to work in MAP mode
  vx_imagepatch_addressing_t dst_addr;
  vxAccessImagePatch(src, &rect, 0, &dst_addr, (void**)&dst_ptr,
                     NVX_WRITE_ONLY_CUDA);
  vision::convertFloatArrayToGrayVX((uchar*)dst_ptr,
                                    rendering_engine.getTexture(), image_w,
                                    image_h, dst_addr.stride_y, 1.0, 2.0);
  vxCommitImagePatch(src, &rect, 0, &dst_addr, dst_ptr);

  //  vx_image src = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, img_addrs,
  //                                         src_ptr, NVX_IMPORT_TYPE_CUDA);

  NVXIO_SAFE_CALL(
      vxQueryImage(src, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
  NVXIO_SAFE_CALL(
      vxQueryImage(src, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
  NVXIO_SAFE_CALL(
      vxQueryImage(src, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
  cout << format << " " << width << " " << height << " " << VX_DF_IMAGE_U8
       << endl;

  vx_uint32 plane_index = 0;
  // vx_rectangle_t rect = {0u, 0u, 640u, 480u};
  void* ptr = NULL;
  vx_imagepatch_addressing_t addr;

  nvx_cv::VXImageToCVMatMapper map(src, plane_index, &rect, VX_READ_ONLY,
                                   VX_IMPORT_TYPE_HOST);
  cv::Mat cv_img = map.getMat();

  imshow("Ogre2VX", cv_img);
  waitKey(0);

  vector<KeyPoint> kps;
  FAST(cv_img, kps, 7, true);

  cv::Mat cv_out, vx_out;

  cv_img.copyTo(cv_out);
  cv_img.copyTo(vx_out);

  for (auto kp : kps) circle(cv_out, kp.pt, 3, 0, 1);

  fato::vx::FeatureTrackerSynth::Params params;

  //  vx_pyramid pyr_exemplar =
  //      vxCreatePyramid(context, params.pyr_levels, VX_SCALE_PYRAMID_HALF,
  //                      width, height, VX_DF_IMAGE_U8);
  //  vxuGaussianPyramid(context, src, pyr_exemplar);
  // vx_scalar corners = 500;
  params.fast_thresh = 7;
  params.detector_cell_size = 3;
  params.fast_type = 9;

  vx_array kp_list =
      vxCreateArray(context, NVX_TYPE_KEYPOINTF, params.array_capacity);
  nvxuFastTrack(context, src, kp_list, NULL, nullptr, params.fast_type,
                params.fast_thresh, params.detector_cell_size, nullptr);

  vx_size prev_size;
  vx_size prev_stride = 0;
  void* prev_data = NULL;

  vxQueryArray(kp_list, VX_ARRAY_ATTRIBUTE_NUMITEMS, &prev_size,
               sizeof(prev_size));

  vxAccessArrayRange(kp_list, 0, prev_size, &prev_stride, &prev_data,
                     VX_READ_ONLY);

  for (auto i = 0; i < prev_size; ++i) {
    nvx_keypointf_t prev_kp =
        vxArrayItem(nvx_keypointf_t, prev_data, i, prev_stride);

    Point2f pt(prev_kp.x, prev_kp.y);
    circle(vx_out, pt, 3, 0, 1);
  }
  vxCommitArrayRange(kp_list, 0, prev_size, prev_data);

  vxReleaseArray(&kp_list);
  vxReleaseContext(&context);

  cout << "cv_points " << kps.size() << " vx " << prev_size << endl;

  imshow("Opnecv", cv_out);
  imshow("VX", vx_out);
  waitKey(0);

  // vxReleasePyramid(&pyr_exemplar);
}

void testObjectMovement() {
  render::WindowLessGLContext dummy(10, 10);

  int image_w = 640;
  int image_h = 480;
  double focal_length_x = 632.7361080533549;
  double focal_length_y = 634.2327075892116;
  double nodal_point_x = 321.9474832561696;
  double nodal_point_y = 223.9353111003978;

  string filename =
      "/home/alessandro/projects/drone_ws/src/fato_tracker/data/kellog/"
      "kellog.obj";

  pose::MultipleRigidModelsOgre rendering_engine(
      image_w, image_h, focal_length_x, focal_length_y, nodal_point_x,
      nodal_point_y, 0.01, 10.0);

  rendering_engine.addModel(filename);

  const render::RigidObject &obj_model = rendering_engine.getRigidObject(0);

  Eigen::Map<const Eigen::MatrixXf> vertices_f(obj_model.getPositions().data(),
                                               3, obj_model.getNPositions());
  Eigen::MatrixXd vertices;
  vertices = vertices_f.cast<double>();

  vector<float> position = obj_model.getBoundingBox();

  Eigen::Map<const Eigen::MatrixXf> bounding_box_f(
      obj_model.getBoundingBox().data(), 3, 8);

  Eigen::Matrix<double, 3, 8> bounding_box;
  bounding_box = bounding_box_f.cast<double>();

  // centralizing translation
  auto mn = vertices.rowwise().minCoeff();
  auto mx = vertices.rowwise().maxCoeff();
  Eigen::Translation<double, 3> tra_center(-(mn + mx) / 2.0f);

//  // compute minimal z-shift required to ensure visibility
//  // assuming cx,cy in image center
//  Eigen::Matrix<double, 1, 8> shift_x =
//      (2.0 * focal_length_x / (double)image_w) * bb.row(0).array().abs();
//  shift_x -= bb.row(2);
//  Eigen::Matrix<double, 1, 8> shift_y =
//      (2.0 * focal_length_y / (double)image_h) * bb.row(1).array().abs();
//  shift_y -= bb.row(2);
//  Eigen::Matrix<double, 1, 16> shift;
//  shift << shift_x, shift_y;
//  double z_shift = shift.maxCoeff();
//  Eigen::Translation<double, 3> tra_z_shift(0, 0, z_shift);

  for(auto i = 0; i <8; ++i)
  {
      cout << fixed << setprecision(3) << tra_center.x() << " " << tra_center.y() << " " << tra_center.z()  << endl;
  }

  Eigen::Vector3d translation(0, 0, 700);
  Eigen::Vector3d rotation(0, 0, 0);

  double T[] = {translation[0], translation[1], translation[2]};
  double R[] = {rotation[0], rotation[1], rotation[2]};
  std::vector<pose::TranslationRotation3D> TR(1);
  TR.at(0) = pose::TranslationRotation3D(T, R);

  rendering_engine.render(TR);

  std::vector<std::vector<double>> bboxes =
      rendering_engine.getBoundingBoxesInCameraImage(TR);

  std::vector<uchar4> h_texture(image_w * image_h);
  util::Device1D<uchar4> d_texture(image_w * image_h);
  vision::convertFloatArrayToGrayRGBA(d_texture.data(),
                                      rendering_engine.getTexture(), image_w,
                                      image_h, 1.0, 2.0);
  h_texture.resize(image_h * image_w);
  d_texture.copyTo(h_texture);

  cv::Mat img_rgba(image_h, image_w, CV_8UC4, h_texture.data());
  cv::Mat rendered_image;
  cv::cvtColor(img_rgba, rendered_image, CV_RGBA2BGR);

  vector<double> box = bboxes[0];
  cout << bboxes.size() << " " << box.size() << endl;

  Point2d pt0(box[0],box[1]);
  Point2d pt1(box[2],box[3]);
  Point2d pt2(box[4],box[5]);
  Point2d pt3(box[6],box[7]);
  Point2d pt4(box[8],box[9]);
  Point2d pt5(box[10],box[11]);
  Point2d pt6(box[12],box[13]);
  Point2d pt7(box[014],box[15]);


//  for(int i = 0; i < 4; i++)
//  {
//      circle(rendered_image, Point2f(box[2*i],box[2*i+1]), 3, Scalar(255,0,0), 1);
//  }

//  for(int i = 0; i < 4; i++)
//  {
//      circle(rendered_image, Point2f(box[2*i+4],box[2*i+1+4]), 3, Scalar(255,0,255), 1);
//  }

  for(int i = 0; i < 8; ++i)
  {
      cout << box[2*i] << " " << box[2*i+1] << endl;
  }

  line(rendered_image, pt0, pt2, Scalar(0,255,0), 1);
  line(rendered_image, pt2, pt6, Scalar(0,255,0), 1);
  line(rendered_image, pt6, pt4, Scalar(0,255,0), 1);
  line(rendered_image, pt4, pt0, Scalar(0,255,0), 1);

  imshow("debug", rendered_image);
  waitKey(0);

}

int main(int argc, char** argv) {
  vx_context context = vxCreateContext();

  // synthgraph_test(context);
  // realgraph_test(context);
  // gl2Vx(context);
  //synthgraph_test2(context);

  //testObjectMovement();
  testAverage();

  //    cout << VX_FAILURE << endl;
  //    cout << VX_ERROR_INVALID_REFERENCE << endl;
  //    cout << VX_ERROR_INVALID_PARAMETERS << endl;
  //    return 0;

  //    cout << sizeof(nvx_keypointf_t) << " " << sizeof(KeyPoint) << endl;

  //    transferData(context);

  //    return 0;

  // uploadPoints(context);

  //    nvxcu_pitch_linear_image_t image;
  //    image.base.image_type = NVXCU_PITCH_LINEAR_IMAGE;
  //    image.base.format = NVXCU_DF_IMAGE_U8;
  //    image.base.width = image_w;
  //    image.base.height = image_h;
  //    image.planes[0].dev_ptr = dev_ptr;
  //    image.planes[0].pitch_in_bytes = pitch;

  return 0;
}
