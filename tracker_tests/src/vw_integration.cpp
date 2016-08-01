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
#include <chrono>

using namespace std;
using namespace cv;

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

      tracker.init(frameGray, nvx_kps);

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
      tracker.getValidPoints(20, prev_points, next_points);
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

  util::Device2D<uchar> d_texture(image_w,image_h);

  vx_imagepatch_addressing_t img_addrs[1];
  img_addrs[0].dim_x = 640;
  img_addrs[0].dim_y = 480;
  img_addrs[0].stride_x = sizeof(uchar);
  img_addrs[0].stride_y = vx_int32(d_texture.pitch_);
  void* src_ptr[] = {d_texture.data()};

  cout << "here" << endl;

  vx_df_image format = VX_DF_IMAGE_VIRT;
  vx_uint32 width = 0;
  vx_uint32 height = 0;


  vx_image src = vxCreateImage(context, 640, 480, VX_DF_IMAGE_U8);
  vx_rectangle_t rect;
  vxGetValidRegionImage(src, &rect);
  vx_uint8* dst_ptr = NULL; // should be NULL to work in MAP mode
  vx_imagepatch_addressing_t dst_addr;
  vxAccessImagePatch(src, &rect, 0, &dst_addr, (void **)&dst_ptr, NVX_WRITE_ONLY_CUDA);
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
  //vx_rectangle_t rect = {0u, 0u, 640u, 480u};
  void* ptr = NULL;
  vx_imagepatch_addressing_t addr;

  nvx_cv::VXImageToCVMatMapper map(src, plane_index, &rect, VX_READ_ONLY,
                                   VX_IMPORT_TYPE_HOST);
  cv::Mat cv_img = map.getMat();

  imshow("debug", cv_img);
  //waitKey(0);

  std::vector<uchar> h_texture(image_w * image_h);
  d_texture.copyTo(h_texture);
  Mat rendered_img;
  cv::Mat img_rgba(image_h, image_w, CV_8UC1, h_texture.data());
  // cv::cvtColor(img_rgba, rendered_img, CV_RGBA2BGR);

  imshow("debug1", img_rgba);
  waitKey(0);

  cout << cv_img.cols << " " << cv_img.rows << " " << cv_img.channels() << " "
       << img_rgba.cols << " " << img_rgba.rows << " " << img_rgba.channels()
       << endl;

  ofstream file("/home/alessandro/debug/color.txt");

  for (int i = 0; i < cv_img.rows; ++i) {
    for (int j = 0; j < cv_img.cols; ++j) {
      uchar col = cv_img.at<uchar>(i, j);
      uchar col2 = img_rgba.at<uchar>(i, j);

      file << (int)col << "-" << (int)col2 << " ";
    }
    file << "\n";
  }
}

int main(int argc, char** argv) {
  vx_context context = vxCreateContext();

  // synthgraph_test(context);
  // realgraph_test(context);
  gl2Vx(context);

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
