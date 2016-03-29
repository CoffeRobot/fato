/*****************************************************************************/
/*  Copyright (c) 2016, Karl Pauwels, Alessandro Pieropan                    */
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

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <stdexcept>

// TOFIX: find why qtcreator cannot pickup header files
//#include <multiple_rigid_models_ogre.h>
//#include <windowless_gl_context.h>
//#include <hdf5_file.h>
//#include <utilities.h>
//#include <device_1d.h>

#include <utility_kernels.h>
#include <utility_kernels_pose.h>
#include <hdf5_file.h>
#include <multiple_rigid_models_ogre.h>

#include "../../fato_rendering/include/multiple_rigid_models_ogre.h"
#include "../../fato_rendering/include/windowless_gl_context.h"
#include "../../tracker/include/feature_matcher.hpp"
#include "../../utilities/include/device_1d.h"
#include "../../utilities/include/hdf5_file.h"
#include "../../utilities/include/utilities.h"

int main(int argc, char **argv) {
  // Create dummy GL context before cudaGL init
  render::WindowLessGLContext dummy(10, 10);

  // CUDA Init
  int device_id = 0;
  fato::initializeCUDARuntime(device_id);

  // default parameters
  int width = 640;
  int height = 480;
  double fx = 500.0;
  double fy = 500.0;
  double cx = width / 2.0;
  double cy = height / 2.0;
  double near_plane = 0.01;  // for init only
  double far_plane = 10.0;   // for init only

  // viewpoint rotations to apply (Euler angles)
  // more or less uniformly distributed
  std::vector<double> rot_x{
      0.0000, 0.2618, 1.3090, 2.3562, 3.4034, 4.4506, 5.4978, 0.0000,
      0.6046, 1.2092, 1.8138, 2.4184, 3.0230, 3.6276, 4.2322, 4.8368,
      5.4414, 0.2618, 0.7854, 1.3090, 1.8326, 2.3562, 2.8798, 3.4034,
      3.9270, 4.4506, 4.9742, 5.4978, 6.0214, 0.0000, 0.6046, 1.2092,
      1.8138, 2.4184, 3.0230, 3.6276, 4.2322, 4.8368, 5.4414, 0.2618,
      1.3090, 2.3562, 3.4034, 4.4506, 5.4978, 0.0000};

  std::vector<double> rot_y{
      -1.5708, -1.0472, -1.0472, -1.0472, -1.0472, -1.0472, -1.0472, -0.5236,
      -0.5236, -0.5236, -0.5236, -0.5236, -0.5236, -0.5236, -0.5236, -0.5236,
      -0.5236, 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
      0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.5236,  0.5236,  0.5236,
      0.5236,  0.5236,  0.5236,  0.5236,  0.5236,  0.5236,  0.5236,  1.0472,
      1.0472,  1.0472,  1.0472,  1.0472,  1.0472,  1.5708};

  if (argc < 2) throw std::runtime_error("Usage: ./generate_model model.obj");

  // storage
  std::vector<float> all_keypoints;
  std::vector<uchar> all_descriptors;
  int all_num_features = 0;

  // setup the engines
  std::string obj_file_name(argv[1]);
  pose::MultipleRigidModelsOgre model_ogre(width, height, fx, fy, cx, cy,
                                           near_plane, far_plane);
  model_ogre.addModel(obj_file_name);

  // check existence output file
  size_t ext_pos = obj_file_name.find_last_of(".");
  std::string h5_file_name = obj_file_name;
  h5_file_name.replace(ext_pos, 4, "_features.h5");

  if (boost::filesystem::exists(boost::filesystem::path(h5_file_name))) {
    char user_input = ' ';
    while ((user_input != 'y') && (user_input != 'n')) {
      std::cout << h5_file_name + " exists. Remove and overwrite? [y/n]: ";
      std::cin >> user_input;
    }
    if (user_input == 'n')
      return EXIT_SUCCESS;
    else  // remove the file
      boost::filesystem::remove(boost::filesystem::path(h5_file_name));
  }

  fato::BriskMatcher matcher;

  // object vertices
  const render::RigidObject &obj_model = model_ogre.getRigidObject(0);
  Eigen::Map<const Eigen::MatrixXf> vertices_f(obj_model.getPositions().data(),
                                               3, obj_model.getNPositions());
  Eigen::MatrixXd vertices;
  vertices = vertices_f.cast<double>();

  // bounding box
  Eigen::Map<const Eigen::MatrixXf> bounding_box_f(
      obj_model.getBoundingBox().data(), 3, 8);
  Eigen::Matrix<double, 3, 8> bounding_box;
  bounding_box = bounding_box_f.cast<double>();

  // centralizing translation
  auto mn = vertices.rowwise().minCoeff();
  auto mx = vertices.rowwise().maxCoeff();
  Eigen::Translation<double, 3> tra_center(-(mn + mx) / 2.0f);

  std::string window_name = "Model Construction";
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

  int dsc_size = 0;

  bool is_init = false;
  cv::Mat all_mat;
  // process all views
  for (int view_ind = 0; view_ind < rot_x.size(); ++view_ind) {
    // compute view rotation (rot_x -> rot_y)
    Eigen::Matrix3d rot_view;
    rot_view = Eigen::AngleAxisd(rot_y.at(view_ind), Eigen::Vector3d::UnitY()) *
               Eigen::AngleAxisd(rot_x.at(view_ind), Eigen::Vector3d::UnitX());

    // rotate around x to 'render' just as Ogre
    Eigen::Matrix3d Rx_180 = Eigen::Matrix<double, 3, 3>::Identity();
    Rx_180(1, 1) = -1.0;
    Rx_180(2, 2) = -1.0;
    rot_view *= Rx_180;

    // apply tra_center -> rot_view to bounding box
    Eigen::Transform<double, 3, Eigen::Affine> t = rot_view * tra_center;
    Eigen::Matrix<double, 3, 8> bb;
    bb = t * bounding_box;

    // compute minimal z-shift required to ensure visibility
    // assuming cx,cy in image center
    Eigen::Matrix<double, 1, 8> shift_x =
        (2.0 * fx / (double)width) * bb.row(0).array().abs();
    shift_x -= bb.row(2);
    Eigen::Matrix<double, 1, 8> shift_y =
        (2.0 * fy / (double)height) * bb.row(1).array().abs();
    shift_y -= bb.row(2);
    Eigen::Matrix<double, 1, 16> shift;
    shift << shift_x, shift_y;
    double z_shift = shift.maxCoeff();
    Eigen::Translation<double, 3> tra_z_shift(0, 0, z_shift);

    // compute bounding box limits after z-shift
    near_plane = (bb.row(2).array() + z_shift).minCoeff();
    far_plane = (bb.row(2).array() + z_shift).maxCoeff();

    // compose render transform (tra_center -> rot_view -> tra_z_shift)
    Eigen::Transform<double, 3, Eigen::Affine> t_render =
        tra_z_shift * rot_view * tra_center;

    double tra_render[3];
    double rot_render[9];
    Eigen::Map<Eigen::Vector3d> tra_render_eig(tra_render);
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> rot_render_eig(
        rot_render);
    tra_render_eig = t_render.translation();
    rot_render_eig = t_render.rotation();

    std::vector<pose::TranslationRotation3D> TR(1);
    TR.at(0).setT(tra_render);
    TR.at(0).setR_mat(rot_render);

    model_ogre.updateProjectionMatrix(fx, fy, cx, cy, near_plane, far_plane);
    model_ogre.render(TR);

    util::Device1D<uchar4> d_texture(height * width);
    vision::convertFloatArrayToGrayRGBA(
        d_texture.data(), model_ogre.getTexture(), width, height, 1.0, 2.0);
    std::vector<uchar4> h_texture(height * width);
    d_texture.copyTo(h_texture);

    cv::Mat sift_img_rgba(height, width, CV_8UC4, h_texture.data());
    cv::Mat sift_img_gray;
    cv::cvtColor(sift_img_rgba, sift_img_gray, CV_RGBA2GRAY);

    // extract SIFT features
    //    bool success = sift_engine.RunSIFT(width, height, sift_img_gray.data,
    //                                       GL_LUMINANCE, GL_UNSIGNED_BYTE);
    //    if (!success)
    //      throw std::runtime_error("SiftGPU failed");

    matcher.extractTarget(sift_img_gray);
    std::vector<cv::KeyPoint> &points = matcher.getTargetPoints();
    cv::Mat &dscs = matcher.getTargetDescriptors();
    int num_features = points.size();

    if (!is_init && num_features > 0) {
      all_mat = dscs.clone();
      is_init = true;
    } else if (num_features > 0) {
      cv::vconcat(all_mat, dscs, all_mat);
    }

    std::vector<uchar> descriptors;

    if (num_features > 0) {
      dsc_size = dscs.cols;

      for (int i = 0; i < dscs.rows; ++i) {
        descriptors.insert(descriptors.end(), dscs.ptr<uchar>(i),
                           dscs.ptr<uchar>(i) + dscs.cols);
      }
    }

    // 3D keypoints (project on z-buffer)
    util::Device1D<float> d_z(height * width);
    pose::convertZbufferToZ(d_z.data(), model_ogre.getZBuffer(), width, height,
                            cx, cy, near_plane, far_plane);
    std::vector<float> h_z(height * width);
    d_z.copyTo(h_z);

    std::vector<float> key_XYZ(3 * num_features);
    for (int k = 0; k < num_features; ++k) {
      float pt_x = points.at(k).pt.x;
      float pt_y = points.at(k).pt.y;

      int x = floor(pt_x);
      x = (x < 0) ? 0 : x;
      x = (x >= width) ? (width - 1) : x;
      int y = floor(pt_y);
      y = (y < 0) ? 0 : y;
      y = (y >= height) ? (height - 1) : y;

      key_XYZ.at(3 * k + 2) = h_z.at(y * width + x);
      key_XYZ.at(3 * k + 0) = (pt_x - cx) * key_XYZ.at(3 * k + 2) / fx;
      key_XYZ.at(3 * k + 1) = (pt_y - cy) * key_XYZ.at(3 * k + 2) / fy;
    }

    // transform 3D keypoints to original 3D model coordinates
    Eigen::Map<Eigen::MatrixXf> key_XYZ_eig(key_XYZ.data(), 3, num_features);
    Eigen::MatrixXd tmp(3, num_features);
    tmp =
        t_render.inverse() * key_XYZ_eig.cast<double>().colwise().homogeneous();
    key_XYZ_eig = tmp.cast<float>();

    // store
    all_keypoints.insert(all_keypoints.end(), key_XYZ.begin(), key_XYZ.end());
    all_descriptors.insert(all_descriptors.end(), descriptors.begin(),
                           descriptors.end());
    all_num_features += num_features;

    // draw keypoints
    cv::Mat sift_img_keys;
    cv::cvtColor(sift_img_gray, sift_img_keys, CV_GRAY2BGR);
    for (auto &it : points)
      cv::circle(sift_img_keys, cv::Point2d(it.pt.x, it.pt.y), 2,
                 CV_RGB(0, 255, 0), -1, 8);

    cv::imshow(window_name, sift_img_keys);
    cv::waitKey(5);
  }

  //TODO: some points are NaN, investigate!
//  std::cout << "Keypoint size: " << all_keypoints.size() << std::endl;
//  for (auto i = 0; i < all_keypoints.size(); i += 3) {
//    std::cout << all_keypoints[i] << " " << all_keypoints[i + 1] << " "
//              << all_keypoints[i + 2] << "\n";
//  }

  // randomly shuffle positions and descriptors (siftgpu limitation)
  //  std::vector<int> shuffle_inds(all_num_features);
  //  std::iota(shuffle_inds.begin(), shuffle_inds.end(), 0);
  //  std::random_shuffle(shuffle_inds.begin(), shuffle_inds.end());

  //  Eigen::Map<Eigen::VectorXi> shuffle_inds_eig(shuffle_inds.data(),
  //                                               all_num_features);
  //  Eigen::Map<Eigen::MatrixXf> all_keypoints_eig(all_keypoints.data(), 3,
  //                                                all_num_features);
  //  Eigen::Map<Eigen::MatrixXf> all_descriptors_eig(all_descriptors.data(),
  //  128,
  //                                                  all_num_features);

  //  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> p(
  //      shuffle_inds_eig);
  //  all_keypoints_eig = all_keypoints_eig * p;
  //  all_descriptors_eig = all_descriptors_eig * p;

  // save
  util::HDF5File out_file(h5_file_name);
  std::vector<int> descriptors_size{all_num_features, dsc_size};
  std::vector<int> positions_size{all_num_features, 3};
  out_file.writeArray("descriptors", all_descriptors, descriptors_size, true);
  out_file.writeArray("positions", all_keypoints, positions_size, true);
  // out_file.writeScalar("feature_size", 64);

  std::vector<uchar> test_descriptors;
  std::vector<int> test_size, point_size;
  std::vector<float> points;
  // int test_feature_size;
  out_file.readArray<uchar>("descriptors", test_descriptors, test_size);
  out_file.readArray<float>("positions", points, point_size);


  std::cout << "feature size in: " << test_size[0] << " " << test_size[1]
            << std::endl;
  std::cout << "points size in: " << point_size[0] << " " << point_size[1]
            << std::endl;

  cv::Mat1b test;  //(test_size[0], test_size[1]);

  fato::vectorToMat(test_descriptors, test_size, test);

  std::cout << "src_dsc " << all_descriptors.size() << " dst_dsc "
            << test_descriptors.size() << std::endl;

  std::cout << "Mat size " << test.cols << "  " << test.rows << std::endl;

  std::cout << "ALL_Mat size " << all_mat.cols << "  " << all_mat.rows
            << std::endl;

  int err_count = 0;
  for (int i = 0; i < all_descriptors.size(); ++i) {
    if (all_descriptors.at(i) != test_descriptors.at(i)) err_count++;
  }
  std::cout << "Errors in the arrays " << err_count << std::endl;

  err_count = 0;
  for (auto i = 0; i < all_mat.rows; ++i) {
    for (auto j = 0; j < all_mat.cols; ++j) {
      if (std::abs((int)all_mat.at<uchar>(i, j) - (int)test.at<uchar>(i, j)) !=
          0)
        err_count++;
    }
  }
  std::cout << "Error in mats " << err_count << "\n";

  err_count = 0;
  for(auto i = 0; i < points.size(); ++i)
  {
    if(all_keypoints.at(i) != points.at(i))
    {
        err_count ++;
        //std::cout << all_keypoints.at(i) << " " << points.at(i) << std::endl;
    }
  }
  std::cout << "Error in points " << err_count << "\n";
  //  for (int i = 0; i < 1; ++i) {
  //    for (int j = 0; j < test.cols; ++j) {
  //      std::cout << "[" << (int)test.at<uchar>(j, i) << ","
  //                << (int)all_mat.at<uchar>(j, i) << "]" <<
  //                (int)test1.at<uchar>(i,j)
  //                << "," << (int)test_descriptors.at(j) << " ";
  //    }
  //    std::cout << "\n";
  //  }

  return EXIT_SUCCESS;
}
