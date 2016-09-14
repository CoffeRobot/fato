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
#include "../../cuda/include/utility_kernels_pose.h"
#include "../../tracker/include/tracker_model_vx.h"
#include "../../utilities/include/draw_functions.h"
#include "../../fato_rendering/include/renderer.h"
#include "../../fato_rendering/include/env_config.h"

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
#include <ros/ros.h>

#include <cuda_runtime.h>

using namespace std;
using namespace cv;
using namespace std::chrono;

struct ParamsBench {
  float translation_x;
  float translation_y;
  float translation_z;
  float rotation_x;
  float rotation_y;
  float min_scale;
  float max_scale;

  int framerate;
  int camera_id;

  float running_time;

  string object_model_file;
  string object_descriptors_file;
  string result_file;
};

struct RenderData {
  vector<Mat> images;
  vector<Eigen::Transform<double, 3, Eigen::Affine>> poses;
  vector<glm::mat4> glm_poses;

  void addFrame(Mat &img, Eigen::Transform<double, 3, Eigen::Affine> &pose) {
    images.push_back(img.clone());
    poses.push_back(pose);
  }

  void addFrame(Mat &img, glm::mat4 &pose) {
    images.push_back(img.clone());
    glm_poses.push_back(pose);
  }
};

// default parameters
int width = 640;
int height = 480;
double fx = 500.0;
double fy = 500.0;
double cx = width / 2.0;
double cy = height / 2.0;
double near_plane = 0.01;  // for init only
double far_plane = 10.0;   // for init only
float deg2rad = M_PI / 180.0;
float rad2deg = 180.0 / M_PI;
Eigen::Matrix<double, 3, 8> bounding_box;

enum IMG_BOUND { UP = 1, DOWN, LEFT, RIGHT };

Eigen::Matrix3d getRotationView(float rot_x, float rot_y) {
  Eigen::Matrix3d rot_view;
  rot_view = Eigen::AngleAxisd(rot_y, Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(rot_x, Eigen::Vector3d::UnitX());
  // rotate around x to 'render' just as Ogre
  Eigen::Matrix3d Rx_180 = Eigen::Matrix<double, 3, 3>::Identity();
  Rx_180(1, 1) = -1.0;
  Rx_180(2, 2) = -1.0;
  // rot_view *= Rx_180;

  return rot_view;
}

double getZshif(Eigen::Matrix3d &rot_view, float scale_apperance,
                Eigen::Translation<double, 3> &tra_center) {
  // apply tra_center -> rot_view to bounding box
  Eigen::Transform<double, 3, Eigen::Affine> t = rot_view * tra_center;
  Eigen::Matrix<double, 3, 8> bb;
  bb = t * bounding_box;

  double size_x = bb.row(0).array().abs().maxCoeff();
  double size_y = bb.row(1).array().abs().maxCoeff();

  size_x = size_x * (2 - scale_apperance);
  size_y = size_y * (2 - scale_apperance);

  double x_shift = (2.0 * fx / (double)width) * size_x;
  double y_shift = (2.0 * fy / (double)height) * size_y;

  if (x_shift > y_shift)
    return x_shift;
  else
    return y_shift;
}

bool getTransformedObjectBox(Eigen::Transform<double, 3, Eigen::Affine> &pose,
                             pose::MultipleRigidModelsOgre &rendering_engine,
                             vector<vector<double>> &boxes) {
  double tra_render[3];
  double rot_render[9];
  Eigen::Map<Eigen::Vector3d> tra_render_eig(tra_render);
  Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> rot_render_eig(
      rot_render);
  tra_render_eig = pose.translation();
  rot_render_eig = pose.rotation();

  std::vector<pose::TranslationRotation3D> TR(1);
  TR.at(0).setT(tra_render);
  TR.at(0).setR_mat(rot_render);

  boxes = rendering_engine.getBoundingBoxesInCameraImage(TR);
}

bool getTransformedObjectBoxNew(
    Eigen::Transform<double, 3, Eigen::Affine> &pose,
    rendering::Renderer &renderer, vector<double> &bbox) {
  bbox = renderer.getBoundingBoxInCameraImage(0, pose);
}

void getPose(float rot_x, float rot_y,
             Eigen::Translation<double, 3> &tra_center,
             Eigen::Transform<double, 3, Eigen::Affine> &t_render) {
  Eigen::Matrix3d rot_view = getRotationView(rot_x, rot_y);

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
  t_render = tra_z_shift * rot_view * tra_center;
}

void renderObject(Eigen::Transform<double, 3, Eigen::Affine> &pose,
                  pose::MultipleRigidModelsOgre &model_ogre) {
  double tra_render[3];
  double rot_render[9];
  Eigen::Map<Eigen::Vector3d> tra_render_eig(tra_render);
  Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> rot_render_eig(
      rot_render);
  tra_render_eig = pose.translation();
  rot_render_eig = pose.rotation();

  std::vector<pose::TranslationRotation3D> TR(1);
  TR.at(0).setT(tra_render);
  TR.at(0).setR_mat(rot_render);

  model_ogre.updateProjectionMatrix(fx, fy, cx, cy, near_plane, far_plane);
  model_ogre.render(TR);
}

void downloadRenderedImg(pose::MultipleRigidModelsOgre &model_ogre,
                         std::vector<uchar4> &h_texture) {
  util::Device1D<uchar4> d_texture(height * width);
  vision::convertFloatArrayToGrayRGBA(d_texture.data(), model_ogre.getTexture(),
                                      width, height, 1.0, 2.0);
  h_texture.resize(height * width);
  d_texture.copyTo(h_texture);
}

void downloadRenderedImg(rendering::Renderer &renderer,
                         std::vector<uchar4> &h_texture) {
  util::Device1D<uchar4> d_texture(height * width);

  vision::downloadTextureToRGBA(d_texture.data(), renderer.getTexture(), width,
                                height);
  h_texture.resize(height * width);
  d_texture.copyTo(h_texture);
}

void downloadDepthBuffer(rendering::Renderer &renderer,
                         std::vector<float> &depth_buffer) {
  util::Device1D<float> rendered_depth(height * width);
  pose::convertZbufferToZ(rendered_depth.data(), renderer.getDepthBuffer(),
                          width, height, cx, cy, near_plane, far_plane);
  depth_buffer.resize(height * width);
  rendered_depth.copyTo(depth_buffer);
}

void drawBox(vector<double> &box, Mat &out) {
  Point2d pt0(box[0], box[1]);
  Point2d pt1(box[2], box[3]);
  Point2d pt2(box[4], box[5]);
  Point2d pt3(box[6], box[7]);
  Point2d pt4(box[8], box[9]);
  Point2d pt5(box[10], box[11]);
  Point2d pt6(box[12], box[13]);
  Point2d pt7(box[14], box[15]);

  line(out, pt0, pt2, Scalar(0, 255, 0), 1);
  line(out, pt2, pt6, Scalar(0, 255, 0), 1);
  line(out, pt6, pt4, Scalar(0, 255, 0), 1);
  line(out, pt4, pt0, Scalar(0, 255, 0), 1);

  line(out, pt1, pt3, Scalar(0, 255, 0), 1);
  line(out, pt3, pt7, Scalar(0, 255, 0), 1);
  line(out, pt7, pt5, Scalar(0, 255, 0), 1);
  line(out, pt5, pt1, Scalar(0, 255, 0), 1);
}

template <typename T>
int isOut(vector<T> &box) {
  for (int i = 0; i < 8; ++i) {
    if (box[2 * i] < 0) return LEFT;
    if (box[2 * i] > width) return RIGHT;
    if (box[2 * i + 1] < 0) return UP;
    if (box[2 * i + 1] > height) return DOWN;
  }

  return 0;
}

void blendImages(const Mat &cam, const Mat &render, Mat &out) {
  cam.copyTo(out);

  for (auto i = 0; i < cam.cols; ++i) {
    for (auto j = 0; j < cam.rows; ++j) {
      Vec3b rend_px = render.at<Vec3b>(j, i);
      if (rend_px[0] != 0 || rend_px[1] != 0 || rend_px[2] != 0) {
        Vec3b &out_px = out.at<Vec3b>(j, i);
        out_px = rend_px;
      }
    }
  }
}

void generateRenderedMovement(const ParamsBench &params,
                              pose::MultipleRigidModelsOgre &rendering_engine,
                              RenderData &data) {
  // object vertices
  const render::RigidObject &obj_model = rendering_engine.getRigidObject(0);
  Eigen::Map<const Eigen::MatrixXf> vertices_f(obj_model.getPositions().data(),
                                               3, obj_model.getNPositions());
  Eigen::MatrixXd vertices;
  vertices = vertices_f.cast<double>();

  // bounding box
  Eigen::Map<const Eigen::MatrixXf> bounding_box_f(
      obj_model.getBoundingBox().data(), 3, 8);

  bounding_box = bounding_box_f.cast<double>();

  // centralizing translation
  auto mn = vertices.rowwise().minCoeff();
  auto mx = vertices.rowwise().maxCoeff();

  Eigen::Translation<double, 3> tra_center(-(mn + mx) / 2.0f);

  Eigen::Transform<double, 3, Eigen::Affine> t_render;

  Eigen::Matrix3d rot_view = getRotationView(0, 0);

  double z_shift = getZshif(rot_view, 0.6, tra_center);
  Eigen::Translation<double, 3> tra_z_shift(0, 0, z_shift);
  // compose render transform (tra_center -> rot_view -> tra_z_shift)
  t_render = tra_z_shift * rot_view * tra_center;

  cout << "OGRE translation " << endl;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      cout << t_render(i, j) << " ";
    }
    cout << "\n";
  }

  renderObject(t_render, rendering_engine);

  std::vector<uchar4> h_texture(height * width);
  downloadRenderedImg(rendering_engine, h_texture);

  cv::Mat img_rgba(height, width, CV_8UC4, h_texture.data());
  cv::Mat img_gray;
  cv::cvtColor(img_rgba, img_gray, CV_RGBA2BGR);

  vector<vector<double>> boxes;
  getTransformedObjectBox(t_render, rendering_engine, boxes);

  drawBox(boxes[0], img_gray);

  float max_vel_h = params.translation_x;
  float max_vel_v = params.translation_y;
  float max_vel_d = params.translation_z;
  float rot_x_vel = params.rotation_x;  // degrees
  float rot_y_vel = params.rotation_y;

  float x_vel = max_vel_h;  // velocity per second
  float y_vel = max_vel_v;
  float z_vel = max_vel_d;

  float milliseconds = 0.010;

  float x_pos = 0;
  float y_pos = 0;
  float z_pos = z_shift;
  float x_rot = 0;

  double min_scale = getZshif(rot_view, params.min_scale, tra_center);
  double max_scale = getZshif(rot_view, params.max_scale, tra_center);

  return;

  high_resolution_clock::time_point last_time = high_resolution_clock::now();

  VideoCapture camera(params.camera_id);

  if (!camera.isOpened()) {
    cout << "cannot open camera!" << endl;
    // return;
  }

  namedWindow("debug", 1);
  Mat cam_img;

  int num_frames = params.framerate * params.running_time;
  int frame_count = 0;

  while (frame_count < num_frames) {
    auto now = high_resolution_clock::now();
    float dt = 0.03;
    // duration_cast<microseconds>(now - last_time).count() / 1000000.0f;

    if (dt > milliseconds) {
      x_pos += x_vel * float(dt);
      y_pos += y_vel * float(dt);
      z_pos += z_vel * float(dt);
      x_rot += rot_x_vel * float(dt);
      last_time = now;
    }

    float angle = x_rot * deg2rad;

    if (x_rot >= 360) x_rot = 0;

    rot_view = getRotationView(angle, 0);

    Eigen::Translation<double, 3> tra_z_shift(x_pos, y_pos, z_pos);
    t_render = tra_z_shift * rot_view * tra_center;
    renderObject(t_render, rendering_engine);
    getTransformedObjectBox(t_render, rendering_engine, boxes);

    if (isOut(boxes[0]) == LEFT)
      x_vel = max_vel_h;
    else if (isOut(boxes[0]) == RIGHT)
      x_vel = -max_vel_h;
    if (isOut(boxes[0]) == UP)
      y_vel = max_vel_v;
    else if (isOut(boxes[0]) == DOWN)
      y_vel = -max_vel_v;

    if (z_pos > min_scale) {
      z_vel = -max_vel_d;
    } else if (z_pos < max_scale) {
      z_vel = max_vel_d;
    }

    downloadRenderedImg(rendering_engine, h_texture);

    cv::Mat img_rgba(height, width, CV_8UC4, h_texture.data());
    cv::Mat img_gray, out;
    cv::cvtColor(img_rgba, img_gray, CV_RGBA2BGR);

    if (camera.isOpened()) {
      camera >> cam_img;
      blendImages(cam_img, img_gray, out);
    } else
      img_gray.copyTo(out);

    data.addFrame(out, t_render);

    imshow("debug", out);
    auto c = waitKey(30);
    if (c == 'q') break;

    frame_count++;
  }
}

void trackGeneratedData(RenderData &data, fato::TrackerVX &vx_tracker) {
  vector<cv::Scalar> axis;
  axis.push_back(cv::Scalar(255, 255, 0));
  axis.push_back(cv::Scalar(0, 255, 255));
  axis.push_back(cv::Scalar(255, 0, 255));

  Mat camera_matrix(3, 3, CV_64FC1);

  for (auto i = 0; i < 3; ++i) {
    for (auto j = 0; j < 3; ++j) {
      camera_matrix.at<double>(i, j) = 0;
    }
  }

  camera_matrix.at<double>(0, 0) = fx;
  camera_matrix.at<double>(1, 1) = fy;
  camera_matrix.at<double>(0, 2) = cx;
  camera_matrix.at<double>(1, 2) = cy;
  camera_matrix.at<double>(2, 2) = 1;

  vector<Mat> &images = data.images;

  //  VideoCapture camera(0);

  //  if (!camera.isOpened()) {
  //    cout << "cannot open camera!" << endl;
  //    return;
  //  }

  // Mat im;
  //  while(1)
  //  {

  //      camera >> im;
  //      vx_tracker.next(im);
  //      const fato::Target &target = vx_tracker.getTarget();
  //      vx_tracker.printProfile();
  //      //imshow("debug", im);
  //      auto c = waitKey(1);
  //      if (c == 'q') break;
  //  }

  for (Mat &im : images) {
    vx_tracker.next(im);
    vx_tracker.printProfile();

    Mat out;
    im.copyTo(out);

    const fato::Target &target = vx_tracker.getTarget();

    if (target.target_found_) {
      cout << "object found\n" << camera_matrix << endl;
      auto w_pose = target.weighted_pose.toCV();
      fato::drawObjectPose(target.centroid_, camera_matrix, w_pose.first,
                           w_pose.second, axis, out);
    } else {
      cout << "object  not found" << endl;
    }

    imshow("debug", out);
    auto c = waitKey(10);
    if (c == 'q') break;
  }
}

void useNewRenderer(const ParamsBench &params, RenderData &data) {
  rendering::Renderer renderer(width, height, fx, fy, cx, cy, near_plane,
                               far_plane);
  renderer.initRenderContext(
      640, 480, render::get_resources_path() + "shaders/framebuffer.vs",
      render::get_resources_path() + "shaders/framebuffer.frag");
  renderer.addModel(params.object_model_file,
                    render::get_resources_path() + "shaders/model.vs",
                    render::get_resources_path() + "shaders/model.frag");

  rendering::RigidObject &obj = renderer.getObject(0);

  cout << "object meshes " << obj.model.getMeshCount() << endl;

  vector<float> bbox = obj.getBoundingBox();

  Eigen::Map<const Eigen::MatrixXf> bounding_box_f(bbox.data(), 3, 8);

  bounding_box = bounding_box_f.cast<double>();

  for (auto val : bbox) cout << val << " ";
  cout << "\n";

  glm::vec3 translation_center = (-(obj.mins_ + obj.maxs_)) / 2.0f;

  Eigen::Translation<double, 3> tra_center(
      translation_center.x, translation_center.y, translation_center.z);

  Eigen::Transform<double, 3, Eigen::Affine> t_render;

  Eigen::Matrix3d rot_view = getRotationView(0, 0);

  double z_shift = getZshif(rot_view, 0.8, tra_center);
  cout << "z_shift " << z_shift << endl;
  Eigen::Translation<double, 3> tra_z_shift(0, 0, -z_shift);
  //  // compose render transform (tra_center -> rot_view -> tra_z_shift)
  t_render = tra_z_shift * rot_view * tra_center;

  obj.updatePose(t_render);

  float max_vel_h = params.translation_x;
  float max_vel_v = params.translation_y;
  float max_vel_d = params.translation_z;
  float rot_x_vel = params.rotation_x;  // degrees
  float rot_y_vel = params.rotation_y;

  float x_vel = max_vel_h;  // velocity per second
  float y_vel = max_vel_v;
  float z_vel = max_vel_d;

  float milliseconds = 0.010;

  float x_pos = 0.0;
  float y_pos = 0.0;
  float z_pos = -z_shift;
  float x_rot = 0;

  double min_scale = getZshif(rot_view, params.min_scale, tra_center);
  double max_scale = getZshif(rot_view, params.max_scale, tra_center);

  high_resolution_clock::time_point last_time = high_resolution_clock::now();

  VideoCapture camera(params.camera_id);

  if (!camera.isOpened()) {
    cout << "cannot open camera!" << endl;
    // return;
  }

  namedWindow("debug", 1);

  int num_frames = params.framerate * params.running_time;
  int frame_count = 0;

  std::vector<double> box1;

  while (frame_count < num_frames) {
    auto now = high_resolution_clock::now();

    float dt = 0.03;
    duration_cast<microseconds>(now - last_time).count() / 1000000.0f;

    if (dt > milliseconds) {
      x_pos += x_vel * float(dt);
      y_pos += y_vel * float(dt);
      z_pos += z_vel * float(dt);
      x_rot += rot_x_vel * float(dt);
      last_time = now;
    }

    float angle = x_rot * deg2rad;

    if (x_rot >= 360) x_rot = 0;

    rot_view = getRotationView(angle, 0);

    Eigen::Translation<double, 3> tra_z_shift(x_pos, y_pos, z_pos);
    t_render = tra_z_shift * rot_view * tra_center;
    obj.updatePose(t_render);

    renderer.render();

    std::vector<uchar4> h_texture(height * width);

    downloadRenderedImg(renderer, h_texture);
    Mat out(height, width, CV_8UC4, h_texture.data());
    vector<uchar4> h_texture2;
    renderer.downloadTextureCuda(h_texture2);
    // cout << "near plane " << near_plane << " far_plane " << far_plane <<
    // endl;

    vector<float> z_buffer;
    renderer.downloadDepthBuffer(z_buffer);
    vector<float> z_buffer2;
    renderer.downloadDepthBufferCuda(z_buffer2);

    float average_depth = 0;
    float count = 0;

    ofstream file("/home/alessandro/debug/cuda_buffer.txt");
    stringstream ss1;
    ss1 << fixed;
    for (auto i = 0; i < height; ++i) {
      for (auto j = 0; j < width; ++j) {
        int id = j + i * width;

        auto val = z_buffer[id];
        auto val2 = z_buffer[id];
        ss1 << val << "[" << val2 << "] ";
      }
      ss1 << "\n";
    }
    file << ss1.str();
    file.close();
    // cout << ss1.str() << "\n\n";

    //    for(auto val : z_buffer)
    //    {
    //        if(val != 0)
    //        {
    //            average_depth += val;
    //            count++;
    //        }
    //    }

    double average = average_depth / double(count);

    stringstream ss;
    cout << "average depth in buffer " << average_depth << " count " << count
         << " average " << average << endl;
    ss << "\n original pose \n";
    for (auto i = 0; i < 4; ++i) {
      for (auto j = 0; j < 4; ++j) {
        ss << t_render(j, i) << " ";
      }
      ss << "\n";
    }
    cout << fixed << setprecision(2) << ss.str();

    // renderObject(t_render, rendering_engine);
    getTransformedObjectBoxNew(t_render, renderer, box1);

    if (isOut(box1) == LEFT)
      x_vel = max_vel_h;
    else if (isOut(box1) == RIGHT)
      x_vel = -max_vel_h;
    if (isOut(box1) == UP)
      y_vel = max_vel_v;
    else if (isOut(box1) == DOWN)
      y_vel = -max_vel_v;

    if (z_pos > min_scale) {
      z_vel = -max_vel_d;
    } else if (z_pos < max_scale) {
      z_vel = max_vel_d;
    }

    data.addFrame(out, t_render);

   
    Mat out2(height, width, CV_8UC4, h_texture2.data());

    ofstream file2("/home/alessandro/debug/cuda_buffer_rgb.txt");
    stringstream ss2;
    ss1 << fixed;
    for (auto i = 0; i < height; ++i) {
      for (auto j = 0; j < width; ++j) {
        int id = j + i * width;

        uchar4 val = h_texture[id];
        uchar4 val2 = h_texture2[id];
        ss2 << (int)val.x << "," << (int)val.y << "," << (int)val.z << "[" << (int)val2.x << "," << (int)val2.y << "," << (int)val2.z << "] ";
      }
      ss2 << "\n";
    }
    file2 << ss2.str();
    file2.close();

    imshow("debug opencv", out);
    imshow("debug opencv_cuda", out2);


    auto c = waitKey(0);
    //    if (c == 'q') break;

    frame_count++;
  }
}

void trackGeneratedData(RenderData &data,
                        fato::TrackerVX::Params &tracker_params) {
  vector<cv::Scalar> axis;
  axis.push_back(cv::Scalar(255, 255, 0));
  axis.push_back(cv::Scalar(0, 255, 255));
  axis.push_back(cv::Scalar(255, 0, 255));

  Mat camera_matrix(3, 3, CV_64FC1);

  for (auto i = 0; i < 3; ++i) {
    for (auto j = 0; j < 3; ++j) {
      camera_matrix.at<double>(i, j) = 0;
    }
  }

  camera_matrix.at<double>(0, 0) = fx;
  camera_matrix.at<double>(1, 1) = fy;
  camera_matrix.at<double>(0, 2) = cx;
  camera_matrix.at<double>(1, 2) = cy;
  camera_matrix.at<double>(2, 2) = 1;

  vector<Mat> &images = data.images;

  std::unique_ptr<fato::FeatureMatcher> matcher =
      std::unique_ptr<fato::BriskMatcher>(new fato::BriskMatcher);
  fato::TrackerVX vx_tracker(tracker_params, std::move(matcher));

  int count = 0;
  for (Mat &im : images) {
    vx_tracker.next(im);
    // vx_tracker.printProfile();

    Mat out, rendered_pose;
    im.copyTo(out);

    const fato::Target &target = vx_tracker.getTarget();

    if (target.target_found_) {
      // cout << "object found\n" << camera_matrix << endl;
      auto w_pose = target.weighted_pose.toCV();
      fato::drawObjectPose(target.centroid_, camera_matrix, w_pose.first,
                           w_pose.second, axis, out);

      glm::mat4 glm_pose = target.weighted_pose.toGL();
      auto pose = data.poses.at(count);
      stringstream ss;
      ss << fixed << setprecision(2) << "weighted_pose \n";
      ;
      for (auto i = 0; i < 4; ++i) {
        for (auto j = 0; j < 4; ++j) {
          ss << glm_pose[i][j] << " ";
        }
        ss << "\n";
      }
      ss << "\n original pose \n";
      for (auto i = 0; i < 4; ++i) {
        for (auto j = 0; j < 4; ++j) {
          ss << pose(j, i) << " ";
        }
        ss << "\n";
      }
      // cout << ss.str() << endl;

      Eigen::Matrix4d Rx_180 = Eigen::Matrix<double, 4, 4>::Identity();
      Rx_180(1, 1) = -1.0;
      Rx_180(2, 2) = -1.0;

      glm::mat4 rotation;
      rotation = glm::rotate(rotation, 180.0f, glm::vec3(1, 0, 0));

      auto rotated_pose = target.weighted_pose;
      rotated_pose.transform(Rx_180);

      glm::mat4 rotated = rotated_pose.toGL();
      //      ss << "\n rotated pose \n";
      //      for(auto i = 0; i < 4; ++i)
      //      {
      //          for(auto j = 0; j < 4; ++j)
      //          {
      //            ss << rotated[i][j] << " ";
      //          }
      //          ss << "\n";
      //      }
      cout << ss.str() << endl;

      rendered_pose = vx_tracker.getRenderedPose();

    } else {
      cout << "object  not found" << endl;
    }

    count++;
    if(rendered_pose.cols > 0)
        imshow("debug tracking", rendered_pose);
    auto c = waitKey(1);
    if (c == 'q') break;
  }
}

void testObjectMovement(const ParamsBench &params) {
  render::WindowLessGLContext dummy(10, 10);

  fato::TrackerVX::Params tracker_params;
  tracker_params.descriptors_file = params.object_descriptors_file;
  tracker_params.model_file = params.object_model_file;
  tracker_params.image_width = width;
  tracker_params.image_height = height;
  tracker_params.fx = fx;
  tracker_params.fy = fy;
  tracker_params.cx = cx;
  tracker_params.cy = cy;
  tracker_params.parallel = false;

  RenderData data;
  useNewRenderer(params, data);
  cout << "generated frames: " << data.images.size() << endl;
  //  cout << "showing generated data " << endl;

  //  for(auto& im : data.images)
  //    {
  //      imshow("data windown", im);
  //      waitKey(10);
  //  }

  trackGeneratedData(data, tracker_params);

  //  fato::TrackerVX vx_tracker(tracker_params, std::move(matcher));

  //  pose::MultipleRigidModelsOgre &rendering_engine =
  //  *vx_tracker.ogre_renderer_;
  //  RenderData data;
  //  generateRenderedMovement(params, rendering_engine, data);

  //  const render::RigidObject &obj_model = rendering_engine.getRigidObject(0);

  //  vector<float> bbox = obj_model.getBoundingBox();

  //  cout << "ogre bbox" << endl;
  //  for (auto val : bbox) cout << val << " ";

  // trackGeneratedData(data, vx_tracker);
}

void readParameters(ParamsBench &params) {
  if (!ros::param::get("fato/model_file", params.object_model_file)) {
    throw std::runtime_error("cannot read model_file file param");
  }

  if (!ros::param::get("fato/object_descriptors_file",
                       params.object_descriptors_file)) {
    throw std::runtime_error("cannot read obj file param");
  }

  if (!ros::param::get("fato/translation_x", params.translation_x)) {
    throw std::runtime_error("cannot read translation_x param");
  }

  if (!ros::param::get("fato/translation_y", params.translation_y)) {
    throw std::runtime_error("cannot read translation_y param");
  }

  if (!ros::param::get("fato/translation_z", params.translation_z)) {
    throw std::runtime_error("cannot read translation_z param");
  }

  if (!ros::param::get("fato/rotation_x", params.rotation_x)) {
    throw std::runtime_error("cannot read rotation_x param");
  }

  if (!ros::param::get("fato/rotation_y", params.rotation_y)) {
    throw std::runtime_error("cannot read rotation_y param");
  }

  if (!ros::param::get("fato/min_scale", params.min_scale)) {
    throw std::runtime_error("cannot read min_scale param");
  }

  if (!ros::param::get("fato/max_scale", params.max_scale)) {
    throw std::runtime_error("cannot read max_scale param");
  }

  if (!ros::param::get("fato/camera_id", params.camera_id)) {
    throw std::runtime_error("cannot read camera_id param");
  }

  if (!ros::param::get("fato/framerate", params.framerate)) {
    throw std::runtime_error("cannot read framerate param");
  }

  if (!ros::param::get("fato/result_file", params.result_file)) {
    throw std::runtime_error("cannot read result_file param");
  }
}

void loadParameters(ParamsBench &params) {
  params.camera_id = 1;
  params.object_model_file =
      "/home/alessandro/projects/drone_ws/src/fato/data/ros_hydro/"
      "ros_hydro.obj";
  params.object_descriptors_file =
      "/home/alessandro/projects/drone_ws/src/fato/data/ros_hydro/"
      "ros_hydro_features.h5";
  params.translation_x = 0.0;
  params.translation_y = 0.0;
  params.translation_z = 0.00;
  params.rotation_x = 0.0;
  params.rotation_y = 0.0;
  params.min_scale = 0.0;
  params.max_scale = 0.0;
  params.framerate = 30;
  params.result_file = "/home/alessandro/debug/result.txt";
  params.running_time = 3.0;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "synthetic_benchmark");
  // ros::NodeHandle nh;

  std::cout << "renderer env variable " << render::get_resources_path() << endl;

  ParamsBench params;
  // readParameters(params);
  loadParameters(params);

  testObjectMovement(params);

  return 0;
}
