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

using namespace std;
using namespace cv;

// default parameters
int width = 640;
int height = 480;
double fx = 500.0;
double fy = 500.0;
double cx = width / 2.0;
double cy = height / 2.0;
double near_plane = 0.01;   // for init only
double far_plane = 1000.0;  // for init only
Eigen::Matrix<double, 3, 8> bounding_box;

Eigen::Matrix3d getRotationView(float rot_x, float rot_y) {
  Eigen::Matrix3d rot_view;
  rot_view = Eigen::AngleAxisd(rot_y, Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(rot_x, Eigen::Vector3d::UnitX());
  // rotate around x to 'render' just as Ogre
  Eigen::Matrix3d Rx_180 = Eigen::Matrix<double, 3, 3>::Identity();
  Rx_180(1, 1) = -1.0;
  Rx_180(2, 2) = -1.0;
  rot_view *= Rx_180;

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

int isOut(vector<double>& box)
{
    for(int i = 0; i < 8;++i)
    {
        if(box[2*i] < 0)
            return 1;
        if(box[2*i] > width)
            return 1;
        if(box[2*i+1] < 0)
            return 2;
        if(box[2*i+1] > height)
            return 2;
    }

    return 0;
}

void testObjectMovement() {
  render::WindowLessGLContext dummy(10, 10);

  string filename =
      "/home/alessandro/projects/drone_ws/src/fato_tracker/data/kellog/"
      "kellog.obj";

  pose::MultipleRigidModelsOgre rendering_engine(width, height, fx, fy, cx, cy,
                                                 near_plane, far_plane);

  rendering_engine.addModel(filename);

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
  // getPose(0, 0, tra_center, t_render);
  // tra_z_shift * rot_view * tra_center;

  Eigen::Matrix3d rot_view = getRotationView(0, 0);

  double z_shift = getZshif(rot_view, 0.6, tra_center);
  Eigen::Translation<double, 3> tra_z_shift(0, 0, z_shift);
  // compose render transform (tra_center -> rot_view -> tra_z_shift)
  t_render = tra_z_shift * rot_view * tra_center;

  cout << fixed << setprecision(3) << t_render.translation() << "\n"
       << t_render.rotation() << endl;

  // t_render.translate()

  renderObject(t_render, rendering_engine);

  std::vector<uchar4> h_texture(height * width);
  downloadRenderedImg(rendering_engine, h_texture);

  cv::Mat img_rgba(height, width, CV_8UC4, h_texture.data());
  cv::Mat img_gray;
  cv::cvtColor(img_rgba, img_gray, CV_RGBA2BGR);

  vector<vector<double>> boxes;
  getTransformedObjectBox(t_render, rendering_engine, boxes);

  drawBox(boxes[0], img_gray);

  float x_vel = 30; // velocity per second
  float y_vel = 20;

  int framerate = 30;

  float x_pos = 0;
  float y_pos = 0;

  while(1)
  {
      x_pos += x_vel/float(framerate);
      y_pos += y_vel/float(framerate);

      Eigen::Translation<double, 3> tra_z_shift(x_pos, y_pos, z_shift);
      t_render = tra_z_shift * rot_view * tra_center;
      renderObject(t_render, rendering_engine);
      getTransformedObjectBox(t_render, rendering_engine, boxes);

      if(isOut(boxes[0]) == 1)
          x_vel = x_vel * - 1;
      if(isOut(boxes[0]) == 2)
          y_vel = y_vel * -1;


      downloadRenderedImg(rendering_engine, h_texture);

      cv::Mat img_rgba(height, width, CV_8UC4, h_texture.data());
      cv::Mat img_gray;
      cv::cvtColor(img_rgba, img_gray, CV_RGBA2BGR);

      imshow("debug", img_gray);
      auto c = waitKey(framerate);
      if(c == 'q')
          break;
  }


  //  for(auto i = 0; i <8; ++i)
  //  {
  //      cout << fixed << setprecision(3) << tra_center.x() << " " <<
  //      tra_center.y() << " " << tra_center.z()  << endl;
  //  }

  //  vector<double> box = bboxes[0];
  //  cout << bboxes.size() << " " << box.size() << endl;

  //  Point2d pt0(box[0],box[1]);
  //  Point2d pt1(box[2],box[3]);
  //  Point2d pt2(box[4],box[5]);
  //  Point2d pt3(box[6],box[7]);
  //  Point2d pt4(box[8],box[9]);
  //  Point2d pt5(box[10],box[11]);
  //  Point2d pt6(box[12],box[13]);
  //  Point2d pt7(box[014],box[15]);

  //  for(int i = 0; i < 4; i++)
  //  {
  //      circle(rendered_image, Point2f(box[2*i],box[2*i+1]), 3,
  //      Scalar(255,0,0), 1);
  //  }

  //  for(int i = 0; i < 4; i++)
  //  {
  //      circle(rendered_image, Point2f(box[2*i+4],box[2*i+1+4]), 3,
  //      Scalar(255,0,255), 1);
  //  }

  //  for(int i = 0; i < 8; ++i)
  //  {
  //      cout << box[2*i] << " " << box[2*i+1] << endl;
  //  }

  //  line(rendered_image, pt0, pt2, Scalar(0,255,0), 1);
  //  line(rendered_image, pt2, pt6, Scalar(0,255,0), 1);
  //  line(rendered_image, pt6, pt4, Scalar(0,255,0), 1);
  //  line(rendered_image, pt4, pt0, Scalar(0,255,0), 1);

  imshow("debug", img_gray);
  waitKey(0);
}

int main(int argc, char **argv) {
  vx_context context = vxCreateContext();

  // synthgraph_test(context);
  // realgraph_test(context);
  // gl2Vx(context);
  // synthgraph_test2(context);

  testObjectMovement();

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
