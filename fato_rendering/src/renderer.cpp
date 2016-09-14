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
#include "renderer.h"

#include <stdexcept>
#include <sstream>
#include <limits>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "device_1d.h"
#include "../../cuda/include/utility_kernels.h"
#include "../../cuda/include/gl_interop.h"

using namespace std;

namespace rendering {

Renderer::Renderer(int image_width, int image_height, float fx, float fy,
                   float cx, float cy, float near_plane, float far_plane)
    : width_(image_width),
      height_(image_height),
      fx_(fx),
      fy_(fy),
      cx_(cx),
      cy_(cy),
      near_plane_(near_plane),
      far_plane_(far_plane),
      camera_(glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.0, 1.0, 0.0),
              glm::vec3(0.0, 0.0, -1.0)) {
  updateProjectionMatrix(fx, fy, cx, cy, near_plane, far_plane);
}

Renderer::~Renderer() {
  glDeleteRenderbuffers(1, &color_buffer_);
  //glDeleteRenderbuffers(1, &depth_buffer_);
  glDeleteRenderbuffers(1, &depth_image_buffer_);
  // Bind to 0 -> back buffer
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDeleteFramebuffers(1, &fbo_);

  glfwTerminate();
}

void Renderer::initRenderContext(int width, int height, string screen_vs_name,
                                 string screen_frag_name) {
  // cout << "getting here" << endl;

  screen_width_ = width;
  screen_height_ = height;

  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

  window_ = glfwCreateWindow(width, height, "Opengl Window", nullptr,
                             nullptr);  // Windowed
  glfwMakeContextCurrent(window_);

  glewExperimental = GL_TRUE;
  glewInit();

  cudaGLSetGLDevice(0);

  // Define the viewport dimensions
  glViewport(0, 0, width, height);

  glEnable(GL_DEPTH_TEST);

  createCustomFramebuffer(width, height);

  initScreenTexture(screen_vs_name, screen_frag_name);

  bindCuda();
}

void Renderer::updateProjectionMatrix(float fx, float fy, float cx, float cy,
                                      float near_plane, float far_plane) {
  projection_matrix_ = glm::mat4(0.0);

  fx_ = fx;
  fy_ = fy;
  cx_ = cx;
  cy_ = cy;
  near_plane_ = near_plane;
  far_plane_ = far_plane;
  float zoom_x = 1;
  float zoom_y = 1;

  // on the contrary of OGRE, projection matrix should be row major!
  projection_matrix_[0][0] = 2.0 * fx / (float)width_ * zoom_x;
  projection_matrix_[1][1] = 2.0 * fy / (float)height_ * zoom_y;
  projection_matrix_[2][0] = 2.0 * (0.5 - cx / (float)width_) * zoom_x;
  projection_matrix_[2][1] = 2.0 * (cy / (float)height_ - 0.5) * zoom_y;
  projection_matrix_[2][2] =
      -(far_plane + near_plane) / (far_plane - near_plane);
  projection_matrix_[3][2] =
      -2.0 * far_plane * near_plane / (far_plane - near_plane);
  projection_matrix_[2][3] = -1;

  double f = (double)(far_plane);
  double n = (double)(near_plane);

  z_conv1_ = (float)((-f * n) / (f - n));
  z_conv2_ = (float)(-(f + n) / (2 * (f - n)) - 0.5);
}

void Renderer::updateProjectionMatrix(glm::mat4 projection) {
  projection_matrix_ = projection;
}

void Renderer::updateCamera(glm::vec3 position, glm::vec3 orientation,
                            glm::mat4 projection_matrix) {}

void Renderer::render() {
  //unmapCudaArrays();
  // Check and call events
  glfwPollEvents();

  glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
  // the fragment shader renders two color buffers, one with the depth buffer in
  // it
  // hack to transfer the depth buffer to cuda
  GLenum drawBuffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
  glDrawBuffers(2, drawBuffers);

  // Clear the colorbuffer
  glClearColor(0.00f, 0.00f, 0.00f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  for (RigidObject& obj : objects_) {
    if (!obj.isVisible()) continue;

    Shader& shader = obj.shader;
    // activating shader
    shader.use();

    glm::mat4 view = camera_.GetViewMatrix();
    // sending to shader custom perspective projection matrix
    glUniformMatrix4fv(glGetUniformLocation(shader.program_id_, "projection"),
                       1, GL_FALSE, glm::value_ptr(projection_matrix_));
    // sending to shader camera view matris
    glUniformMatrix4fv(glGetUniformLocation(shader.program_id_, "view"), 1,
                       GL_FALSE, glm::value_ptr(view));
    // sending to shader model matrix
    glUniformMatrix4fv(glGetUniformLocation(shader.program_id_, "model"), 1,
                       GL_FALSE, glm::value_ptr(obj.model_matrix_));
    // sending to shader parameters to calculate depth from z buffer
    glUniform1f(glGetUniformLocation(shader.program_id_, "z_conv1"), z_conv1_);
    glUniform1f(glGetUniformLocation(shader.program_id_, "z_conv2"), z_conv2_);

    // drawing the model with the current shader
    obj.model.draw(shader);
  }
  glBindFramebuffer(GL_FRAMEBUFFER, 0);  // swap with back buffer
  // render to texture to show on the screen, used for debugging purposes.
  // it is not needed
  // glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );

  renderToTexture();

  //mapCudaArrays();

  glfwSwapBuffers(window_);
}

void Renderer::addModel(string model_name, string ver_shader,
                        string frag_shader) {
  RigidObject obj(model_name, ver_shader, frag_shader);
  objects_.push_back(obj);
}

RigidObject& Renderer::getObject(int id) {
  if (id < objects_.size())
    return objects_[id];
  else
    throw runtime_error("Renderer: object id out of bound!");
}

void Renderer::downloadDepthBuffer(std::vector<float>& h_buffer) {
    int num_elems = screen_width_ * screen_height_;

//    vector<float> tmp_buffer;
//    tmp_buffer.resize(num_elems, numeric_limits<float>::quiet_NaN());
    h_buffer.resize(num_elems, numeric_limits<float>::quiet_NaN());

    glBindTexture(GL_TEXTURE_2D, depth_image_buffer_);
    // read from bound texture to CPU
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, h_buffer.data());
    // unbind texture again
    glBindTexture(GL_TEXTURE_2D, 0);

    // flipping why due to different coordinate system
//    for(auto i = 0; i < height_;++i)
//    {
//        for(auto j = 0; j < width_; ++j)
//        {
//            auto id = j + i * width_;
//            auto inv_id = j + (height_ - 1 - i) * width_;

//            h_buffer[id] = tmp_buffer[inv_id];
//        }
//    }
}

void Renderer::downloadDepthBufferCuda(std::vector<float> &h_buffer)
{
    float* d_buffer;
    h_buffer.resize(height_*width_);
    int size = sizeof(float) * height_ * width_;
    cudaError_t error = cudaMalloc((void **)&d_buffer, size);
    if (error != cudaSuccess)
      throw gpu::cudaException("Renderer::downloadDepthBuffer(243): ", error);

    gpu::downloadDepthTexture(d_buffer, *cuda_gl_depth_array_, width_, height_);

    error = cudaMemcpy(h_buffer.data(), d_buffer, size,
                                   cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
      throw gpu::cudaException("Renderer::downloadDepthBuffer(251): ", error);

    cudaFree(d_buffer);
}

void Renderer::downloadTexture(std::vector<uchar4>& h_texture) {
  util::Device1D<uchar4> d_texture(height_ * width_);

  vision::downloadTextureToRGBA(d_texture.data(), getTexture(), width_,
                                height_);
  h_texture.resize(height_ * width_);
  d_texture.copyTo(h_texture);

  cudaError err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu::cudaException("downloadTexture", err);
    std::cout << "downloadTexture(-) :" + std::string(cudaGetErrorString(err)) << std::endl;
   exit(0);
  }
}

void Renderer::downloadTextureCuda(std::vector<uchar4> &h_texture)
{
    uchar4* d_buffer;
    h_texture.resize(height_*width_);
    int size = sizeof(uchar4) * height_ * width_;
    size_t pitch;

    cudaError err = cudaGetLastError();

    if (err != cudaSuccess)
    {
      throw gpu::cudaException("Renderer::downloadTextureCuda(276): ", err);
      exit(0);
    }

    cudaError_t error = cudaMalloc((void **)&d_buffer, size);
    if (error != cudaSuccess)
      throw gpu::cudaException("Renderer::downloadTextureCuda(243): ", error);

    gpu::downloadTextureToRGBA(d_buffer, *cuda_gl_color_array_, width_, height_);

    // error = cudaMemset(d_buffer, 255, size);
    // if (error != cudaSuccess)
    //   throw gpu::cudaException("Renderer::downloadDepthBuffer(280): ", error);

    error = cudaMemcpy(h_texture.data(), d_buffer, size,
                                   cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
      throw gpu::cudaException("Renderer::downloadTextureCuda(251): ", error);

    cudaFree(d_buffer);
}

std::vector<double> Renderer::getBoundingBoxInCameraImage(
    int obj_id, Eigen::Transform<double, 3, Eigen::Affine>& pose) {
  if (obj_id < 0 || obj_id > objects_.size())
    throw std::runtime_error("Renderer: object id out of bounds\n");

  RigidObject& obj = objects_.at(obj_id);
  Eigen::Map<const Eigen::MatrixXf> bb_f(obj.getBoundingBox().data(), 3, 8);
  std::vector<double> bb_vec(3 * 8);
  Eigen::Map<Eigen::Matrix<double, 3, 8> > bounding_box(bb_vec.data());
  bounding_box = bb_f.cast<double>();
  // rotating the object to camera frame
  bounding_box = pose * bounding_box;

  // project to image
  std::vector<double> bb_pixel_vec(2 * 8);
  Eigen::Map<Eigen::Matrix<double, 2, 8> > bb_pixel(bb_pixel_vec.data());
  bb_pixel = bounding_box.block<2, 8>(0, 0);
  // X = Z*(x-ox)/fx -> x = X*fx/Z + ox
  // Y = Z*(y-oy)/fy -> y = Y*fy/Z + oy
  bb_pixel.array() /= bounding_box.row(2).replicate(2, 1).array();
  bb_pixel.row(0) *= fx_;
  bb_pixel.row(1) *= fy_;
  Eigen::Vector2d c;
  c << cx_, cy_;
  bb_pixel.colwise() += c;

  return bb_pixel_vec;
}

void Renderer::renderToTexture() {
  // First pass

  // Second pass
  glBindFramebuffer(GL_FRAMEBUFFER, 0);  // back to default
  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);

//  /cout << "before rendering depth buffer" << endl;
  screen_shader_->use();
  glBindVertexArray(screen_vao_);
  glDisable(GL_DEPTH_TEST);
  glBindTexture(GL_TEXTURE_2D, color_buffer_);
  glDrawArrays(GL_TRIANGLES, 0, 6);
  glBindVertexArray(0);
  //cout << "after rendering depth buffer" << endl;
}

void Renderer::createCustomFramebuffer(int screen_width, int screen_height) {
  // create the framebuffer
  glGenFramebuffers(1, &fbo_);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo_);

  glGenTextures(1, &color_buffer_);
  glBindTexture(GL_TEXTURE_2D, color_buffer_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, screen_width, screen_height, 0, GL_RGB,
               GL_UNSIGNED_BYTE, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  // glBindTexture(GL_TEXTURE_2D, 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         color_buffer_, 0);

  glGenTextures(1, &depth_image_buffer_);
  glBindTexture(GL_TEXTURE_2D, depth_image_buffer_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, screen_width, screen_height, 0, GL_RED,
               GL_FLOAT, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  // glBindTexture(GL_TEXTURE_2D, 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D,
                         depth_image_buffer_, 0);

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Renderer::initScreenTexture(std::string screen_vs_name,
                                 std::string screen_frag_name) {
  // creating texture covering the screen to render the custom buffer
  screen_shader_ = unique_ptr<Shader>(new Shader(
      (GLchar*)screen_vs_name.c_str(), (GLchar*)screen_frag_name.c_str()));

  // Vertex attributes for a quad that fills the
  // entire screen in Normalized Device Coordinates.
  // Positions and TexCoords
  GLfloat quadVertices[] = {-1.0f, 1.0f, 0.0f, 1.0f,  -1.0f, -1.0f,
                            0.0f,  0.0f, 1.0f, -1.0f, 1.0f,  0.0f,

                            -1.0f, 1.0f, 0.0f, 1.0f,  1.0f,  -1.0f,
                            1.0f,  0.0f, 1.0f, 1.0f,  1.0f,  1.0f};
  glGenVertexArrays(1, &screen_vao_);
  glGenBuffers(1, &screen_vbo_);
  glBindVertexArray(screen_vao_);
  glBindBuffer(GL_ARRAY_BUFFER, screen_vbo_);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices,
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat),
                        (GLvoid*)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat),
                        (GLvoid*)(2 * sizeof(GLfloat)));
  glBindVertexArray(0);
}

void Renderer::bindCuda() {
  gpuErrchk(cudaGraphicsGLRegisterImage(&color_buffer_cuda_, color_buffer_,
                                        GL_TEXTURE_2D,
                                        cudaGraphicsRegisterFlagsReadOnly));

  gpuErrchk(cudaGraphicsGLRegisterImage(&depth_buffer_cuda_,
                                        depth_image_buffer_, GL_TEXTURE_2D,
                                        cudaGraphicsRegisterFlagsReadOnly));

  //  gpuErrchk(cudaGraphicsGLRegisterBuffer(&depth_buffer_cuda_, depth_buffer_,
  //                                         cudaGraphicsRegisterFlagsReadOnly));

  cuda_gl_color_array_ = new cudaArray*;
  cuda_gl_depth_array_ = new cudaArray*;

  mapCudaArrays();

  unmapCudaArrays();
}

void Renderer::mapCudaArrays() {
  gpuErrchk(cudaGraphicsMapResources(1, &color_buffer_cuda_, 0));
  gpuErrchk(cudaGraphicsSubResourceGetMappedArray(cuda_gl_color_array_,
                                                  color_buffer_cuda_, 0, 0));

  gpuErrchk(cudaGraphicsMapResources(1, &depth_buffer_cuda_, 0));
  gpuErrchk(cudaGraphicsSubResourceGetMappedArray(cuda_gl_depth_array_,
                                                  depth_buffer_cuda_, 0, 0));
}

void Renderer::unmapCudaArrays() {
  gpuErrchk(cudaGraphicsUnmapResources(1, &color_buffer_cuda_, 0));
  gpuErrchk(cudaGraphicsUnmapResources(1, &depth_buffer_cuda_, 0));
}

cudaArray* Renderer::getTexture() { return (*cuda_gl_color_array_); }

cudaArray* Renderer::getDepthBuffer() { return (*cuda_gl_depth_array_); }


string Renderer::str(GLenum error) {
  switch (error) {
    case GL_FRAMEBUFFER_UNDEFINED:
      return "frame buffer undefined";
    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
      return "incomplete attachment";
    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
      return "missing attachment";
    case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
      return "incomplete draw buffer";
    case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
      return "incomplete read buffer";
    case GL_FRAMEBUFFER_UNSUPPORTED:
      return "frame buffer unsupported";
    case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
      return "incomplete multisample";
    case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
      return "incomplete layer";
    case GL_INVALID_ENUM:
      return "invalid enum";
    default:
      return "undefined error";
  }
}
}
