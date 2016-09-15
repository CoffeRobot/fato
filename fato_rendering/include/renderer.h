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
#ifndef RENDERER_H
#define RENDERER_H

#include "glm/glm.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define GLEW_STATIC
#include <GL/glew.h>

#include <GLFW/glfw3.h>

#include <vector>
#include <string>
#include <memory>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "camera.h"
#include "model.h"
#include "shader.h"
#include "rigid_object_gl.h"

namespace rendering {

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class Renderer {
 public:
  Renderer(int image_width, int image_height, float fx, float fy, float cx,
           float cy, float near_plane, float far_plane);

  ~Renderer();

  void initRenderContext(int width = 640, int height = 480,
                         std::string screen_vs_name = "",
                         std::string screen_frag_name = "");

  void updateProjectionMatrix(float fx, float fy, float cx, float cy,
                              float near_plane, float far_plane);

  void updateProjectionMatrix(glm::mat4 projection);

  void updateCamera(glm::vec3 position, glm::vec3 orientation,
                    glm::mat4 projection_matrix);
  void render();

  void addModel(std::string model_name, std::string ver_shader,
                std::string frag_shader);

  RigidObject& getObject(int id);

  void downloadDepthBuffer(std::vector<float>& h_buffer);

  void downloadDepthBufferCuda(std::vector<float>& h_buffer);

  void downloadTexture(std::vector<uchar4>& h_texture);

  void downloadTextureCuda(std::vector<uchar4>& h_texture);

  std::vector<double> getBoundingBoxInCameraImage(
      int obj_id, Eigen::Transform<double, 3, Eigen::Affine>& pose);


  cudaArray* getTexture();
  cudaArray* getDepthBuffer();

  std::vector<RigidObject> objects_;
  glm::mat4 projection_matrix_;
  Camera camera_;

  GLFWwindow* window_;

 private:
  void renderToTexture();

  void createCustomFramebuffer(int screen_width, int screen_height);

  void initScreenTexture(std::string screen_vs_name,
                         std::string screen_frag_name);

  void bindCuda();

  void mapCudaArrays();

  void unmapCudaArrays();


  int width_, height_, screen_width_, screen_height_;
  float fx_, fy_, cx_, cy_;
  float near_plane_, far_plane_;
  float z_conv1_, z_conv2_;

  GLuint fbo_, color_buffer_, screen_vao_, screen_vbo_, depth_image_buffer_;
  //GLuint depth_buffer_;

  cudaGraphicsResource* color_buffer_cuda_;
  cudaGraphicsResource* depth_buffer_cuda_;
  cudaArray** cuda_gl_color_array_;
  cudaArray** cuda_gl_depth_array_;

  std::unique_ptr<Shader> screen_shader_;

  std::string str(GLenum error);

};

}  // end namespace

#endif  // RENDERER_H
