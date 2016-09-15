/*****************************************************************************/
/*  Copyright (c) 2015, Karl Pauwels                                         */
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

#include <OgreRoot.h>
#include <RenderSystems/GL/OgreGLTexture.h>
#include <cuda_gl_interop.h>
#include <GL/glx.h>

#include "../include/ogre_multi_render_target.h"


namespace render {

OgreMultiRenderTarget::OgreMultiRenderTarget(std::string name, int width,
                                             int height,
                                             Ogre::SceneManager *scene_manager
                                             )
    : width_{width},
      height_{height},
      n_rtt_textures_{6},
      name_{name},
      scene_manager_{scene_manager} {
  camera_ = scene_manager_->createCamera(name_);

  multi_render_target_ =
      Ogre::Root::getSingleton().getRenderSystem()->createMultiRenderTarget(
          name_);



  for (int t = 0; t < n_rtt_textures_; t++) {
    std::stringstream ss;
    ss << name_ << t;
    Ogre::PixelFormat format = Ogre::PF_FLOAT32_R;
    Ogre::TexturePtr rtt_texture =
        Ogre::TextureManager::getSingleton().createManual(
            ss.str(), Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
            Ogre::TEX_TYPE_2D, width, height, 0, format, Ogre::TU_RENDERTARGET);

    // Register for CUDA interop
    Ogre::GLTexturePtr gl_tex = rtt_texture.staticCast<Ogre::GLTexture>();

    GLuint id = gl_tex->getGLID();
    GLenum target = gl_tex->getGLTextureTarget();


    //    std::cout << "before register: " <<
    //cudaGetErrorString(cudaGetLastError()) << std::endl;
    cudaGraphicsResource *rtt_CUDA;
    cudaGraphicsGLRegisterImage(&rtt_CUDA, id, target,
                                cudaGraphicsRegisterFlagsReadOnly);
    //    std::cout << "after register: " <<
    //cudaGetErrorString(cudaGetLastError()) << std::endl;
    cuda_resources_.push_back(rtt_CUDA);

    Ogre::RenderTexture *render_texture =
        rtt_texture->getBuffer()->getRenderTarget();

    // doing this for debuggin purposes, still cannot see anything on the TX1
    // and have no clue why
    if(t == (int)ArrayType::texture)
    {
        std::cout << "binding rendered texture to debug camera" << std::endl;
        //render_texture->addViewport(debug_rendering->camera_);
        //render_texture->getViewport(0)->setClearEveryFrame(false);
        //render_texture->getViewport(0)->setBackgroundColour(Ogre::ColourValue::White);
        //render_texture->getViewport(0)->setOverlaysEnabled(false);
        //debug_rendering->window_->_updateViewport(render_texture->getViewport(0));
    }

    multi_render_target_->bindSurface(t, render_texture);

  }

  multi_render_target_->setAutoUpdated(true);

  Ogre::Viewport *m_viewmrt = multi_render_target_->addViewport(camera_);
  m_viewmrt->setMaterialScheme(name_);
  m_viewmrt->setClearEveryFrame(true);
  m_viewmrt->setOverlaysEnabled(false);
  m_viewmrt->setSkiesEnabled(false);
  m_viewmrt->setBackgroundColour(Ogre::ColourValue(0, 255, 0, 0));
}

OgreMultiRenderTarget::~OgreMultiRenderTarget() {
  // delete textures?
  Ogre::Root::getSingleton().getRenderSystem()->destroyRenderTarget(name_);
  scene_manager_->destroyCamera(camera_);
}

void OgreMultiRenderTarget::updateCamera(
    const Ogre::Vector3 &camera_position,
    const Ogre::Quaternion &camera_orientation,
    const Ogre::Matrix4 &projection_matrix) {
  camera_->setPosition(camera_position);
  camera_->setOrientation(camera_orientation);
  camera_->setCustomProjectionMatrix(true, projection_matrix);
  camera_->setNearClipDistance(.05);
}

void OgreMultiRenderTarget::render() { multi_render_target_->update(); }

void OgreMultiRenderTarget::render(Ogre::RenderWindow* window)
{
    multi_render_target_->update();
    window->_updateViewport(multi_render_target_->getViewport(0));
}

void OgreMultiRenderTarget::mapCudaArrays(
    std::vector<cudaArray **> cuda_arrays) {
  for (int i = 0; i < cuda_resources_.size(); i++) {
    cudaGraphicsMapResources(1, &cuda_resources_.at(i), 0);
    cudaGraphicsSubResourceGetMappedArray(cuda_arrays.at(i),
                                          cuda_resources_.at(i), 0, 0);
  }

  //  std::cout << "after map: " << cudaGetErrorString(cudaGetLastError()) <<
  // std::endl;
}

void OgreMultiRenderTarget::unmapCudaArrays() {
  for (int i = 0; i < cuda_resources_.size(); i++) {
    cudaGraphicsUnmapResources(1, &cuda_resources_.at(i), 0);
  }

  //  std::cout << "after unmap: " << cudaGetErrorString(cudaGetLastError()) <<
  // std::endl;
}

OgreRendererWindow::OgreRendererWindow(std::string name, int width, int height,
                                       std::unique_ptr<Ogre::Root>& ogre_root,
                                       Ogre::SceneManager* scene_manager):
    width_{width},
    height_{height},
    name_{name},
    scene_manager_(scene_manager)
{

 std::cout << "getting scene manager" << std::endl;
 camera_ = scene_manager_->createCamera(name_);
 window_ = ogre_root->initialise(false);
 window_ =ogre_root->createRenderWindow("debug_window", 640, 480, false);


 //Ogre::Viewport *vp = window_->addViewport(camera_);
 //vp->setClearEveryFrame(true);
 //vp->setOverlaysEnabled(true);
 //vp->setSkiesEnabled(false);
 //vp->setAutoUpdated(true);
 //vp->setBackgroundColour(Ogre::ColourValue::White);

}

OgreRendererWindow::~OgreRendererWindow() {
  // delete textures?
  Ogre::Root::getSingleton().getRenderSystem()->destroyRenderTarget(name_);
  scene_manager_->destroyCamera(camera_);
}

void OgreRendererWindow::updateCamera(const Ogre::Vector3 &camera_position, const Ogre::Quaternion &camera_orientation, const Ogre::Matrix4 &projection_matrix)
{
    camera_->setPosition(camera_position);
    camera_->setOrientation(camera_orientation);
    camera_->setCustomProjectionMatrix(true, projection_matrix);
    camera_->setNearClipDistance(.05);
}

void OgreRendererWindow::render()
{
    window_->update();
}
}
