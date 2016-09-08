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
#ifndef TRACKER_RENDERER_H
#define TRACKER_RENDERER_H

#include <OgreCamera.h>
#include <OgreRenderTexture.h>

#include "rigid_object.h"
#include "base_render.h"

class TrackerRenderer : public BaseRenderer {
 public:
  TrackerRenderer(void);
  virtual ~TrackerRenderer(void);

  virtual void go();

  void addModel(std::string model_filename);

  void loadModels();

 protected:
  virtual void createScene(void);
  virtual bool configure(void);
  virtual void createCamera(void);
  virtual void createViewports(void);
  virtual void setupResources(void);
  virtual void loadResources(void);
  virtual bool setup(void);
  virtual void chooseSceneManager(void);


  void updateCamera(double fx, double fy, double cx, double cy,
                    double near_plane, double far_plane);

 private:
  double fx_, fy_, cx_, cy_;
  Ogre::Vector3 camera_position_;
  Ogre::Quaternion camera_orientation_;
  Ogre::Matrix4 projection_matrix_;
  int image_width_, image_height_;

  Ogre::MultiRenderTarget *multi_render_target_;

  std::string camera_name_;

  std::vector<std::unique_ptr<render::RigidObject> > rigid_objects_;
};

#endif  // TRACKER_RENDERER_H
