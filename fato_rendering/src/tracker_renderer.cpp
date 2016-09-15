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

#include "../include/tracker_renderer.h"
#include "../include/AssimpLoader.h"
#include <ros/package.h>
#include <iostream>

//-------------------------------------------------------------------------------------
TrackerRenderer::TrackerRenderer(void) {}
//-------------------------------------------------------------------------------------
TrackerRenderer::~TrackerRenderer(void) {}

//-------------------------------------------------------------------------------------
void TrackerRenderer::createScene(void) {
  // create your scene here :)
  scene_manager_->setAmbientLight(Ogre::ColourValue(0.5, 0.5, 0.5));
  //  Ogre::Entity* ogreEntity = mSceneMgr->createEntity("ros_hydro_obj.mesh");
  //  Ogre::SceneNode* ogreNode =
  //      mSceneMgr->getRootSceneNode()->createChildSceneNode();
  //  ogreNode->attachObject(ogreEntity);
  Ogre::Light* light = scene_manager_->createLight("MainLight");
  light->setPosition(20, 80, 50);

  // mCamera->setPosition(0, 0, 1);
}

void TrackerRenderer::createViewports(void) {
  // Create one viewport, entire window
  Ogre::Viewport* vp = mWindow->addViewport(camera_);
  vp->setBackgroundColour(Ogre::ColourValue(0, 0, 0));

}

bool TrackerRenderer::setup(void) {
  mRoot = new Ogre::Root(mPluginsCfg);

  setupResources();

  bool carryOn = configure();
  if (!carryOn) return false;

  chooseSceneManager();
  createCamera();
  //createViewports();

  // Set default mipmap level (NB some APIs ignore this)
  Ogre::TextureManager::getSingleton().setDefaultNumMipmaps(5);

  // Create any resource listeners (for loading screens)
  createResourceListener();
  // Load resources
  loadResources();

  // Create the scene
  createScene();

  createFrameListener();

  return true;
};

bool TrackerRenderer::configure() {
  std::cout << "running this configuration" << std::endl;
  Ogre::RenderSystem* rs =
      mRoot->getRenderSystemByName("OpenGL Rendering Subsystem");
  mRoot->setRenderSystem(rs);
  mWindow = mRoot->initialise(true, "TutorialApplication Render Window");
  return true;
}

void TrackerRenderer::chooseSceneManager(void) {
  // Get the SceneManager, in this case a generic one
  scene_manager_ = mRoot->createSceneManager("DefaultSceneManager");
  mOverlaySystem = new Ogre::OverlaySystem();
  scene_manager_->addRenderQueueListener(mOverlaySystem);
}

void TrackerRenderer::createCamera() {
  camera_name_ = "RenderCam";
  camera_ = scene_manager_->createCamera(camera_name_);
  camera_position_ = Ogre::Vector3(0.0, 0.0, 0.0);
  camera_orientation_ = Ogre::Quaternion::IDENTITY;
  // convert vision (Z-forward) frame to ogre frame (Z-out)
  camera_orientation_ =
      camera_orientation_ *
      Ogre::Quaternion(Ogre::Degree(180), Ogre::Vector3::UNIT_X);

  image_width_ = 640;
  image_height_ = 480;
  updateCamera(500, 500, 320, 240, 0.01, 100);

  // Position it at 500 in Z direction

  mCameraMan = new OgreBites::SdkCameraMan(camera_);
}

void TrackerRenderer::updateCamera(double fx, double fy, double cx, double cy,
                                   double near_plane, double far_plane) {
  fx_ = fx;
  fy_ = fy;
  cx_ = cx;
  cy_ = cy;
  float zoom_x = 1;
  float zoom_y = 1;

  projection_matrix_ = Ogre::Matrix4::ZERO;
  projection_matrix_[0][0] = 2.0 * fx / (double)image_width_ * zoom_x;
  projection_matrix_[1][1] = 2.0 * fy / (double)image_height_ * zoom_y;
  projection_matrix_[0][2] = 2.0 * (0.5 - cx / (double)image_width_) * zoom_x;
  projection_matrix_[1][2] = 2.0 * (cy / (double)image_height_ - 0.5) * zoom_y;
  projection_matrix_[2][2] =
      -(far_plane + near_plane) / (far_plane - near_plane);
  projection_matrix_[2][3] =
      -2.0 * far_plane * near_plane / (far_plane - near_plane);
  projection_matrix_[3][2] = -1;

  camera_->setPosition(camera_position_);
  camera_->setOrientation(camera_orientation_);
  // Look back along -Z
  camera_->setCustomProjectionMatrix(true, projection_matrix_);
  camera_->setNearClipDistance(0.05);
}

void TrackerRenderer::go() {
  std::string package_path = ros::package::getPath(ROS_PACKAGE_NAME);

  mResourcesCfg = package_path + "/ogre_media/resources.cfg";
  mPluginsCfg = package_path + "/ogre_media/plugins.cfg";

  if (!setup()) return;

  mRoot->startRendering();

  // clean up
  destroyScene();
}

void TrackerRenderer::setupResources(void) {
  // Load resource paths from config file
  Ogre::ConfigFile cf;
  cf.load(mResourcesCfg);

  std::string package_path =
      ros::package::getPath(ROS_PACKAGE_NAME) + +"/ogre_media/";

  // Go through all sections & settings in the file
  Ogre::ConfigFile::SectionIterator seci = cf.getSectionIterator();

  Ogre::String secName, typeName, archName;
  while (seci.hasMoreElements()) {
    secName = seci.peekNextKey();
    Ogre::ConfigFile::SettingsMultiMap* settings = seci.getNext();
    Ogre::ConfigFile::SettingsMultiMap::iterator i;
    for (i = settings->begin(); i != settings->end(); ++i) {
      typeName = i->first;
      archName = i->second;
      archName = package_path + archName;

      Ogre::ResourceGroupManager::getSingleton().addResourceLocation(
          archName, typeName, secName);
    }
  }
}

void TrackerRenderer::loadResources(void)
{
    Ogre::ResourceGroupManager::getSingleton().initialiseAllResourceGroups();

    multi_render_target_ =
        Ogre::Root::getSingleton().getRenderSystem()->createMultiRenderTarget(
            camera_name_);


multi_render_target_->setAutoUpdated(true);
Ogre::Viewport* m_viewmrt = mWindow->addViewport(camera_);
//Ogre::Viewport *m_viewmrt = multi_render_target_->addViewport(camera_);
    m_viewmrt->setMaterialScheme(camera_name_);
    m_viewmrt->setClearEveryFrame(true);
    m_viewmrt->setOverlaysEnabled(true);
    m_viewmrt->setSkiesEnabled(false);
    m_viewmrt->setBackgroundColour(Ogre::ColourValue(0, 0, 0, 0));

    //scene_manager_ = Ogre::Root::getSingleton().createSceneManager("DefaultSceneManager");

}

void TrackerRenderer::addModel(std::string model_filename) {
  int segment_ind = rigid_objects_.size() + 1;
  std::string model_resource = "file://" + model_filename;
  std::cout << 1 << " " << model_resource << std::endl;
  rigid_objects_.push_back(
      std::unique_ptr<render::RigidObject>{ new render::RigidObject(
          model_resource, scene_manager_, segment_ind) });
}


void TrackerRenderer::loadModels() {
  // Set up options
  Ogre::UnaryOptionList unOpt;
  Ogre::BinaryOptionList binOpt;

  unOpt["-q"] = false;
  unOpt["-3ds_ani_fix"] = false;
  unOpt["-3ds_dae_fix"] = false;
  unOpt["-shader"] = false;
  binOpt["-log"] = "ass.log";
  binOpt["-aniName"] = "";
  binOpt["-aniSpeedMod"] = 0.0f;
  binOpt["-l"] = "";
  binOpt["-v"] = "";
  binOpt["-s"] = "Distance";
  binOpt["-p"] = "";
  binOpt["-f"] = "";

  AssimpLoader::AssOptions opts;
  opts.quietMode = false;
  opts.logFile = "ass.log";
  opts.customAnimationName = "";
  opts.dest = "";
  opts.animationSpeedModifier = 1.0;
  opts.lodValue = 250000;
  opts.lodFixed = 0;
  opts.lodPercent = 20;
  opts.numLods = 0;
  opts.usePercent = true;
  opts.source =
      "/home/alessandro/projects/drone_ws/src/fato_tracker/data/ros_hydro/"
      "ros_hydro.obj";

  AssimpLoader loader;
  loader.convert(opts);

  std::cout << "assimp model loaded" << std::endl;
}
