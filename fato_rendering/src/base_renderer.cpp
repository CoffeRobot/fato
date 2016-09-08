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

#include "../include/base_render.h"
#include <OgreRoot.h>
#include <RenderSystems/GL/OgreGLTexture.h>
#include "AssimpLoader.h"
#include <iostream>
//#include <cuda_gl_interop.h>

//-------------------------------------------------------------------------------------
BaseRenderer::BaseRenderer()
    : mRoot(0),
      camera_(0),
      scene_manager_(0),
      mWindow(0),
      mResourcesCfg(Ogre::StringUtil::BLANK),
      mPluginsCfg(Ogre::StringUtil::BLANK),
      mTrayMgr(0),
      mCameraMan(0),
      mDetailsPanel(0),
      mCursorWasVisible(false),
      mShutDown(false),
      mInputManager(0),
      mMouse(0),
      mKeyboard(0) {}

//-------------------------------------------------------------------------------------
BaseRenderer::~BaseRenderer(void) {
  if (mTrayMgr) delete mTrayMgr;
  if (mCameraMan) delete mCameraMan;

  // Remove ourself as a Window listener
  Ogre::WindowEventUtilities::removeWindowEventListener(mWindow, this);
  windowClosed(mWindow);
  delete mRoot;
}

//-------------------------------------------------------------------------------------
bool BaseRenderer::configure(void) {
  // Show the configuration dialog and initialise the system
  // You can skip this and use root.restoreConfig() to load configuration
  // settings if you were sure there are valid ones saved in ogre.cfg
  if (mRoot->showConfigDialog()) {
    // If returned true, user clicked OK so initialise
    // Here we choose to let the system create a default rendering window by
    // passing 'true'
    mWindow = mRoot->initialise(true, "TutorialApplication Render Window");

    return true;
  } else {
    return false;
  }
}
//-------------------------------------------------------------------------------------
void BaseRenderer::chooseSceneManager(void) {
  // Get the SceneManager, in this case a generic one
  scene_manager_ = mRoot->createSceneManager(Ogre::ST_GENERIC);
  mOverlaySystem = new Ogre::OverlaySystem();
  scene_manager_->addRenderQueueListener(mOverlaySystem);
}
//-------------------------------------------------------------------------------------
void BaseRenderer::createCamera(void) {
  // Create the camera
  camera_ = scene_manager_->createCamera("PlayerCam");

  // Position it at 500 in Z direction
  camera_->setPosition(Ogre::Vector3(0, 0, 80));
  // Look back along -Z
  camera_->lookAt(Ogre::Vector3(0, 0, -300));
  camera_->setNearClipDistance(5);

  mCameraMan = new OgreBites::SdkCameraMan(
      camera_);  // create a default camera controller
}
//-------------------------------------------------------------------------------------
void BaseRenderer::createFrameListener(void) {
  Ogre::LogManager::getSingletonPtr()->logMessage("*** Initializing OIS ***");
  OIS::ParamList pl;
  size_t windowHnd = 0;
  std::ostringstream windowHndStr;

  mWindow->getCustomAttribute("WINDOW", &windowHnd);
  windowHndStr << windowHnd;
  pl.insert(std::make_pair(std::string("WINDOW"), windowHndStr.str()));

  mInputManager = OIS::InputManager::createInputSystem(pl);

  mKeyboard = static_cast<OIS::Keyboard *>(
      mInputManager->createInputObject(OIS::OISKeyboard, true));
  mMouse = static_cast<OIS::Mouse *>(
      mInputManager->createInputObject(OIS::OISMouse, true));

  mMouse->setEventCallback(this);
  mKeyboard->setEventCallback(this);

  // Set initial mouse clipping size
  windowResized(mWindow);

  // Register as a Window listener
  Ogre::WindowEventUtilities::addWindowEventListener(mWindow, this);

  OgreBites::InputContext inputContext;
  inputContext.mMouse = mMouse;
  inputContext.mKeyboard = mKeyboard;
  mTrayMgr = new OgreBites::SdkTrayManager("InterfaceName", mWindow,
                                           inputContext, this);
  mTrayMgr->showFrameStats(OgreBites::TL_BOTTOMLEFT);
  mTrayMgr->showLogo(OgreBites::TL_BOTTOMRIGHT);
  mTrayMgr->hideCursor();

  // create a params panel for displaying sample details
  Ogre::StringVector items;
  items.push_back("cam.pX");
  items.push_back("cam.pY");
  items.push_back("cam.pZ");
  items.push_back("");
  items.push_back("cam.oW");
  items.push_back("cam.oX");
  items.push_back("cam.oY");
  items.push_back("cam.oZ");
  items.push_back("");
  items.push_back("Filtering");
  items.push_back("Poly Mode");

  mDetailsPanel = mTrayMgr->createParamsPanel(OgreBites::TL_NONE,
                                              "DetailsPanel", 200, items);
  mDetailsPanel->setParamValue(9, "Bilinear");
  mDetailsPanel->setParamValue(10, "Solid");
  mDetailsPanel->hide();

  mRoot->addFrameListener(this);
}
//-------------------------------------------------------------------------------------
void BaseRenderer::destroyScene(void) {}
//-------------------------------------------------------------------------------------
void BaseRenderer::createViewports(void) {
  // Create one viewport, entire window
  Ogre::Viewport *vp = mWindow->addViewport(camera_);
  vp->setBackgroundColour(Ogre::ColourValue(0, 0, 0));

  // Alter the camera aspect ratio to match the viewport
  camera_->setAspectRatio(Ogre::Real(vp->getActualWidth()) /
                          Ogre::Real(vp->getActualHeight()));

}
//-------------------------------------------------------------------------------------
void BaseRenderer::setupResources(void) {
  // Load resource paths from config file
  Ogre::ConfigFile cf;
  cf.load(mResourcesCfg);

  // Go through all sections & settings in the file
  Ogre::ConfigFile::SectionIterator seci = cf.getSectionIterator();

  Ogre::String secName, typeName, archName;
  while (seci.hasMoreElements()) {
    secName = seci.peekNextKey();
    Ogre::ConfigFile::SettingsMultiMap *settings = seci.getNext();
    Ogre::ConfigFile::SettingsMultiMap::iterator i;
    for (i = settings->begin(); i != settings->end(); ++i) {
      typeName = i->first;
      archName = i->second;
      Ogre::ResourceGroupManager::getSingleton().addResourceLocation(
          archName, typeName, secName);

    }
  }
}
//-------------------------------------------------------------------------------------
void BaseRenderer::createResourceListener(void) {}
//-------------------------------------------------------------------------------------
void BaseRenderer::loadResources(void) {
  Ogre::ResourceGroupManager::getSingleton().initialiseAllResourceGroups();
}
//-------------------------------------------------------------------------------------
void BaseRenderer::go(void) {
#ifdef _DEBUG
  mResourcesCfg = "resources_d.cfg";
  mPluginsCfg = "plugins_d.cfg";
#else
  mResourcesCfg = "resources.cfg";
  mPluginsCfg = "plugins.cfg";
#endif

  if (!setup()) return;

  mRoot->startRendering();

  // clean up
  destroyScene();
}
//-------------------------------------------------------------------------------------
bool BaseRenderer::setup(void) {
  mRoot = new Ogre::Root(mPluginsCfg);

  setupResources();

  bool carryOn = configure();
  if (!carryOn) return false;

  chooseSceneManager();
  createCamera();
  createViewports();

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

//-------------------------------------------------------------------------------------
bool BaseRenderer::frameRenderingQueued(const Ogre::FrameEvent &evt) {
  if (mWindow->isClosed()) return false;

  if (mShutDown) return false;

  // Need to capture/update each device
  mKeyboard->capture();
  mMouse->capture();

  mTrayMgr->frameRenderingQueued(evt);

  if (!mTrayMgr->isDialogVisible()) {
    mCameraMan->frameRenderingQueued(
        evt);  // if dialog isn't up, then update the camera
    if (mDetailsPanel->isVisible())  // if details panel is visible, then update
                                     // its contents
    {
      mDetailsPanel->setParamValue(
          0, Ogre::StringConverter::toString(camera_->getDerivedPosition().x));
      mDetailsPanel->setParamValue(
          1, Ogre::StringConverter::toString(camera_->getDerivedPosition().y));
      mDetailsPanel->setParamValue(
          2, Ogre::StringConverter::toString(camera_->getDerivedPosition().z));
      mDetailsPanel->setParamValue(4, Ogre::StringConverter::toString(
                                          camera_->getDerivedOrientation().w));
      mDetailsPanel->setParamValue(5, Ogre::StringConverter::toString(
                                          camera_->getDerivedOrientation().x));
      mDetailsPanel->setParamValue(6, Ogre::StringConverter::toString(
                                          camera_->getDerivedOrientation().y));
      mDetailsPanel->setParamValue(7, Ogre::StringConverter::toString(
                                          camera_->getDerivedOrientation().z));
    }
  }

  return true;
}
//-------------------------------------------------------------------------------------
bool BaseRenderer::keyPressed(const OIS::KeyEvent &arg) {
  if (mTrayMgr->isDialogVisible())
    return true;  // don't process any more keys if dialog is up

  if (arg.key == OIS::KC_F)  // toggle visibility of advanced frame stats
  {
    mTrayMgr->toggleAdvancedFrameStats();
  } else if (arg.key ==
             OIS::KC_G)  // toggle visibility of even rarer debugging details
  {
    if (mDetailsPanel->getTrayLocation() == OgreBites::TL_NONE) {
      mTrayMgr->moveWidgetToTray(mDetailsPanel, OgreBites::TL_TOPRIGHT, 0);
      mDetailsPanel->show();
    } else {
      mTrayMgr->removeWidgetFromTray(mDetailsPanel);
      mDetailsPanel->hide();
    }
  } else if (arg.key == OIS::KC_T)  // cycle polygon rendering mode
  {
    Ogre::String newVal;
    Ogre::TextureFilterOptions tfo;
    unsigned int aniso;

    switch (mDetailsPanel->getParamValue(9).asUTF8()[0]) {
      case 'B':
        newVal = "Trilinear";
        tfo = Ogre::TFO_TRILINEAR;
        aniso = 1;
        break;
      case 'T':
        newVal = "Anisotropic";
        tfo = Ogre::TFO_ANISOTROPIC;
        aniso = 8;
        break;
      case 'A':
        newVal = "None";
        tfo = Ogre::TFO_NONE;
        aniso = 1;
        break;
      default:
        newVal = "Bilinear";
        tfo = Ogre::TFO_BILINEAR;
        aniso = 1;
    }

    Ogre::MaterialManager::getSingleton().setDefaultTextureFiltering(tfo);
    Ogre::MaterialManager::getSingleton().setDefaultAnisotropy(aniso);
    mDetailsPanel->setParamValue(9, newVal);
  } else if (arg.key == OIS::KC_R)  // cycle polygon rendering mode
  {
    Ogre::String newVal;
    Ogre::PolygonMode pm;

    switch (camera_->getPolygonMode()) {
      case Ogre::PM_SOLID:
        newVal = "Wireframe";
        pm = Ogre::PM_WIREFRAME;
        break;
      case Ogre::PM_WIREFRAME:
        newVal = "Points";
        pm = Ogre::PM_POINTS;
        break;
      default:
        newVal = "Solid";
        pm = Ogre::PM_SOLID;
    }

    camera_->setPolygonMode(pm);
    mDetailsPanel->setParamValue(10, newVal);
  } else if (arg.key == OIS::KC_F5)  // refresh all textures
  {
    Ogre::TextureManager::getSingleton().reloadAll();
  } else if (arg.key == OIS::KC_SYSRQ)  // take a screenshot
  {
    mWindow->writeContentsToTimestampedFile("screenshot", ".jpg");
  } else if (arg.key == OIS::KC_ESCAPE) {
    mShutDown = true;
  }

  mCameraMan->injectKeyDown(arg);
  return true;
}

bool BaseRenderer::keyReleased(const OIS::KeyEvent &arg) {
  mCameraMan->injectKeyUp(arg);
  return true;
}

bool BaseRenderer::mouseMoved(const OIS::MouseEvent &arg) {
  if (mTrayMgr->injectMouseMove(arg)) return true;
  mCameraMan->injectMouseMove(arg);
  return true;
}

bool BaseRenderer::mousePressed(const OIS::MouseEvent &arg,
                                   OIS::MouseButtonID id) {
  if (mTrayMgr->injectMouseDown(arg, id)) return true;
  mCameraMan->injectMouseDown(arg, id);
  return true;
}

bool BaseRenderer::mouseReleased(const OIS::MouseEvent &arg,
                                    OIS::MouseButtonID id) {
  if (mTrayMgr->injectMouseUp(arg, id)) return true;
  mCameraMan->injectMouseUp(arg, id);
  return true;
}

// Adjust mouse clipping area
void BaseRenderer::windowResized(Ogre::RenderWindow *rw) {
  unsigned int width, height, depth;
  int left, top;
  rw->getMetrics(width, height, depth, left, top);

  const OIS::MouseState &ms = mMouse->getMouseState();
  ms.width = width;
  ms.height = height;
}

// Unattach OIS before window shutdown (very important under Linux)
void BaseRenderer::windowClosed(Ogre::RenderWindow *rw) {
  // Only close for window that created OIS (the main window in these demos)
  if (rw == mWindow) {
    if (mInputManager) {
      mInputManager->destroyInputObject(mMouse);
      mInputManager->destroyInputObject(mKeyboard);

      OIS::InputManager::destroyInputSystem(mInputManager);
      mInputManager = 0;
    }
  }
}

