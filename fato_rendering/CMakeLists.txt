cmake_minimum_required(VERSION 2.8.7)
project(fato_rendering)

find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  roscpp
  resource_retriever
)

#if(TRACKER_VERBOSE_LOGGING)
#  add_definitions(-DVERBOSE_LOGGING)
#endif()

find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDA 6.5 REQUIRED)
find_package(Eigen REQUIRED)
add_definitions(${EIGEN_DEFINITIONS})
find_package(urdfdom REQUIRED)
find_package(GLEW REQUIRED)

#pkg_check_modules(GLEW REQUIRED glew)

#set(TRACKER_OGRE_LIB_DIR /home/alessandro/libs/ogre/lib)
#set(TRACKER_OGRE_INCLUDE_DIR "/home/alessandro/libs/ogre/include")
#set(TRACKER_OGRE_PLUGIN_DIR "/home/alessandro/libs/ogre/lib/OGRE")

#set(TRACKER_OGRE_LIB_DIR CACHE PATH "path to OGRE includes")
#set(TRACKER_OGRE_INCLUDE_DIR CACHE PATH "path to OGRE libraries")
#set(TRACKER_OGRE_PLUGIN_DIR CACHE PATH "path to RenderSystem_GL.so ")

#if("$TRACKER_OGRE_LIB_DIR" STREQUAL "")

#endif()

set(CMAKE_MODULE_PATH "/home/alessandro/ogre/lib/OGRE/cmake/;${CMAKE_MODULE_PATH} ")

#if (CMAKE_BUILD_TYPE STREQUAL "")
#  # CMake defaults to leaving CMAKE_BUILD_TYPE empty. This screws up
#  # differentiation between debug and release builds.
#  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
#endif ()

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/home/alessandro/ogre/lib/OGRE/cmake/")
find_package(OGRE REQUIRED)

if(OGRE_FOUND)
  message(STATUS "OGRE_INCLUDE_DIRS: " ${OGRE_INCLUDE_DIRS})
  set(OGRE_INCLUDES ${OGRE_INCLUDE_DIRS};${OGRE_INCLUDE_DIRS}/OGRE;${OGRE_INCLUDE_DIRS}/OGRE/Overlay)
  message(STATUS "OGRE_INCLUDES: " ${OGRE_INCLUDES})
  message(STATUS "OGRE_LIBRARIES: " ${OGRE_LIBRARIES})
  message(STATUS "OGRE_PLUGIN_DIR_REL: " ${OGRE_PLUGIN_DIR_REL})
  message(STATUS "OGRE_MEDIA_DIR: " ${OGRE_MEDIA_DIR})
endif()

if(${OGRE_Overlay_FOUND})
    message(STATUS "OGRE_Overlay_LIBRARIES: " ${OGRE_Overlay_LIBRARIES})
endif()

if(${OGRE_RenderSystem_GL_FOUND})
    message(STATUS "OGRE_RenderSystem_GL_LIBRARIES: " ${OGRE_RenderSystem_GL_LIBRARIES})
endif()


#set(OGRE_INCLUDES ${TRACKER_OGRE_INCLUDE_DIR}
#                  ${TRACKER_OGRE_INCLUDE_DIR}/OGRE
#                  ${TRACKER_OGRE_INCLUDE_DIR}/OGRE/Overlay
#                  )
#set(OGRE_LIBS ${TRACKER_OGRE_LIB_DIR}/libOgreMain.so
#              /usr/lib/x86_64-linux-gnu/libpthread.so
#              ${TRACKER_OGRE_PLUGIN_DIR}/RenderSystem_GL.so
#              ${TRACKER_OGRE_LIB_DIR}/libOgreOverlay.so
#              )

# path where built ogre is installed

#if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
#  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/usr/local/lib/OGRE/cmake/")
#  find_package(OGRE REQUIRED)
#else()
#  # OGRE SETUP DEFINED BY KARL
#  pkg_check_modules(OGRE_OV OGRE OGRE-Overlay)
#  #Old versions of OGRE (pre 1.9) included OGRE-Overlay in the main package
#  #i.e. there was no OGRE-Overlay component).  So if the above
#  #pkg_check_modules() failed, try looking for just OGRE.
#  if(NOT OGRE_OV_FOUND)
#    pkg_check_modules(OGRE_OV REQUIRED OGRE)
#  endif(NOT OGRE_OV_FOUND)
#  ## Find OGRE Plugin path (not necessarily platform-independent, I guess)
#  execute_process(COMMAND
#    pkg-config --variable=plugindir OGRE
#    OUTPUT_VARIABLE OGRE_PLUGIN_PATH
#    OUTPUT_STRIP_TRAILING_WHITESPACE
#  )
#endif()

set(ENV_CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/env_config.cpp)
#set(ROS_DISTRO $ENV{ROS_DISTRO})
set(OGRE_PLUGIN_PATH "${OGRE_PLUGIN_DIR_REL}")
configure_file(src/env_config.cpp.in ${ENV_CONFIG_FILE} @ONLY)

# find absolute path of ogre libraries.
# This is stored in the cache to save time on cmake re-run.
# Using absolute paths is necessary if pkg-config finds Ogre in a different
# location than the default.  This can happen when Ogre is built from source,
# or when 2 versions of Ogre are installed.  Using absolute paths ensures that
# components that link against Ogre (rviz and any packages depending on rviz)
# all find the same Ogre shared library.
#if(NOT DEFINED OGRE_OV_LIBRARIES_ABS)
#  unset(OGRE_OV_LIBRARIES_ABS_TMP)
#  foreach(_lib ${OGRE_OV_LIBRARIES})
#    set(OGRE_OV_LIB_TAG "OGRE_OV_RVIZ_LIB_${_lib}")

#    find_library(${OGRE_OV_LIB_TAG}
#            NAMES ${_lib}
#            HINTS ${OGRE_OV_LIBRARY_DIRS}
#            PATHS ${OGRE_OV_LIBRARY_DIRS}
#    )

#    set(OGRE_OV_LIBRARIES_ABS_TMP ${OGRE_OV_LIBRARIES_ABS_TMP} ${${OGRE_OV_LIB_TAG}})
#  endforeach(_lib)

#  set(OGRE_OV_LIBRARIES_ABS ${OGRE_OV_LIBRARIES_ABS_TMP}
#        CACHE FILEPATH "Pathname of library ${_lib}")
#endif(NOT DEFINED OGRE_OV_LIBRARIES_ABS)
## add gl rendersystem
#list(APPEND OGRE_OV_LIBRARIES_ABS
#  ${OGRE_PLUGIN_PATH}/RenderSystem_GL.so
#)
#message(STATUS "OGRE_OV_LIBRARIES_ABS=${OGRE_OV_LIBRARIES_ABS}")

# OGRE SETUP

find_package(ASSIMP QUIET)
if (NOT ASSIMP_FOUND)
  pkg_check_modules(ASSIMP assimp)
endif()
if (ASSIMP_FOUND)
  if( NOT ${ASSIMP_VERSION} VERSION_LESS "2.0.1150" )
    add_definitions(-DASSIMP_UNIFIED_HEADER_NAMES)
    message(STATUS "Assimp version has unified headers")
  else()
    message(STATUS "Assimp version does not have unified headers")
  endif()
  include_directories(${ASSIMP_INCLUDE_DIRS})
  link_directories(${ASSIMP_LIBRARY_DIRS})
else()
  message(STATUS "could not find assimp (perhaps available through ROS package?), so assimping assimp v2")
  set(ASSIMP_LIBRARIES assimp)
  set(ASSIMP_LIBRARY_DIRS)
  set(ASSIMP_CXX_FLAGS)
  set(ASSIMP_CFLAGS_OTHER)
  set(ASSIMP_LINK_FLAGS)
  set(ASSIMP_INCLUDE_DIRS)
endif()

catkin_package(
  INCLUDE_DIRS
    include
    ${OGRE_INCLUDE_DIRS}
    ${OGRE_RenderSystem_GL_INCLUDE_DIRS}
    ${OGRE_Overlay_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
  DEPENDS
    libx11-dev
    opengl
    libglew-dev
  LIBRARIES
    ${PROJECT_NAME}
    ${OGRE_LIBRARIES}
    ${OGRE_RenderSystem_GL_LIBRARIES}
    ${OGRE_Overlay_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${urdfdom_LIBRARIES}
  CATKIN_DEPENDS
    fato_cuda
    utilities
)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# show additional files in qtcreator
execute_process(COMMAND find ${PROJECT_SOURCE_DIR} -type f
  OUTPUT_VARIABLE FILES_STRING)
string(REPLACE "\n" ";" FILES_LIST ${FILES_STRING})
add_custom_target(additional_files_${PROJECT_NAME} SOURCES
  ${FILES_LIST}
  ${PROJECT_SOURCE_DIR}/../README.md
)

include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIR}
  ${OGRE_INCLUDE_DIRS}
  ${OGRE_RenderSystem_GL_INCLUDE_DIRS}
  ${OGRE_Overlay_INCLUDE_DIRS}
  ${urdfdom_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

#add_executable(dummy_rendering
#    src/dummy.cpp)

#message("OGRE_Overlay_LIBRARIES: " ${OGRE_Overlay_LIBRARIES})
#message("OGRE_RenderSystem_GL_LIBRARIES: " ${OGRE_RenderSystem_GL_LIBRARIES})


#target_link_libraries(dummy_rendering
#  ${OGRE_RenderSystem_GL_LIBRARIES} ${OGRE_Overlay_LIBRARIES} ${OGRE_LIBRARIES}
#)

add_library(${PROJECT_NAME}
  src/multiple_rigid_models_ogre.cpp
  src/ogre_context.cpp
  src/ogre_multi_render_target.cpp
  src/translation_rotation_3d.cpp
  src/windowless_gl_context
  src/rigid_object.cpp
  src/mesh_loader.cpp
  src/stl_loader.cpp
  ${ENV_CONFIG_FILE}
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${GLEW_LIBRARIES}
  ${OPENGL_LIBRARIES}
  ${catkin_LIBRARIES}
  ${OGRE_RenderSystem_GL_LIBRARIES}
  ${OGRE_Overlay_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${urdfdom_LIBRARIES}
  assimp
  ${OGRE_LIBRARIES}
)
#target_link_libraries(${PROJECT_NAME}
#  ${catkin_LIBRARIES}
#  ${OpenCV_LIBS}
#  ${OGRE_Overlay_LIBRARIES}
#  assimp
#)

#target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OGRE_LIBRARIES} ${OGRE_Overlay_LIBRARIES} ${OGRE_RenderSystem_GL_LIBRARIES} assimp ${GLEW_LIBRARIES})

