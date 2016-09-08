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
#pkg_search_module(GLFW REQUIRED)


# set this path to where OGRE is located
set(CMAKE_MODULE_PATH "/home/alessandro/ogre/lib/OGRE/cmake/;${CMAKE_MODULE_PATH} ")
# custom findASSIMP defined in the local folder
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/;${CMAKE_MODULE_PATH} ")

find_package(GLFW REQUIRED)
find_package(OGRE REQUIRED)
find_package(ASSIMP REQUIRED)
# definition required by ROS, why the hell?
add_definitions(-DASSIMP_UNIFIED_HEADER_NAMES)
find_package(OIS REQUIRED)

if(NOT OIS_FOUND)
  message(SEND_ERROR "Failed to find OIS.")
else()
  message("OIS includes "  ${OIS_INCLUDE_DIRS})
endif()

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

set(ENV_CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/env_config.cpp)
#set(ROS_DISTRO $ENV{ROS_DISTRO})
set(OGRE_PLUGIN_PATH "${OGRE_PLUGIN_DIR_REL}")
configure_file(src/env_config.cpp.in ${ENV_CONFIG_FILE} @ONLY)


catkin_package(
  INCLUDE_DIRS
    include
    ${OIS_INCLUDE_DIRS}
    ${OGRE_INCLUDE_DIRS}
    ${OGRE_RenderSystem_GL_INCLUDE_DIRS}
    ${OGRE_Overlay_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    #${GLFW_INCLUDE_DIRS}
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
    #${GLFW_LIBRARIES}
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
  ${OIS_INCLUDE_DIRS}
  ${OGRE_INCLUDE_DIRS}
  ${OGRE_RenderSystem_GL_INCLUDE_DIRS}
  ${OGRE_Overlay_INCLUDE_DIRS}
  ${urdfdom_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${GLFW_INCLUDE_DIR}
)

add_library(${PROJECT_NAME}
  src/multiple_rigid_models_ogre.cpp
  src/ogre_context.cpp
  src/ogre_multi_render_target.cpp
  src/translation_rotation_3d.cpp
  src/windowless_gl_context
  src/rigid_object.cpp
  src/mesh_loader.cpp
  src/stl_loader.cpp
  src/mesh.cpp
  src/shader.cpp
  src/renderer.cpp
  src/rigid_object_gl.cpp
  src/model.cpp
  src/camera.cpp
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
  ${assimp_LIBRARIES}
  ${OGRE_LIBRARIES}
  ${GLFW_LIBRARIES}
)

add_executable(test_opengl
  src/opengl_integration.cpp
 )

target_link_libraries(test_opengl
 ${catkin_LIBRARIES}
 ${OPENCV_LIBS}
 ${Boost_LIBRARIES}
 ${OpenCV_LIBS}
 ${GLEW_LIBRARIES}
 ${OPENGL_LIBRARIES}
 ${GLFW_LIBRARIES}
 ${assimp_LIBRARIES}
 ${PROJECT_NAME}
 )

