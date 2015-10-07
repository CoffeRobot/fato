project(tracker_gpu)
cmake_minimum_required(VERSION 2.8.7)

set(TRACKER_WITH_GPU OFF CACHE BOOL "set to true if opencv gpu is installed")
set(TRACKER_CUSTON_OPENCV_PATH CACHE PATH "path to gpu compiled opencv")

#link_directories()
if(TRACKER_WITH_GPU)
  if ("${TRACKER_CUSTON_OPENCV_PATH}" STREQUAL "")
    message(FATAL_ERROR "please set the path pointing to opencv compiled with cuda")
  endif()
  find_package(CUDA REQUIRED)
endif()

find_package(catkin REQUIRED COMPONENTS
  clustering
  utilities
)

catkin_package(
  INCLUDE_DIRS include ${CUDA_INCLUDE_DIRS}
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS utilities clustering
)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -march=native")

execute_process(COMMAND find ${PROJECT_SOURCE_DIR} -type f
  OUTPUT_VARIABLE FILES_STRING)
string(REPLACE "\n" ";" FILES_LIST ${FILES_STRING})
add_custom_target(additional_files_${PROJECT_NAME} SOURCES
  ${FILES_LIST}
  ${PROJECT_SOURCE_DIR}/../../README.md
)

if(TRACKER_WITH_GPU)

  include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${CUSTON_OPENCV_PATH}/include
  ${CUDA_INCLUDE_DIRS}
  )

  add_library(${PROJECT_NAME}
  src/matcher.cpp
  src/tracker.cpp
  )

  target_link_libraries(${PROJECT_NAME}
  ${CUSTON_OPENCV_PATH}/lib/opencv_gpu
  ${CUSTON_OPENCV_PATH}/lib/opencv_highgui
  ${CUSTON_OPENCV_PATH}/lib/opencv_gpu
  ${CUSTON_OPENCV_PATH}/lib/opencv_flann
  ${CUSTON_OPENCV_PATH}/lib/opencv_features2d
  ${CUSTON_OPENCV_PATH}/lib/opencv_core
  ${CUSTON_OPENCV_PATH}/lib/opencv_contrib
  ${CUSTON_OPENCV_PATH}/lib/opencv_calib3d
  ${CUDA_LIBRARIES}
  )
endif()


# super ugly, linking manually to the gpu library. No other way found right now
# to override ros opencv path

