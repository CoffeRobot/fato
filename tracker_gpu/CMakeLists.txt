project(tracker_gpu)
cmake_minimum_required(VERSION 2.8.7)

#set(TRACKER_WITH_GPU OFF CACHE BOOL "set to true if opencv gpu is installed")
#set(TRACKER_CUSTON_OPENCV_PATH CACHE PATH "path to gpu compiled opencv")

set(TRACKER_WITH_GPU OFF CACHE BOOL "set to true if opencv gpu is installed")
set(TRACKER_CUSTON_OPENCV_PATH CACHE PATH "path to gpu compiled opencv")
set(VERBOSE_LOGGING ON CACHE BOOL "set to true if opencv gpu is installed")


#link_directories()
if(TRACKER_WITH_GPU)
  find_package(CUDA REQUIRED)
  if ("${TRACKER_CUSTON_OPENCV_PATH}" STREQUAL "")
      message(FATAL_ERROR "please set the path pointing to opencv compiled with cuda")
  else()
      message(INFO "custom opencv path: ${TRACKER_CUSTON_OPENCV_PATH}")
      set(TRACKER_WITH_CUSTOM_OPENCV ON PARENT_SCOPE)
      set(CUSTOM_OPENCV_LIBS
        ${TRACKER_CUSTON_OPENCV_PATH}/lib/libopencv_highgui.so
        ${TRACKER_CUSTON_OPENCV_PATH}/lib/libopencv_gpu.so
        ${TRACKER_CUSTON_OPENCV_PATH}/lib/libopencv_flann.so
        ${TRACKER_CUSTON_OPENCV_PATH}/lib/libopencv_features2d.so
        ${TRACKER_CUSTON_OPENCV_PATH}/lib/libopencv_core.so
        ${TRACKER_CUSTON_OPENCV_PATH}/lib/libopencv_contrib.so
        ${TRACKER_CUSTON_OPENCV_PATH}/lib/libopencv_calib3d.so
        ${TRACKER_CUSTON_OPENCV_PATH}/lib/libopencv_nonfree.so
      )
  endif()
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
  ${PROJECT_SOURCE_DIR}/../README.md
)

if(TRACKER_WITH_GPU)

  include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${TRACKER_CUSTON_OPENCV_PATH}/include
  ${CUDA_INCLUDE_DIRS}
  )

  add_library(${PROJECT_NAME}
  src/matcher.cpp
  src/tracker.cpp
  )

  message(INFO " tracker_gpu: opencv libs ${CUSTOM_OPENCV_LIBS}")

  target_link_libraries(${PROJECT_NAME}
  ${CUSTOM_OPENCV_LIBS}
  ${CUDA_LIBRARIES}
  )
endif()


