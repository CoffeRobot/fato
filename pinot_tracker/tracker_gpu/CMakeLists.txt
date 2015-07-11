project(tracker_gpu)
cmake_minimum_required(VERSION 2.8.7)

set(WITH_GPU OFF CACHE BOOL "set to true if opencv gpu is installed")

#LINK_DIRECTORIES("/home/alessandro/Libs/lib")

# OPENCV find package finds always the ros installation no matter what flag is
# set, this part requires cuda compilation therefore it needs to be manually set

#find_package(OpenCV REQUIRED NO_DEFAULT_PATH PATHS "/home/alessandro/Libs")

#if(OpenCV_FOUND)
#  message(status: ": Opencv path: ${OpenCV_INCLUDE_DIRS}")
#  message(status: ": Opencv libs: ${OpenCV_LIBS}")
#endif()

#link_directories()

find_package(CUDA 6.5 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  clustering
  utilities
)

catkin_package(
  INCLUDE_DIRS include ${CUDA_INCLUDE_DIRS}
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS utilities clustering
)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native")

# show additional files in qtcreator
execute_process(COMMAND find ${PROJECT_SOURCE_DIR} -type f
  OUTPUT_VARIABLE FILES_STRING)
string(REPLACE "\n" ";" FILES_LIST ${FILES_STRING})
add_custom_target(additional_files_${PROJECT_NAME} SOURCES
  ${FILES_LIST}
  ${PROJECT_SOURCE_DIR}/../../README.md
)

include_directories(include
  ${catkin_INCLUDE_DIRS}
  #${OpenCV_INCLUDE_DIRS}
  "/home/alessandro/Libs/inlcude"
  "/home/alessandro/Libs/include/opencv"
  ${CUDA_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/matcher.cpp
  src/tracker.cpp
)

# super ugly, linking manually to the gpu library. No other way found right now
# to override ros opencv path
target_link_libraries(${PROJECT_NAME}
  /home/alessandro/Libs/lib/libopencv_gpu.so
  opencv_highgui
  opencv_gpu
  opencv_flann
  opencv_features2d
  opencv_core
  opencv_contrib
  opencv_calib3d
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
)
