project(tracker_cpu)

cmake_minimum_required(VERSION 2.8.7)

find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  clustering
  utilities
  image_geometry
  roscpp
)

if(TRACKER_VERBOSE_LOGGING)
  add_definitions(-DVERBOSE_LOGGING)
endif()

find_package(OpenCV REQUIRED)

find_package(Eigen REQUIRED)
add_definitions(${EIGEN_DEFINITIONS})

catkin_package(
  INCLUDE_DIRS include
  DEPENDS image_geometry
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS utilities clustering
)

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -march=native")

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
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIR}
)

add_library(${PROJECT_NAME}
  src/matcher.cpp
  src/pose_estimation.cpp
  src/feature_matcher.cpp
  src/tracker_2d.cpp
  src/tracker_model.cpp
  src/tracker_2d_v2.cpp
  src/tracker_3d.cpp
  src/bounding_cube.cpp
  src/model.cpp
  src/config.cpp
  src/utilities.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
)
