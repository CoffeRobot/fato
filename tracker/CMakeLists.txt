cmake_minimum_required(VERSION 2.8.7)
project(tracker)

find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  clustering
  utilities
  fato_rendering
  image_geometry
  roscpp
)

find_package(OpenCV REQUIRED)

find_package(Eigen REQUIRED)
add_definitions(${EIGEN_DEFINITIONS})
find_package(VisionWorks REQUIRED)

message("Architecture: " ${CMAKE_SYSTEM_PROCESSOR})
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native")
endif()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS image_geometry utilities clustering fato_rendering
)

message("Project source dir: " ${PROJECT_SOURCE_DIR})


# show additional files in qtcreator
execute_process(COMMAND find ${PROJECT_SOURCE_DIR} -type f
  OUTPUT_VARIABLE FILES_STRING)
string(REPLACE "\n" ";" FILES_LIST ${FILES_STRING})
add_custom_target(additional_files_${PROJECT_NAME} SOURCES
  ${FILES_LIST}
  ${PROJECT_SOURCE_DIR}/../README.md
)

include_directories(include
  ${PROJECT_SOURCE_DIR}
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIR}
  ${VisionWorks_INCLUDE_DIR}
)

add_library(${PROJECT_NAME}
  src/matcher.cpp
  src/target.cpp
  src/pose_estimation.cpp
  src/feature_matcher.cpp
  src/tracker_2d.cpp
  src/tracker_model.cpp
  src/tracker_2d_v2.cpp
  src/tracker_3d.cpp
  src/bounding_cube.cpp
  src/model.cpp
  src/config.cpp
  src/epnp.cpp
  src/synthetic_track.cpp
  src/tracker_model_vx.cpp
  src/flow_graph.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${VisionWorks_LIBRARIES}
)
