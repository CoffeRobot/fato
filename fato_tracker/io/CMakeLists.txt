cmake_minimum_required(VERSION 2.8.7)
project(tracker_io)

find_package(catkin REQUIRED COMPONENTS
  cmake_modules
#  pcl_ros
)

# needed to find libav package
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules/;${CMAKE_MODULE_PATH}")

if(TRACKER_VERBOSE_LOGGING)
  add_definitions(-DVERBOSE_LOGGING)
endif()

find_package(OpenCV REQUIRED)

find_package(Eigen REQUIRED)
add_definitions(${EIGEN_DEFINITIONS})

find_package(Boost COMPONENTS filesystem system REQUIRED)

find_package(LibAV REQUIRED)

catkin_package(
  INCLUDE_DIRS include
#  DEPENDS pcl_ros
  LIBRARIES ${PROJECT_NAME}
)

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

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
  ${Boost_INCLUDE_DIRS} 
  ${LIBAV_INCLUDE_DIRS} 
)

add_library(${PROJECT_NAME}
  src/Encoderx264.cpp
  src/VideoWriter.cpp
  src/filemanager.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES} 
  ${LIBAV_LIBRARIES}
)

