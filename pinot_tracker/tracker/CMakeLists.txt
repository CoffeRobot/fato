project(tracker_cpu)

cmake_minimum_required(VERSION 2.8.7)

find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  clustering
  utilities
)

catkin_package(
  INCLUDE_DIRS include
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
)

add_library(${PROJECT_NAME}
  src/matcher.cpp
  src/tracker_2d.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
)
