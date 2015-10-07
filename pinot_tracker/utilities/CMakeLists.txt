cmake_minimum_required(VERSION 2.8.7)
project(utilities)

find_package(catkin REQUIRED COMPONENTS
  cmake_modules
#  pcl_ros
)

find_package(OpenCV REQUIRED)

find_package(Eigen REQUIRED)
add_definitions(${EIGEN_DEFINITIONS})

if(TRACKER_WITH_GPU)
  add_definitions(-DTRACKER_WITH_GPU)
  find_package(CUDA REQUIRED)
endif()

#find_package(PCL 1.7 REQUIRED)
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

if(TRACKER_WITH_GPU)
catkin_package(
  INCLUDE_DIRS include ${CUDA_INCLUDE_DIRS}
  LIBRARIES ${PROJECT_NAME}
 )
else()
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
 )
endif()

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

# show additional files in qtcreator
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
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
)
else()
  include_directories(include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIR}
  )
endif()


add_library(${PROJECT_NAME}
  src/DebugFunctions.cpp
  src/ToString.cpp
  src/utilities.cpp
  src/GpuDebugFunctions.cpp
  src/draw_functions.cpp
  src/profiler.cpp
  src/visualization_ros.cpp
  src/data_structures.cpp
  src/logger.cpp
)

if(TRACKER_WITH_GPU)

else()

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
)
endif()

