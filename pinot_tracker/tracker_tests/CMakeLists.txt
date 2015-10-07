project(pinot_tracker_tests)

cmake_minimum_required(VERSION 2.8.3)
find_package(catkin REQUIRED)

set(TRACKER_VERBOSE_LOGGING ON CACHE BOOL "set to true to have verbose logs")

if(TRACKER_VERBOSE_LOGGING)
  add_definitions(-DVERBOSE_LOGGING)
endif()

if(TRACKER_WITH_GPU)
  add_definitions(-DTRACKER_WITH_GPU)
endif()


find_package(catkin REQUIRED COMPONENTS
  roscpp
  cmake_modules
  cv_bridge
  image_transport
  tf
  image_geometry
  utilities
  tracker_cpu
  tracker_io
  cuda_sift
)

find_package(OpenCV 2.4.8 REQUIRED)
find_package(Eigen REQUIRED)
add_definitions(${EIGEN_DEFINITIONS})
find_package(Boost COMPONENTS filesystem system REQUIRED)


catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  DEPENDS cv_bridge
          image_transport
          roscpp
          tf
          image_geometry
)


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

# show additional files in qtcreator
execute_process(COMMAND find ${PROJECT_SOURCE_DIR} -type f
  OUTPUT_VARIABLE FILES_STRING)
string(REPLACE "\n" ";" FILES_LIST ${FILES_STRING})
add_custom_target(additional_files_${PROJECT_NAME} SOURCES
  ${FILES_LIST}
  #${PROJECT_SOURCE_DIR}/../README.md
)

include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
)


add_executable(test_projection
  src/projection.cpp
  )

target_link_libraries(test_projection
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
)

add_executable(benchmark_features
  src/feature_matching.cpp
  )

target_link_libraries(benchmark_features
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
)

