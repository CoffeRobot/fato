project(pinot_tracker_tests)

cmake_minimum_required(VERSION 2.8.3)
find_package(catkin REQUIRED)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  cmake_modules
  cv_bridge
  image_transport
  pcl_ros
  tf
  image_geometry
  utilities
  tracker_cpu
)

find_package(OpenCV 2.4.8 REQUIRED)
find_package(Eigen REQUIRED)
add_definitions(${EIGEN_DEFINITIONS})
find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

catkin_package(
  INCLUDE_DIRS include
  DEPENDS cv_bridge
          image_transport
          pcl_ros
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
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIR} 
)

add_executable(test_projection
  src/projection.cpp
  )


target_link_libraries(test_projection
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)


#add_executable(cloud_publisher
#    src/cloud_publisher.cpp)

#target_link_libraries(cloud_publisher
#  ${catkin_LIBRARIES}
#  ${OpenCV_LIBS}
#  ${PCL_LIBRARIES}
#)

#MESSAGE( STATUS "PCL INCLUDES:         " ${PCL_INCLUDE_DIRS} )
