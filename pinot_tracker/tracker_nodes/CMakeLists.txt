project(pinot_tracker_nodes)

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
  tracker_gpu
)

find_package(OpenCV 2.4.8 REQUIRED)
find_package(Eigen REQUIRED)
add_definitions(${EIGEN_DEFINITIONS})
find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

## Generate actions in the 'action' folder
#add_action_files(
#  DIRECTORY action
#  FILES BT.action
#)

#add_service_files(
#  DIRECTORY srv
#  FILES ReceiveCloud.srv StartAggregator.srv CropShelf.srv
#)

#generate_messages(
#  DEPENDENCIES actionlib_msgs std_msgs sensor_msgs# Or other packages containing msgs
#)

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

if(WITH_GPU)
  add_executable(tracker_2d_gpu
    src/tracker_gpu.cpp
  )

  target_link_libraries(tracker_2d_gpu
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
  )
endif()




#add_executable(cloud_publisher
#    src/cloud_publisher.cpp)

#target_link_libraries(cloud_publisher
#  ${catkin_LIBRARIES}
#  ${OpenCV_LIBS}
#  ${PCL_LIBRARIES}
#)

#MESSAGE( STATUS "PCL INCLUDES:         " ${PCL_INCLUDE_DIRS} )
