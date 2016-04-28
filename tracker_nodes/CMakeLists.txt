cmake_minimum_required(VERSION 2.8.7)

project(fato_tracker_nodes)
find_package(catkin REQUIRED COMPONENTS
    roscpp
    cmake_modules
    cv_bridge
    image_transport
    tf
    image_geometry
    tracker
    fato_rendering
    tracker_io
)


find_package(OpenCV 2.4.8 REQUIRED)
find_package(Eigen REQUIRED)
add_definitions(${EIGEN_DEFINITIONS})

catkin_package(
  INCLUDE_DIRS
    include
  DEPENDS
    cv_bridge
    image_transport
    roscpp
    tf
    image_geometry
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS tracker fato_rendering utilities tracker_io
)


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# show additional files in qtcreator
execute_process(COMMAND find ${PROJECT_SOURCE_DIR} -type f
  OUTPUT_VARIABLE FILES_STRING)
string(REPLACE "\n" ";" FILES_LIST ${FILES_STRING})
add_custom_target(additional_files_${PROJECT_NAME} SOURCES
  ${FILES_LIST}
  ${PROJECT_SOURCE_DIR}/../README.md
)

MESSAGE("INCLUDE DIRS " + ${catkin_INCLUDE_DIRS})

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIR} 
)

if(TRACKER_WITH_GPU)
  add_executable(tracker_2d_gpu
    src/tracker_gpu.cpp
  )

  target_link_libraries(tracker_2d_gpu
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
  )
endif()

add_executable(tracker_node_2d
  src/tracker_node.cpp
  src/tracker_node_2d.cpp)

target_link_libraries(tracker_node_2d
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
)

add_executable(tracker_node_3d
  src/tracker_node.cpp
  src/tracker_node_3d.cpp)


target_link_libraries(tracker_node_3d
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

add_executable(tracker_node_offline
  src/tracker_offline.cpp)


target_link_libraries(tracker_node_offline
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

add_executable(generate_model
  src/generate_model.cpp)


target_link_libraries(generate_model
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

add_executable(track_model
    src/tracker_model_based.cpp
    src/tracker_node.cpp)


target_link_libraries(track_model
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

