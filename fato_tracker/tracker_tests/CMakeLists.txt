project(fato_tracker_tests)
cmake_minimum_required(VERSION 2.8.3)

if(TRACKER_WITH_CUSTOM_OPENCV)
  MESSAGE("${PROJECT_NAME}: using custom opencv path ${TRACKER_CUSTON_OPENCV_PATH}")
  set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules;${CMAKE_MODULE_PATH}")
  #set(OpenCV_DIR "${TRACKER_CUSTON_OPENCV_PATH}/share/OpenCV")
  find_package(OpenCV REQUIRED NO_DEFAULT_PATH)
  #set(OpenCV_INCLUDE_DIRS "${TRACKER_CUSTON_OPENCV_PATH}/include")

  #include_directories(${TRACKER_CUSTON_OPENCV_PATH}/include)
  #link_directories(${TRACKER_CUSTON_OPENCV_PATH}/lib)

  MESSAGE("${PROJECT_NAME}: opencv incl dirs " ${OpenCV_INCLUDE_DIRS})
  MESSAGE("${PROJECT_NAME}: opencv lib ${OpenCV_LIBS}")
  #SET(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${TRACKER_CUSTON_OPENCV_PATH} )
  MESSAGE("${PROJECT_NAME}: cmake prefix path ${CMAKE_PREFIX_PATH}")

endif()


if(TRACKER_VERBOSE_LOGGING)
  add_definitions(-DVERBOSE_LOGGING)
endif()

set(TRACKER_MODULES
      roscpp
      cmake_modules
      cv_bridge
      image_transport
      tf
      image_geometry
      utilities
      tracker_cpu
      tracker_io
)

if(TRACKER_WITH_GPU)
  add_definitions(-DTRACKER_WITH_GPU)
set(TRACKER_MODULES ${TRACKER_MODULES} cuda_sift)
endif()

find_package(catkin REQUIRED COMPONENTS
  ${TRACKER_MODULES}
)

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
  ${PROJECT_SOURCE_DIR}/../../README.md
  ${PROJECT_SOURCE_DIR}/CMakeLists.txt
)


set(include_dirs include
             ${catkin_INCLUDE_DIRS}
             ${EIGEN_INCLUDE_DIR}
             ${Boost_INCLUDE_DIRS}
)

if(TRACKER_WITH_GPU)
  include_directories(include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
  )
else()
  include_directories(include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
  )
endif()

add_executable(test_projection
  src/projection.cpp
  )

target_link_libraries(test_projection
  ${catkin_LIBRARIES}
  ${OPENCV_LIBS}
  ${Boost_LIBRARIES})

  add_executable(test_tracker2d
    src/tracker2d_test.cpp
   )

  target_link_libraries(test_tracker2d
    ${catkin_LIBRARIES}
    ${OPENCV_LIBS}
    ${Boost_LIBRARIES})
