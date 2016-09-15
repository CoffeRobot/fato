cmake_minimum_required(VERSION 2.8.7)
project(fato_cuda)

find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  roscpp
  resource_retriever
)

if(TRACKER_VERBOSE_LOGGING)
  add_definitions(-DVERBOSE_LOGGING)
endif()

find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDA 7.0 REQUIRED)
find_package(Eigen REQUIRED)
add_definitions(${EIGEN_DEFINITIONS})
find_package(urdfdom REQUIRED)
find_package(VisionWorks REQUIRED)

message("Vision works includes: ${VisionWorks_INCLUDE_DIRS}")
message("Vision works libraries: ${VisionWorks_LIBRARIES}")

set(GENCODE_SM35 -gencode=arch=compute_35,code=sm_35)
set(GENCODE_SM50 -gencode=arch=compute_50,code=sm_50)
set(GENCODE_SM52 -gencode=arch=compute_52,code=sm_52)
set(GENCODE_SM53 -gencode=arch=compute_53,code=sm_53)
set(CUDA_NVCC_FLAGS -O3;--compiler-options;-fPIC;-ftz=true;-prec-sqrt=false;-prec-div=false)
set(CUDA_PROPAGATE_HOST_FLAGS OFF) # don't propagate c++11 options to nvcc

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
  set(CUDA_OPTIONS ${GENCODE_SM53} ${VERBOSE_PTXAS})
else()
  set(CUDA_OPTIONS ${GENCODE_SM35} ${GENCODE_SM50} ${GENCODE_SM52} ${VERBOSE_PTXAS})
endif()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

pkg_check_modules(GLEW REQUIRED glew)

find_package(ASSIMP QUIET)
if (NOT ASSIMP_FOUND)
  pkg_check_modules(ASSIMP assimp)
endif()
if (ASSIMP_FOUND)
  if( NOT ${ASSIMP_VERSION} VERSION_LESS "2.0.1150" )
    add_definitions(-DASSIMP_UNIFIED_HEADER_NAMES)
    message(STATUS "Assimp version has unified headers")
  else()
    message(STATUS "Assimp version does not have unified headers")
  endif()
  include_directories(${ASSIMP_INCLUDE_DIRS})
  link_directories(${ASSIMP_LIBRARY_DIRS})
else()
  message(STATUS "could not find assimp (perhaps available through ROS package?), so assimping assimp v2")
  set(ASSIMP_LIBRARIES assimp)
  set(ASSIMP_LIBRARY_DIRS)
  set(ASSIMP_CXX_FLAGS)
  set(ASSIMP_CFLAGS_OTHER)
  set(ASSIMP_LINK_FLAGS)
  set(ASSIMP_INCLUDE_DIRS)
endif()



catkin_package(
  INCLUDE_DIRS
    include
    #${OGRE_OV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${VisionWorks_INCLUDE_DIRS}
  DEPENDS
    libx11-dev
    opengl
    libglew-dev
  LIBRARIES
    ${PROJECT_NAME}
    #${OGRE_OV_LIBRARIES_ABS}
  CATKIN_DEPENDS
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

include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIR}
  #${OGRE_OV_INCLUDE_DIRS}
  ${urdfdom_INCLUDE_DIRS}
  ${VisionWorks_INCLUDE_DIRS}
)

cuda_add_library(${PROJECT_NAME}
  src/utility_kernels_pose.cu
  src/utility_kernels.cu
  src/gl_interop.cu
  OPTIONS ${CUDA_OPTIONS}
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
  #${OGRE_OV_LIBRARIES_ABS}
  ${CUDA_LIBRARIES}
  ${urdfdom_LIBRARIES}
  ${VisionWorks_LIBRARIES}
  assimp
)
