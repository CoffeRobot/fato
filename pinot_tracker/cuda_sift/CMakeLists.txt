cmake_minimum_required(VERSION 2.8.3)
project(cuda_sift)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED)

find_package(OpenCV REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
)

# show additional files in qtcreator
execute_process(COMMAND find ${PROJECT_SOURCE_DIR} -type f
  OUTPUT_VARIABLE FILES_STRING)
string(REPLACE "\n" ";" FILES_LIST ${FILES_STRING})
add_custom_target(additional_files_${PROJECT_NAME} SOURCES
  ${FILES_LIST}
)

include_directories(include
                    ${catkin_INCLUDE_DIRS}
                    ${CUDA_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
)

set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2 -std=c++0x ")

if(TRACKER_WITH_GPU)
  include(CPack)

  find_package(CUDA)

  set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20)
  set(GENCODE_SM30 -gencode=arch=compute_30,code=sm_30)
  set(GENCODE_SM35 -gencode=arch=compute_35,code=sm_35)
  set(CUDA_NVCC_FLAGS -O3;-ftz=true;-prec-sqrt=false;-prec-div=false)
  set(CUDA_OPTIONS ${GENCODE_SM20} ${GENCODE_SM30} ${GENCODE_SM35} ${VERBOSE_PTXAS})
  set(CUDA_PROPAGATE_HOST_FLAGS ON)

  set(cuda_sources
      src/cudaImage.cu
      src/cudaSiftH4.cu
      src/singular.cu
  )

  set(sources
      src/geomFuncs.cpp
      src/cudaSift_gold.cpp
  )

  cuda_add_library(${PROJECT_NAME}
                   ${cuda_sources}
                   ${sources}
                   OPTIONS ${CUDA_OPTIONS}
  )

  target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
  )

endif()

