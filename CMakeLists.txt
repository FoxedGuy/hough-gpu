cmake_minimum_required(VERSION 3.0.0)
project(hough-seq LANGUAGES CXX)

set(OpenCV_DIR ~/opencv/build)
find_package( OpenCV REQUIRED )
# find_package( CUDA )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# add_executable(hough_gpu hough_gpu.cu)
add_executable(hough_seq hough_seq.cpp)

# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=sm_30)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++12)

target_link_libraries(hough_seq ${OpenCV_LIBS} )