cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CUDA_ARCHITECTURES "80")
project(hough-seq LANGUAGES CXX CUDA)

set(OpenCV_DIR /usr/local/include/opencv4)
find_package( OpenCV REQUIRED )
find_package( CUDAToolkit )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(hough_gpu hough_gpu.cu)
add_executable(hough_seq hough_seq.cpp)

target_compile_options(hough_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -std=c++11 -fopenmp")

target_link_libraries(hough_seq ${OpenCV_LIBS} )
target_link_libraries(hough_gpu ${OpenCV_LIBS} )