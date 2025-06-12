cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_ARCHITECTURES "80")
project(hough-seq LANGUAGES CXX CUDA)

set(OpenCV_DIR /usr/local/include/opencv4)
find_package( OpenCV REQUIRED )
find_package( CUDAToolkit )
find_package( OpenMP REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(hough_gpu_streams hough_gpu_streams.cu)
add_executable(hough_seq hough_seq.cpp)
add_executable(hough_circle_gpu hough_circle_gpu.cu)

target_compile_options(hough_gpu_streams PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
)


target_compile_options(hough_circle_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
)

target_link_libraries(hough_seq ${OpenCV_LIBS} )
target_link_libraries(hough_circle_gpu ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
target_link_libraries(hough_gpu_streams ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -std=c++11 -fopenmp")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler=-Wall -Xcompiler=-fopenmp --allow-unsupported-compiler")

