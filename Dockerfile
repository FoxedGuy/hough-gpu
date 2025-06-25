FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install base dependencies
RUN apt update && apt install -y \
    build-essential \
    gcc-11 g++-11 \
    cmake git ninja-build \
    libgtk2.0-dev pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libtbb2 libtbb-dev \
    python3-dev python3-numpy \
    libopenblas-dev liblapack-dev libeigen3-dev \
    libomp-dev \
    && apt clean

# Add Nsight tools
RUN apt update && apt install -y \
    nsight-compute-2025.2.0 \
    nsight-systems-2024.6.2

ENV CC=/usr/bin/gcc-11
ENV CXX=/usr/bin/g++-11
ENV CUDAHOSTCXX=/usr/bin/g++-11

# Set working directory
WORKDIR /workspace

# Clone OpenCV
RUN git clone --depth 1 https://github.com/opencv/opencv.git && \
    git clone --depth 1 https://github.com/opencv/opencv_contrib.git

# Build OpenCV with CUDA
RUN mkdir -p opencv/build && cd opencv/build && \
    cmake -G Ninja \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=/workspace/opencv_contrib/modules \
    -D WITH_CUDA=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D WITH_OPENGL=OFF \
    -D BUILD_opencv_python3=OFF \
    -D BUILD_opencv_python2=OFF \
    .. && \
    ninja -j$(nproc) && ninja install

# Clean build files
RUN rm -rf /workspace/opencv /workspace/opencv_contrib

# Default command
CMD ["/bin/bash"]