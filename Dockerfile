
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc-11 g++-11 \
    cmake git ninja-build \
    libgtk2.0-dev pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libtbb-dev \
    python3-dev python3-numpy \
    libopenblas-dev liblapack-dev libeigen3-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    nsight-compute-2025.2.0 \
    nsight-systems-2024.6.2 \
    && rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc-11
ENV CXX=/usr/bin/g++-11
ENV CUDAHOSTCXX=/usr/bin/g++-11

WORKDIR /workspace

RUN git clone --depth 1 --branch 5.0.0 \
    https://github.com/opencv/opencv.git \
    /opt/opencv && \
    git clone --depth 1 --branch 5.0.0 \
    https://github.com/opencv/opencv_contrib.git \
    /opt/opencv_contrib

RUN cmake -S /opt/opencv \
    -B /opt/opencv/build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
    -DWITH_CUDA=ON \
    -DCUDA_ARCH_BIN=6.1 \
    -DCUDA_ARCH_PTX= \
    -DENABLE_FAST_MATH=OFF \
    -DCUDA_FAST_MATH=OFF \
    -DWITH_CUBLAS=OFF \
    -DWITH_NVCUVID=OFF \
    -DWITH_NVCUVENC=OFF \
    -DWITH_OPENGL=OFF \
    -DBUILD_LIST=core,imgproc,imgcodecs,highgui,videoio,cudev,cudaimgproc \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_python3=OFF \
    -DBUILD_opencv_python2=OFF

RUN cmake --build /opt/opencv/build --parallel 4 && \
    cmake --install /opt/opencv/build

RUN rm -rf /opt/opencv /opt/opencv_contrib

RUN ldconfig

# Default command
CMD ["/bin/bash"]
