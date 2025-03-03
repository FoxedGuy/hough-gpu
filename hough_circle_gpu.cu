#include<cuda_runtime.h>
#include<vector>
#include<omp.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define CUDA_CHECK(call)                                                   \
    {                                                                      \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA Error: %s (error code %d), at %s:%d\n",  \
                    cudaGetErrorString(err), err, __FILE__, __LINE__);     \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    }

struct circle{
    int x;
    int y;
    int r;
};

__global__ void compute_trig_tables(float* sin_table, float* cos_table){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < 360){
        float angle = idx * M_PI / 180.0;
        sin_table[idx] = sinf(angle);
        cos_table[idx] = cosf(angle);
    }
}

__global__ void fill_accum(unsigned char* data, int width, int height, int radius_min, int radius_max, unsigned int* accum, float* sin_table, float* cos_table){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int r = blockIdx.z * blockDim.z + threadIdx.z + radius_min;
    if(x >= width || y >= height || r > radius_max) return;
    if(data[y * width + x] == 0) return;


    for(int angle = 0; angle < 360; angle++){
        int a = x - r * cos_table[angle];
        int b = y - r * sin_table[angle];
        if(a >= 0 && a < width && b >= 0 && b < height){
            atomicAdd(&accum[(r - radius_min)* height * width + b * width + a], 1);
        }
    }
}

__global__ void find_maxims(unsigned int* acc, int radius_min, int radius_max, int height, int width, int threshold, circle* d_centers, int* d_count, int * lock) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int r = blockIdx.z * blockDim.z + threadIdx.z + radius_min;
    
    if (x >= width || y >= height || r > radius_max) return;
    
    int idx = (r - radius_min) * height * width + y * width + x;
    if (acc[idx] > threshold) {
        bool is_local_max = true;
        for (int dy = -1; dy <= 1 && is_local_max; dy++) {
            for (int dx = -1; dx <= 1 && is_local_max; dx++) {
                if (dx == 0 && dy == 0) continue;
                int ny = y + dy;
                int nx = x + dx;
                if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                    int n_idx = (r - radius_min) * height * width + ny * width + nx;
                    if (acc[n_idx] > acc[idx]) {
                        is_local_max = false;
                    }
                }
            }
        }
        if (is_local_max) {
            // Critical Section Using Lock
            while (atomicCAS(lock, 0, 1) != 0);  // Spinlock

            // Critical Section: Add new circle center
            d_centers[*d_count].x = x;
            d_centers[*d_count].y = y;
            d_centers[*d_count].r = r;
            atomicAdd(d_count,1);

            // Release Lock
            atomicExch(lock, 0);
        }
    }
}

std::pair<int, circle*> hough_circle_gpu(unsigned char* data, int width, int height, int radius_min, int radius_max, int threshold){
    int radii_range = radius_max - radius_min + 1;
    int hough_space_size = width * height * radii_range;
    unsigned int* accum;
    cudaMalloc(&accum, hough_space_size * sizeof(unsigned int));
    cudaMemset(accum, 0, hough_space_size * sizeof(unsigned int));

    float *sin_table, *cos_table;
    cudaMalloc(&sin_table, 360 * sizeof(float));
    cudaMalloc(&cos_table, 360 * sizeof(float));
    
    dim3 blockDim(360);
    dim3 gridDim(1);
    
    compute_trig_tables<<<gridDim, blockDim>>>(sin_table, cos_table);

    unsigned char* d_data;
    cudaMalloc(&d_data, width * height * sizeof(unsigned char));
    cudaMemcpy(d_data, data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    dim3 fillingBlockDim(16,16,4);
    dim3 fillingGridDim((width + fillingBlockDim.x - 1) / fillingBlockDim.x, (height + fillingBlockDim.y - 1) / fillingBlockDim.y, (radius_max - radius_min + fillingBlockDim.z) / fillingBlockDim.z);
    fill_accum<<<fillingGridDim, fillingBlockDim>>>(d_data, width, height, radius_min, radius_max, accum, sin_table, cos_table);

    int* d_count;
    circle* d_centers;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));
    cudaMalloc(&d_centers, width * height * sizeof(circle));
    int *d_lock;
    cudaMalloc(&d_lock, sizeof(int));
    cudaMemset(d_lock, 0, sizeof(int));  // Initialize to 0
    find_maxims<<<fillingGridDim, fillingBlockDim>>>(accum, radius_min, radius_max, height, width, threshold, d_centers, d_count, d_lock);
    
    cudaFree(accum);
    cudaFree(sin_table);
    cudaFree(cos_table);
    cudaFree(d_data);
    
    int count;
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    circle* centers = new circle[count];
    
    cudaMemcpy(centers, d_centers, count * sizeof(circle), cudaMemcpyDeviceToHost);
    cudaFree(d_count);
    cudaFree(d_centers);

    return std::pair<int,circle*>(count,centers);
}


int main(int argc, char** argv){

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount == 0){
        std::cerr << "No CUDA device found" << std::endl;
        return -1;
    }

    int device = 0;
    cudaSetDevice(device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "CUDA Device: " << prop.name << std::endl;

    std::string filename = argv[1];
    int radius_min = std::stoi(argv[2]);
    int radius_max = std::stoi(argv[3]);
    int threshold = std::stoi(argv[4]);
    std::string path = "pictures_circles/" + filename;
    cv::Mat img = cv::imread(path, 1);

    if(img.empty()){
        std::cerr << "Image not found" << std::endl;
        return -1;
    }

    cv::Mat img_result = img.clone();
    cv::Mat edges, blur;
    cv::GaussianBlur(img, blur, cv::Size(9, 9), 2, 2);
    cv::Canny(blur, edges, 100, 200);

    auto start = omp_get_wtime();
    std::pair<int,circle*> result = hough_circle_gpu(edges.data, edges.cols, edges.rows, radius_min, radius_max, threshold);
    auto end = omp_get_wtime();

    std::cout << "Time: " << end - start << std::endl;

    int count = result.first;
    circle* circles = result.second;

    std::cout << "Number of circles: " << count << std::endl;
    for (int i = 0; i < count; i++) {
        cv::circle(img_result, cv::Point(circles[i].x, circles[i].y), circles[i].r, cv::Scalar(255, 0, 255), 2);
    }

    cv::Mat img_concat = cv::Mat::zeros(img.rows, img.cols + edges.cols, CV_8UC3);
    cv::cvtColor(edges, edges, cv::COLOR_GRAY2BGR);
    cv::hconcat(img, edges, img_concat);
    cv::hconcat(img_concat, img_result, img_concat);
    cv::imshow("Original image, edges, result", img_concat);
    cv::imwrite("results/circles/gpu/result.png", img_concat);
    cv::waitKey(0);



    delete [] circles;
    return 0;
}