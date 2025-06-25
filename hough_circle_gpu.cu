#include <cuda_runtime.h>
#include <vector>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cmath>

#define CUDA_CHECK(call)                                                   \
    {                                                                      \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA Error: %s (error code %d), at %s:%d\n",  \
                    cudaGetErrorString(err), err, __FILE__, __LINE__);     \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    }

struct circle {
    int x;
    int y;
    int r;
    int votes;
    int valid;
};

__global__ void compute_trig_tables(float* sin_table, float* cos_table) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < 360) {
        float angle = idx * M_PI / 180.0f;
        sin_table[idx] = sinf(angle);
        cos_table[idx] = cosf(angle);
    }
}

__global__ void fill_accum(unsigned char* data, int width, int height, int radius_min, int radius_max, unsigned int* accum, float* sin_table, float* cos_table) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int r = blockIdx.z * blockDim.z + threadIdx.z + radius_min;
    if (x >= width || y >= height || r > radius_max) return;
    if (data[y * width + x] == 0) return;

    for (int angle = 0; angle < 360; angle++) {
        int a = x - r * cos_table[angle];
        int b = y - r * sin_table[angle];
        if (a >= 0 && a < width && b >= 0 && b < height) {
            atomicAdd(&accum[(r - radius_min) * (height + 2) * (width + 2) + (b + 1) * (width + 2) + (a + 1)], 1);
        }
    }
}

__global__ void find_maxims(unsigned int* acc, int radius_min, int radius_max, int height, int width, int threshold, circle* d_centers, int* d_count) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int r = blockIdx.z * blockDim.z + threadIdx.z + radius_min;

    if (x >= width || x < 1 || y < 1 || y >= height || r > radius_max || r < radius_min) return;

    int idx = (r - radius_min) * (height + 2) * (width + 2) + (y + 1) * (width + 2) + (x + 1);
    if (acc[idx] > threshold) {
        bool is_local_max = true;
        for (int dr = -1; dr <= 1 && is_local_max; ++dr) {
            int rr = r + dr;
            if (rr < radius_min || rr > radius_max) continue;
            for (int dy = -1; dy <= 1 && is_local_max; ++dy) {
                for (int dx = -1; dx <= 1 && is_local_max; ++dx) {
                    if (dx == 0 && dy == 0 && dr == 0) continue;
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        int n_idx = (rr - radius_min) * (height + 2) * (width + 2) + (ny + 1) * (width + 2) + (nx + 1);
                        if (acc[n_idx] > acc[idx]) {
                            is_local_max = false;
                        }
                    }
                }
            }
        }
        if (is_local_max) {
            int write_idx = atomicAdd(d_count, 1);
            d_centers[write_idx].x = x;
            d_centers[write_idx].y = y;
            d_centers[write_idx].r = r;
            d_centers[write_idx].votes = acc[idx];
            d_centers[write_idx].valid = 1;
        }
    }
}

__global__ void suppress_similar_circles(circle* circles, int num, float dist_thresh, int radius_thresh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num || circles[i].valid == 0) return;

    for (int j = 0; j < num; ++j) {
        if (i == j || circles[j].valid == 0) continue;
        float dx = circles[i].x - circles[j].x;
        float dy = circles[i].y - circles[j].y;
        int dr = abs(circles[i].r - circles[j].r);
        float dist = sqrtf(dx * dx + dy * dy);

        if (dist < dist_thresh && dr <= radius_thresh) {
            if (circles[j].votes > circles[i].votes) {
                circles[i].valid = 0;
                return;
            }
        }
    }
}

std::vector<circle> hough_circle_gpu(unsigned char* data, int width, int height, int radius_min, int radius_max, int threshold) {
    int radii_range = radius_max - radius_min + 1;
    int hough_space_size = (width + 2) * (height + 2) * radii_range;

    unsigned int* accum;
    circle* d_centers;
    float *sin_table, *cos_table;
    unsigned char* d_data;
    int* d_count;

    cudaMalloc(&accum, hough_space_size * sizeof(unsigned int));
    cudaMemset(accum, 0, hough_space_size * sizeof(unsigned int));
    cudaMalloc(&sin_table, 360 * sizeof(float));
    cudaMalloc(&cos_table, 360 * sizeof(float));

    compute_trig_tables<<<1, 360>>>(sin_table, cos_table);

    cudaMalloc(&d_data, width * height * sizeof(unsigned char));
    cudaMemcpy(d_data, data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));
    cudaMalloc(&d_centers, width * height * sizeof(circle));

    dim3 blockDim(16, 16, 4);
    dim3 gridDim((width + 15) / 16, (height + 15) / 16, (radii_range + 3) / 4);
    fill_accum<<<gridDim, blockDim>>>(d_data, width, height, radius_min, radius_max, accum, sin_table, cos_table);

    find_maxims<<<gridDim, blockDim>>>(accum, radius_min, radius_max, height, width, threshold, d_centers, d_count);

    int max_candidates = width * height;
    suppress_similar_circles<<<(max_candidates + 255) / 256, 256>>>(d_centers, max_candidates, 10.0f, 6);

    std::vector<circle> final;
    circle* host_circles = new circle[max_candidates];
    cudaMemcpy(host_circles, d_centers, max_candidates * sizeof(circle), cudaMemcpyDeviceToHost);

    for (int i = 0; i < max_candidates; ++i) {
        if (host_circles[i].valid) {
            final.push_back(host_circles[i]);
        }
    }

    delete[] host_circles;
    cudaFree(accum); cudaFree(sin_table); cudaFree(cos_table);
    cudaFree(d_data); cudaFree(d_count); cudaFree(d_centers);

    return final;
}

std::vector<circle> hough_circle_cpu(const cv::Mat& img, int height, int width, int radius_min, int radius_max, int threshold) {
    const int numangle = 360;
    const int numradii = radius_max - radius_min + 1;
    const int accu_size = (height + 2) * (width + 2) * numradii;

    std::vector<unsigned int> accu(accu_size, 0);
    std::vector<float> sin_table(numangle);
    std::vector<float> cos_table(numangle);

    for (int i = 0; i < numangle; i++) {
        float angle = i * CV_PI / 180.0f;
        sin_table[i] = sinf(angle);
        cos_table[i] = cosf(angle);
    }

    std::vector<cv::Point> non_zero;
    cv::findNonZero(img, non_zero);

    for (const auto& pt : non_zero) {
        for (int r = radius_min; r <= radius_max; r++) {
            int radius_idx = r - radius_min;
            int base_idx = radius_idx * (height + 2) * (width + 2);
            for (int i = 0; i < numangle; i++) {
                int a = pt.x - r * cos_table[i];
                int b = pt.y - r * sin_table[i];
                if (a >= 0 && a < width && b >= 0 && b < height) {
                    accu[base_idx + (b + 1) * (width + 2) + (a + 1)]++;
                }
            }
        }
    }

    std::vector<circle> centers;
    for (int r = radius_min; r <= radius_max; r++) {
        int radius_idx = r - radius_min;
        for (int y = 1; y < height; y++) {
            for (int x = 1; x < width; x++) {
                int idx = radius_idx * (height + 2) * (width + 2) + (y + 1) * (width + 2) + (x + 1);
                int val = accu[idx];
                if (val <= threshold) continue;

                bool is_local_max = true;
                for (int dr = -1; dr <= 1 && is_local_max; ++dr) {
                    int rr = r + dr;
                    if (rr < radius_min || rr > radius_max) continue;
                    for (int dy = -1; dy <= 1 && is_local_max; ++dy) {
                        for (int dx = -1; dx <= 1 && is_local_max; ++dx) {
                            if (dx == 0 && dy == 0 && dr == 0) continue;
                            int nx = x + dx;
                            int ny = y + dy;
                            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                                int n_idx = (rr - radius_min) * (height + 2) * (width + 2) + (ny + 1) * (width + 2) + (nx + 1);
                                if (accu[n_idx] > val) {
                                    is_local_max = false;
                                }
                            }
                        }
                    }
                }
                if (is_local_max) {
                    centers.push_back({x, y, r, val, 1});
                }
            }
        }
    }

    std::vector<circle> final;
    for (size_t i = 0; i < centers.size(); ++i) {
        if (!centers[i].valid) continue;
        for (size_t j = 0; j < centers.size(); ++j) {
            if (i == j || !centers[j].valid) continue;
            float dx = centers[i].x - centers[j].x;
            float dy = centers[i].y - centers[j].y;
            int dr = abs(centers[i].r - centers[j].r);
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist < 10.0f && dr <= 2 && centers[j].votes > centers[i].votes) {
                centers[i].valid = 0;
                break;
            }
        }
        if (centers[i].valid) final.push_back(centers[i]);
    }

    return final;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: ./hough_gpu image.jpg radius_min radius_max threshold\n";
        return -1;
    }

    std::string filename = argv[1];
    int radius_min = std::stoi(argv[2]);
    int radius_max = std::stoi(argv[3]);
    int threshold = std::stoi(argv[4]);
    int n = std::stoi(argv[5]);

    cv::Mat img = cv::imread("../pictures_circles/" + filename);
    if (img.empty()) {
        std::cerr << "Failed to load image\n";
        return -1;
    }

    cv::Mat img_result = img.clone();
    cv::Mat img_result_gpu = img.clone();
    cv::Mat gray, blur, edges;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    // cv::GaussianBlur(gray, blur, cv::Size(5, 5), 2, 2);
    cv::Canny(gray, edges, 250, 350);

    imwrite("../results/circles/gpu/edges.png", edges);

    float time_gpu = 0.0f;
    float time_cpu = 0.0f;
    int circles_found_cpu = 0;
    int circles_found_gpu = 0;

    for (int i = 0; i < n; ++i){
        auto start_gpu = omp_get_wtime();
        auto circles_gpu = hough_circle_gpu(edges.data, edges.cols, edges.rows, radius_min, radius_max, threshold);
        auto end_gpu = omp_get_wtime();
        time_gpu += end_gpu - start_gpu;
        circles_found_gpu += circles_gpu.size();

        // auto start_cpu = omp_get_wtime();
        // auto circles_cpu = hough_circle_cpu(edges, edges.rows, edges.cols, radius_min, radius_max, threshold);
        // auto end_cpu = omp_get_wtime();
        // time_cpu += end_cpu - start_cpu;
        // circles_found_cpu += circles_cpu.size();

        if (n == 1) {
            for (const auto& circle : circles_gpu) {
                cv::circle(img_result_gpu, cv::Point(circle.x, circle.y), circle.r, cv::Scalar(0, 255, 0), 2);
            }
            imwrite("../results/circles/gpu/results_gpu.png", img_result_gpu);
        
            // for (const auto& circle : circles_cpu) {
            //     cv::circle(img_result, cv::Point(circle.x, circle.y), circle.r, cv::Scalar(255, 0, 0), 2);
            // }
            // imwrite("../results/circles/cpu/results_cpu.png", img_result);
        }
        
    }
    std::cout << "After " << n << " iterations:\n";
    std::cout << "GPU Time: " << time_gpu / n  << " seconds, Circles Found: " << circles_found_gpu/n << "\n";
    // std::cout << "CPU Time: " << time_cpu / n  << " seconds, Circles Found: " << circles_found_cpu/n << "\n";
    
    return 0;
}