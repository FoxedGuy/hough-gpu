#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/cudaimgproc.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include "omp.h"

struct line{
    int rho;
    float theta;
};

__global__ void segment_image(const unsigned char* inputImage,
                                        unsigned char* subImage1,
                                        unsigned char* subImage2,
                                        unsigned char* subImage3,
                                        unsigned char* subImage4,
                                        int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    int idx_in = row * N + col;

    int halfCols = N / 2;

    int out_row = row / 2;
    int out_col = col / 2;

    if ((row % 2 == 0) && (col % 2 == 0)) {
        subImage4[out_row * halfCols + out_col] = inputImage[idx_in];
    } else if ((row % 2 == 0) && (col % 2 == 1)) {
        subImage3[out_row * halfCols + out_col] = inputImage[idx_in];
    } else if ((row % 2 == 1) && (col % 2 == 0)) {
        subImage2[out_row * halfCols + out_col] = inputImage[idx_in];
    } else {
        subImage1[out_row * halfCols + out_col] = inputImage[idx_in];
    }
}

__global__ void extract_non_zero_coords(unsigned char* subImage,
                                        int* x_coords, int* y_coords, 
                                        int* count, int halfN, 
                                        int offsetX, int offsetY) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= halfN || col >= halfN) return;

    int idx = row * halfN + col;
    if (subImage[idx] != 0) {
        int global_x = col * 2 + offsetX;
        int global_y = row * 2 + offsetY;

        int index = atomicAdd(count, 1);
        x_coords[index] = global_x;
        y_coords[index] = global_y;
    }
}

__global__ void extract_non_zero(unsigned char* image,
                                 int * x, int *y,
                                 int *count, int N) {
    int x_coord = blockIdx.x * blockDim.x + threadIdx.x;
    int y_coord = blockIdx.y * blockDim.y + threadIdx.y;
    if (x_coord >= N || y_coord >= N) return;

    int idx = y_coord * N + x_coord;

    if (idx >= N * N) return;
    if (image[idx] != 0) {
        int index = atomicAdd(count, 1);
        x[index] = x_coord;
        y[index] = y_coord;    
    }
}

__global__ void sum_accumulators(int* output, int* accum2, int* accum3, int* accum4, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] +=  accum2[idx] + accum3[idx] + accum4[idx];
    }
}

__global__ void fill_trig_tables(float *sin_table, float *cos_table, float min_theta, float theta, int numangle, float irho){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < numangle){
        float angle = min_theta + index * theta;
        sin_table[index] = sinf(angle) *irho;
        cos_table[index] = cosf(angle) *irho;
    }
}

__global__ void fill_accum(int* accum,int* x_coords,int* y_coords,int num_points,float* cos_table,float* sin_table, int numangle,int numrho){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    int x = x_coords[idx];
    int y = y_coords[idx];

    for (int angle = 0; angle < numangle; angle++) {
        int r = roundf(x * cos_table[angle] + y * sin_table[angle]);
        r += (numrho - 1) / 2.f;
        atomicAdd(&accum[(angle + 1) * (numrho + 2) + r + 1], 1);
    }
}

__global__ void fill_accum_from_coords(int* accum,int* x_coords,int* y_coords,int *num_points,float* cos_table,float* sin_table, int numangle,int numrho){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *num_points) return;

    int x = x_coords[idx];
    int y = y_coords[idx];

    for (int angle = 0; angle < numangle; angle++) {
        int r = roundf(x * cos_table[angle] + y * sin_table[angle]);
        r += (numrho - 1) / 2.f;
        atomicAdd(&accum[(angle + 1) * (numrho + 2) + r + 1], 1);
    }
}

__global__ void find_maxims(int* accum, int numangle, int numrho, int threshold, float min_theta, float theta_step, float rho_step, line* lines, int *current_size){
    int angle = blockDim.x * blockIdx.x + threadIdx.x;
    int rho = blockDim.y * blockIdx.y + threadIdx.y;

    if (angle < 1 || angle >= numangle - 1 ) return;
    if (rho < 1 || rho >= numrho - 1 ) return;

    int base = (angle+1) * (numrho+2) + rho + 1;
    if( accum[base] > threshold &&
        accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] &&
        accum[base] > accum[base - numrho - 2] && accum[base] >= accum[base + numrho + 2]){
        int real_rho = (rho - (numrho - 1) *0.5f) * rho_step;
        float real_theta = min_theta + angle * theta_step;
        int index = atomicAdd(current_size,1);
        lines[index].rho = real_rho;
        lines[index].theta = real_theta;
    }
}

std::pair<int,line*> hough_parallel(unsigned char* d_img, int N, int threshold,
                                    float rho, float theta_step,double *duration,
                                    float min_theta=0.0, float max_theta=CV_PI){

    unsigned char *input;    

    cudaMalloc(&input, N*N);
    cudaMemcpy(input, d_img, N*N, cudaMemcpyHostToDevice);

    auto start_count = omp_get_wtime();

    const float irho = 1.f / rho;
    const int max_rho = N + N;
    const int numangle = (int)cvFloor((max_theta - min_theta) / theta_step) + 1;
    const int numrho = cvRound(((max_rho*2+1))*irho);
    const int accu_size = (numangle+2)*(numrho+2);

    int *accum;
    float *sinvalues, *cosvalues;
    line *d_lines;
    int *d_counter, counter = 0;
    int *d_x_coords, *d_y_coords, *d_count;

    cudaMalloc(&accum, accu_size * sizeof(int));
    cudaMemset(accum, 0, accu_size * sizeof(int));

    cudaMalloc(&sinvalues, numangle * sizeof(float));
    cudaMalloc(&cosvalues, numangle * sizeof(float));
    cudaMalloc(&d_lines, accu_size * sizeof(line));
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));
    cudaMalloc(&d_x_coords, N * N * sizeof(int));
    cudaMalloc(&d_y_coords, N * N * sizeof(int));
    cudaMallocManaged(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    fill_trig_tables<<<1, numangle>>>(sinvalues, cosvalues, min_theta, theta_step, numangle, irho);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (N + 15) / 16);
    extract_non_zero<<<gridDim, blockDim>>>(input, d_x_coords, d_y_coords, d_count, N);
    
    int h_count;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    int block = 256;
    int grid = (h_count + block - 1) / block;
    fill_accum<<<grid, block>>>(accum, d_x_coords, d_y_coords, h_count, cosvalues, sinvalues, numangle, numrho);

    dim3 block_max(16, 16);
    dim3 grid_max((numangle+15)/16, (numrho+15)/16);
    find_maxims<<<grid_max, block_max>>>(accum, numangle, numrho, threshold, min_theta, theta_step, rho, d_lines, d_counter);

    cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    line* result = (line*)malloc(counter * sizeof(line));
    cudaMemcpy(result, d_lines, counter * sizeof(line), cudaMemcpyDeviceToHost);

    auto stop_count = omp_get_wtime();
    if (duration) *duration += stop_count - start_count;

    cudaFree(input);
    cudaFree(accum);
    cudaFree(sinvalues);
    cudaFree(cosvalues);
    cudaFree(d_lines);
    cudaFree(d_counter);
    cudaFree(d_x_coords);
    cudaFree(d_y_coords);
    cudaFree(d_count);

    
    return {counter, result};
}

std::pair<int,line*> hough_parallel_segmented(unsigned char* d_img, int N, int threshold,
    float rho, float theta_step,double *duration,
    float min_theta=0.0, float max_theta=CV_PI) {

    unsigned char *input,*sub1, *sub2, *sub3, *sub4;    
    cudaStream_t streams[4];
    for (int i = 0; i < 4; ++i) cudaStreamCreate(&streams[i]);

    cudaMalloc(&input, N*N);
    cudaMemcpyAsync(input, d_img, N*N, cudaMemcpyHostToDevice, streams[0]);

    auto start_count = omp_get_wtime();
    const float irho = 1.f / rho;
    
    const int height = N;
    const int width = N;
    const int halfN = N / 2;
    const int max_rho = width + height;

    const int numangle = (int)cvFloor((max_theta - min_theta) / theta_step) + 1;
    const int numrho = cvRound(((max_rho*2+1))*irho);
    const int accu_size = (numangle+2)*(numrho+2);

    int *acc1, *acc2, *acc3, *acc4;
    float *sinvalues, *cosvalues;
    line *d_lines;
    int *d_counter, counter = 0;

    cudaMalloc(&sub1, halfN*halfN);
    cudaMalloc(&sub2, halfN*halfN);
    cudaMalloc(&sub3, halfN*halfN);
    cudaMalloc(&sub4, halfN*halfN);

    cudaMalloc(&acc1, accu_size * sizeof(int));
    cudaMalloc(&acc2, accu_size * sizeof(int));
    cudaMalloc(&acc3, accu_size * sizeof(int));
    cudaMalloc(&acc4, accu_size * sizeof(int));

    cudaMalloc(&sinvalues, numangle * sizeof(float));
    cudaMalloc(&cosvalues, numangle * sizeof(float));

    cudaMalloc(&d_lines, accu_size * sizeof(line));
    cudaMalloc(&d_counter, sizeof(int));

    cudaMemsetAsync(acc1, 0, accu_size * sizeof(int), streams[0]);
    cudaMemsetAsync(acc2, 0, accu_size * sizeof(int), streams[1]);
    cudaMemsetAsync(acc3, 0, accu_size * sizeof(int), streams[2]);
    cudaMemsetAsync(acc4, 0, accu_size * sizeof(int), streams[3]);
    cudaMemsetAsync(d_counter, 0, sizeof(int), streams[0]);

    
    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (N + 15) / 16);
    segment_image<<<gridDim, blockDim, 0, streams[0]>>>(input, sub1, sub2, sub3, sub4, N);
 
    fill_trig_tables<<<(numangle+255)/256, 256, 0, streams[0]>>>(sinvalues, cosvalues, min_theta, theta_step, numangle, irho);

    int MPQ = 32;
    int total_points = halfN * halfN;
    int threads = (total_points + MPQ - 1) / MPQ;

    int max_points = halfN * halfN;
    int *x1, *y1, *x2, *y2, *x3, *y3, *x4, *y4;
    int *cnt1, *cnt2, *cnt3, *cnt4;
    
    cudaMalloc(&x1, max_points * sizeof(int)); cudaMalloc(&y1, max_points * sizeof(int)); cudaMalloc(&cnt1, sizeof(int));
    cudaMalloc(&x2, max_points * sizeof(int)); cudaMalloc(&y2, max_points * sizeof(int)); cudaMalloc(&cnt2, sizeof(int));
    cudaMalloc(&x3, max_points * sizeof(int)); cudaMalloc(&y3, max_points * sizeof(int)); cudaMalloc(&cnt3, sizeof(int));
    cudaMalloc(&x4, max_points * sizeof(int)); cudaMalloc(&y4, max_points * sizeof(int)); cudaMalloc(&cnt4, sizeof(int));

    cudaMemsetAsync(cnt1, 0, sizeof(int), streams[0]);
    cudaMemsetAsync(cnt2, 0, sizeof(int), streams[1]);
    cudaMemsetAsync(cnt3, 0, sizeof(int), streams[2]);
    cudaMemsetAsync(cnt4, 0, sizeof(int), streams[3]);
    
    extract_non_zero_coords<<<gridDim, blockDim,0, streams[0]>>>(sub1, x1, y1, cnt1, halfN, 1, 1);
    extract_non_zero_coords<<<gridDim, blockDim,0, streams[1]>>>(sub2, x2, y2, cnt2, halfN, 0, 1);
    extract_non_zero_coords<<<gridDim, blockDim,0, streams[2]>>>(sub3, x3, y3, cnt3, halfN, 1, 0);
    extract_non_zero_coords<<<gridDim, blockDim,0, streams[3]>>>(sub4, x4, y4, cnt4, halfN, 0, 0);

    fill_accum_from_coords<<<(threads+255)/256, 256, 0, streams[0]>>>(acc1, x1, y1, cnt1, cosvalues, sinvalues, numangle, numrho);
    fill_accum_from_coords<<<(threads+255)/256, 256, 0, streams[1]>>>(acc2, x2, y2, cnt2, cosvalues, sinvalues, numangle, numrho);
    fill_accum_from_coords<<<(threads+255)/256, 256, 0, streams[2]>>>(acc3, x3, y3, cnt3, cosvalues, sinvalues, numangle, numrho);
    fill_accum_from_coords<<<(threads+255)/256, 256, 0, streams[3]>>>(acc4, x4, y4, cnt4, cosvalues, sinvalues, numangle, numrho);
    
    for (int i = 0; i < 4; ++i) cudaStreamSynchronize(streams[i]);

    sum_accumulators<<<(accu_size+255)/256, 256, 0, streams[0]>>>(acc1, acc2, acc3, acc4, accu_size);

    dim3 block_max(16, 16);
    dim3 grid_max((numangle+15)/16, (numrho+15)/16);
    find_maxims<<<grid_max, block_max, 0, streams[0]>>>(acc1, numangle, numrho, threshold, min_theta, theta_step, rho, d_lines, d_counter);

    
    cudaStreamSynchronize(streams[0]);
    
    cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    line* result = (line*)malloc(counter * sizeof(line));
    cudaMemcpy(result, d_lines, counter * sizeof(line), cudaMemcpyDeviceToHost);

    auto stop_count = omp_get_wtime();
    if (duration) *duration += stop_count - start_count;

    for (int i = 0; i < 4; ++i) cudaStreamDestroy(streams[i]);
    cudaFree(sub1); cudaFree(sub2); cudaFree(sub3); cudaFree(sub4);
    cudaFree(acc1); cudaFree(acc2); cudaFree(acc3); cudaFree(acc4);
    cudaFree(sinvalues); cudaFree(cosvalues);
    cudaFree(d_lines); cudaFree(d_counter);

    return {counter, result};
}


int main(int argc, char** argv){

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount == 0){
        std::cout << "No CUDA device found" << std::endl;
        return -1;
    }

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "Cuda device: " << prop.name << std::endl;

    // check if image is provided
    if(argc < 4){
        std::cout << "not enough parametrs" << std::endl;
        return -1;
    }

    std::string filename = argv[1];
    int threshold = std::stoi(argv[2]);
    int N = std::stoi(argv[3]);

    std::string path ="../pictures/" + filename;
    double total_time_opencv_cpu = 0.0; 
    double total_time_opencv_gpu = 0.0;
    double total_time_opencv_gpu_full = 0.0;
    double total_time_mine_gpu_basic = 0.0;
    double total_time_mine_gpu_basic_full = 0.0;
    double total_time_mine_gpu_segmented = 0.0;
    double total_time_mine_gpu_segmented_full = 0.0;

    double total_lines_opencv_cpu = 0.0;
    double total_lines_opencv_gpu = 0.0;
    double total_lines_mine_gpu_basic = 0.0;
    double total_lines_mine_gpu_segmented = 0.0;

    for (int experiment = 0; experiment < N; ++experiment) {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);  
        if (img.empty()) {
            std::cerr << "Failed to load image: " << path << std::endl;
            return -1;
        }

        cv::Mat blur, edges;
        cv::blur(img, blur, cv::Size(5, 5));
        cv::Canny(blur, edges, 50, 150, 3);
        int biggest = std::max(img.rows, img.cols);

        std::vector<cv::Vec2f> lines;
        auto start_cpu = omp_get_wtime();
        cv::HoughLines(edges, lines, 1, CV_PI/180, threshold);
        auto stop_cpu = omp_get_wtime();
        double duration_cpu = stop_cpu - start_cpu;
        total_time_opencv_cpu += duration_cpu;
        total_lines_opencv_cpu += lines.size();

        cv::cuda::GpuMat img_gpu, img_lines_gpu;
        cv::Mat img_lines_cpu;
        auto cv_hough = cv::cuda::createHoughLinesDetector(1, CV_PI/180, threshold);

        auto start_gpu_full = omp_get_wtime();
        img_gpu.upload(edges);
        auto start_gpu = omp_get_wtime();
        cv_hough->detect(img_gpu, img_lines_gpu);
        img_lines_gpu.download(img_lines_cpu);
        auto stop_gpu = omp_get_wtime();
        double duration_gpu = stop_gpu - start_gpu;
        double duration_gpu_full = stop_gpu - start_gpu_full;
        total_time_opencv_gpu += duration_gpu;
        total_time_opencv_gpu_full += duration_gpu_full;
        total_lines_opencv_gpu += img_lines_gpu.cols;

        unsigned char *d_img = edges.ptr();
        int N = edges.rows;

        auto start_mine_basic = omp_get_wtime();
        std::pair<int,line*> result_basic = hough_parallel(d_img, N, threshold, 1, CV_PI/180,&total_time_mine_gpu_basic);
        auto stop_mine_basic = omp_get_wtime();
        total_time_mine_gpu_basic_full += (stop_mine_basic - start_mine_basic);
        total_lines_mine_gpu_basic += result_basic.first;
        delete[] result_basic.second;

        auto start_mine_segmented = omp_get_wtime();
        std::pair<int,line*> result = hough_parallel_segmented(d_img, N, threshold, 1, CV_PI/180,&total_time_mine_gpu_segmented);
        auto stop_mine_segmented = omp_get_wtime();
        total_lines_mine_gpu_segmented += result.first;
        total_time_mine_gpu_segmented_full += (stop_mine_segmented - start_mine_segmented);
        delete[] result.second;
    }

    std::cout << "\n=== AVERAGE TIMES OVER " << N << " EXPERIMENTS ===\n";
    std::cout << "OpenCV CPU Hough: " << (total_time_opencv_cpu / N)*1000.0 << " ms\n" << "Lines found: " << total_lines_opencv_cpu / N << "\n";
    std::cout << "OpenCV GPU Hough (kernel only): " << (total_time_opencv_gpu / N)*1000.0 << "ms\n" << "Lines found: " << total_lines_opencv_gpu / N << "\n";
    std::cout << "Mine GPU Hough (basic): " << (total_time_mine_gpu_basic / N)*1000.0 << "ms\n" << "Lines found: " << total_lines_mine_gpu_basic / N << "\n";
    std::cout << "Mine GPU Hough (segmented): " << (total_time_mine_gpu_segmented / N)*1000.0 << "ms\n" << "Lines found: " << total_lines_mine_gpu_segmented / N << "\n";
    std::cout << "OpenCV GPU Hough (full incl. transfer): " << (total_time_opencv_gpu_full / N)*1000.0 << "ms\n" << "Lines found: " << total_lines_opencv_gpu / N << "\n";
    std::cout << "Mine GPU Hough (basic incl. transfer): " << (total_time_mine_gpu_basic_full / N)*1000.0 << "ms\n" << "Lines found: " << total_lines_mine_gpu_basic / N << "\n";
    std::cout << "Mine GPU Hough (segmented incl. transfer): " << (total_time_mine_gpu_segmented_full / N)*1000.0 << "ms\n" << "Lines found: " << total_lines_mine_gpu_segmented / N << "\n";

    return 0;
}
