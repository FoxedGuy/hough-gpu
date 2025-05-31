#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include "omp.h"

struct line{
    int rho;
    float theta;
};



int compute_angle(float min_theta, float max_theta, float theta_step){
    int numangle = cvFloor((max_theta - min_theta) / theta_step) + 1;
    if ( numangle > 1 && fabs(CV_PI - (numangle-1)*theta_step) < theta_step/2 ) --numangle;
    return numangle;
}

__global__ void segmentImage(const unsigned char* inputImage,
    unsigned char* subImage1,
    unsigned char* subImage2,
    unsigned char* subImage3,
    unsigned char* subImage4,
    int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    int halfN = N / 2;
    if (row < halfN) {
        for (int col = 0; col < halfN; col++) {
            subImage1[row * halfN + col] = inputImage[row * N + col];
            subImage2[row * halfN + col] = inputImage[row * N + halfN + col];
        }
    } else {
        int adjustedRow = row - halfN;
        for (int col = 0; col < halfN; col++) {
            subImage3[adjustedRow * halfN + col] = inputImage[row * N + col];
            subImage4[adjustedRow * halfN + col] = inputImage[row * N + halfN + col];
        }
    }
}

__global__ void extract_nonzero(unsigned char* img, int width, int height, int *x_cords, int *y_cords, int *count, int x_offset, int y_offset){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >=  width * height) return;

    int x = idx % width;
    int y = idx / height;

    if (img[y*width+x] != 0){
        int index = atomicAdd(count, 1);
        x_cords[index] = x + x_offset;
        y_cords[index] = y + y_offset;
    }
}

__global__ void sum_accumulators(int* accum1, int* accum2, int* accum3, int* accum4, int* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = accum1[idx] + accum2[idx] + accum3[idx] + accum4[idx];
    }
}

__global__ void fill_trig_tables(float *sin_table, float *cos_table, float min_theta, float theta, int num_angle, float irho){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < num_angle){
        float angle = min_theta + index * theta;
        sin_table[index] = sinf(angle) *irho;
        cos_table[index] = cosf(angle) *irho;
    }
}

__global__ void fill_accum_from_coords(int* accum,int* x_coords,int* y_coords,int *num_points,float* cos_table,float* sin_table,int num_angle,int num_rho, int x_offset, int y_offset){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *num_points) return;

    int x = x_coords[idx];
    int y = y_coords[idx];

    for (int angle = 0; angle < num_angle; angle++) {
        int r = roundf(x * cos_table[angle] + y * sin_table[angle]);
        r += (num_rho - 1) / 2.f;
        atomicAdd(&accum[(angle + 1) * (num_rho + 2) + r + 1], 1);
    }
}

__global__ void find_maxims(int* accum, int num_angle, int num_rho, int threshold, float min_theta, float theta_step, float rho_step, line* lines, int *current_size){
    int angle = blockDim.x * blockIdx.x + threadIdx.x;
    int rho = blockDim.y * blockIdx.y + threadIdx.y;
    if(angle >= num_angle || rho >= num_rho) return;
    int base = (angle+1) * (num_rho+2) + rho + 1;
    if( accum[base] > threshold &&
        accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] &&
        accum[base] > accum[base - num_rho - 2] && accum[base] >= accum[base + num_rho + 2]){
        int real_rho = (rho - (num_rho - 1) *0.5f) * rho_step;
        float real_theta = min_theta + angle * theta_step;
        int index = atomicAdd(current_size,1);
        lines[index].rho = real_rho;
        lines[index].theta = real_theta;
    }
}

std::pair<int,line*> hough_parallel_segmented(unsigned char* d_img, int N, int threshold,
    float rho, float theta_step,
    float min_theta=0.0, float max_theta=CV_PI) {

    float irho = 1.f / rho;
    int height = N;
    int width = N;
    int halfN = N / 2;

    int rho_max = width + height;
    int rho_min = -rho_max;

    int num_angle = (int)floor((max_theta - min_theta) / theta_step) + 1;
    int num_rho = round(((rho_max - rho_min)) * irho);

    int acc_size = (num_angle+2)*(num_rho+2);

    unsigned char *input,*sub1, *sub2, *sub3, *sub4;
    int *acc1, *acc2, *acc3, *acc4, *accum_total;
    float *dsin_table, *dcos_table;
    line *d_lines;
    int *d_counter, counter = 0;

    // ==== MALLOC SECTION ====
    
    // subimages
    cudaMalloc(&input, N*N);
    cudaMalloc(&sub1, halfN*halfN);
    cudaMalloc(&sub2, halfN*halfN);
    cudaMalloc(&sub3, halfN*halfN);
    cudaMalloc(&sub4, halfN*halfN);

    // accumulators
    cudaMalloc(&acc1, acc_size * sizeof(int));
    cudaMalloc(&acc2, acc_size * sizeof(int));
    cudaMalloc(&acc3, acc_size * sizeof(int));
    cudaMalloc(&acc4, acc_size * sizeof(int));
    cudaMalloc(&accum_total, acc_size * sizeof(int));

    // trigonometric tables
    cudaMalloc(&dsin_table, num_angle * sizeof(float));
    cudaMalloc(&dcos_table, num_angle * sizeof(float));

    // lines
    cudaMalloc(&d_lines, acc_size * sizeof(line));
    cudaMalloc(&d_counter, sizeof(int));

    // Create separate streams for voting phase
    cudaStream_t streams[4];
    for (int i = 0; i < 4; ++i) cudaStreamCreate(&streams[i]);

    // Set values to zero
    cudaMemsetAsync(acc1, 0, acc_size * sizeof(int), streams[0]);
    cudaMemsetAsync(acc2, 0, acc_size * sizeof(int), streams[1]);
    cudaMemsetAsync(acc3, 0, acc_size * sizeof(int), streams[2]);
    cudaMemsetAsync(acc4, 0, acc_size * sizeof(int), streams[3]);
    cudaMemsetAsync(accum_total, 0, acc_size * sizeof(int), streams[0]);
    cudaMemsetAsync(d_counter, 0, sizeof(int), streams[0]);
    
    // Copy image data to device
    cudaMemcpyAsync(input, d_img, N*N, cudaMemcpyHostToDevice, streams[0]);
    auto start = omp_get_wtime();

    segmentImage<<<(N+255)/256, 256, 0, streams[0]>>>(input, sub1, sub2, sub3, sub4, N);
 
    fill_trig_tables<<<(num_angle+255)/256, 256, 0, streams[0]>>>(dsin_table, dcos_table, min_theta, theta_step, num_angle, irho);

    int MPQ = 32;
    int total_points = halfN * halfN;
    int threads = (total_points + MPQ - 1) / MPQ;

    //==== EXTRACTING POINTS ====
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
    
    int threads_extract = (max_points + 255) / 256;
    extract_nonzero<<<threads_extract, 256, 0, streams[0]>>>(sub1, halfN, halfN, x1, y1, cnt1, 0, 0);
    extract_nonzero<<<threads_extract, 256, 0, streams[1]>>>(sub2, halfN, halfN, x2, y2, cnt2, halfN, 0);
    extract_nonzero<<<threads_extract, 256, 0, streams[2]>>>(sub3, halfN, halfN, x3, y3, cnt3, 0, halfN);
    extract_nonzero<<<threads_extract, 256, 0, streams[3]>>>(sub4, halfN, halfN, x4, y4, cnt4, halfN, halfN);

    fill_accum_from_coords<<<(threads+255)/256, 256, 0, streams[0]>>>(acc1, x1, y1, cnt1, dcos_table, dsin_table, num_angle, num_rho, 0, 0);
    fill_accum_from_coords<<<(threads+255)/256, 256, 0, streams[1]>>>(acc2, x2, y2, cnt2, dcos_table, dsin_table, num_angle, num_rho, halfN, 0);
    fill_accum_from_coords<<<(threads+255)/256, 256, 0, streams[2]>>>(acc3, x3, y3, cnt3, dcos_table, dsin_table, num_angle, num_rho, 0, halfN);
    fill_accum_from_coords<<<(threads+255)/256, 256, 0, streams[3]>>>(acc4, x4, y4, cnt4, dcos_table, dsin_table, num_angle, num_rho, halfN, halfN);
    
    for (int i = 0; i < 4; ++i) cudaStreamSynchronize(streams[i]);

    sum_accumulators<<<(acc_size+255)/256, 256, 0, streams[0]>>>(acc1, acc2, acc3, acc4, accum_total, acc_size);

    dim3 block_max(16, 16);
    dim3 grid_max((num_angle+15)/16, (num_rho+15)/16);
    find_maxims<<<grid_max, block_max, 0, streams[0]>>>(accum_total, num_angle, num_rho, threshold, min_theta, theta_step, rho, d_lines, d_counter);

    auto stop = omp_get_wtime();
    std::cout << "\nTime for calculations: " << stop-start << "\n";
    cudaStreamSynchronize(streams[0]);
    cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    line* result = (line*)malloc(counter * sizeof(line));
    cudaMemcpy(result, d_lines, counter * sizeof(line), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; ++i) cudaStreamDestroy(streams[i]);
    cudaFree(sub1); cudaFree(sub2); cudaFree(sub3); cudaFree(sub4);
    cudaFree(acc1); cudaFree(acc2); cudaFree(acc3); cudaFree(acc4); cudaFree(accum_total);
    cudaFree(dsin_table); cudaFree(dcos_table);
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
    if(argc < 3){
        std::cout << "image not provided" << std::endl;
        return -1;
    }

    std::string filename = argv[1];
    int threshold = std::stoi(argv[2]);

    std::string path ="../pictures/" + filename;
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);  
    if (img.empty()) {
        std::cerr << "Failed to load image: " << path << std::endl;
        return -1;
    }

    cv::Mat blur = img.clone();
    cv::Mat edges = img.clone();
    cv::Mat img_dst = img.clone();
    cv::GaussianBlur(img, blur, cv::Size(5, 5), 1.5);
    cv::Canny(blur, edges, 50, 150, 3);


    cv::imwrite("../results/edges.jpg", edges);

    int biggest = std::max(img.rows, img.cols);

    std::vector<cv::Vec2f> lines;
    auto start = omp_get_wtime();
    cv::HoughLines(edges, lines, 1, CV_PI/180, threshold);
    auto stop = omp_get_wtime();
    
    auto duration = stop-start;

    std::cout << "====CPU PARAMS====\nTime taken: " << duration << "\nDetected lines: " << lines.size() << '\n';

    start = omp_get_wtime();
    unsigned char *d_img = edges.ptr();
    int N = edges.rows;
    std::cout << "====GPU PARAMS====";
    std::pair<int,line*> result = hough_parallel_segmented(d_img, N, threshold, 1, CV_PI/180);
    stop = omp_get_wtime();
    duration = stop-start;

    int size = result.first;
    line * lines_mine = nullptr;
    lines_mine = result.second;

    std::cout << "\nTime taken with copying: " << duration << "\nDetected lines: " << size << "\n\nDrawing lines for cv and gpu...";
    std::cout << " done";

    std::cout << "\nSaving both results...";
    cv::imwrite("../results/lines/gpu/result_mine.png",img_dst);
    cv::imwrite("../results/lines/gpu/result_cv.png",img);
    std::cout << " done, exiting\n";

    delete [] lines_mine;

    return 0;

}
