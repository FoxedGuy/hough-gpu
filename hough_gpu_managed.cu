#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include "omp.h"

#define CUDA_CHECK(call)                                                   \
    {                                                                      \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA Error: %s (error code %d), at %s:%d\n",  \
                    cudaGetErrorString(err), err, __FILE__, __LINE__);     \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    }

struct line{
    int rho;
    float theta;
};


int compute_angle(float min_theta, float max_theta, float theta_step){
    int numangle = cvFloor((max_theta - min_theta) / theta_step) + 1;
    if ( numangle > 1 && fabs(CV_PI - (numangle-1)*theta_step) < theta_step/2 ) --numangle;
    return numangle;
}

__global__ void fill_trig_tables(float *sin_table, float *cos_table, float min_theta, float theta, int num_angle, float irho){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < num_angle){
        float angle = min_theta + index * theta;
        sin_table[index] = sinf(angle) *irho;
        cos_table[index] = cosf(angle) *irho;
    }
}

__global__ void fill_accum(int *accum, unsigned char *data, float *cos_table, float *sin_table,
                           int width, int height, int num_angle, int num_rho){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if(y >= height || x >= width) return;
    if(data[y * width + x] != 0){
        for(int angle = 0; angle < num_angle; angle++){
            int r = roundf(x * cos_table[angle] + y * sin_table[angle]);
            r+= (num_rho - 1) / 2.f;
            atomicAdd(&accum[(angle+1) * (num_rho+2) + r + 1], 1);
        }
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

std::pair<int,line*> hough_parallel(cv::Mat img, int threshold,
                    float rho, float theta, float min_theta=0.0, float max_theta=CV_PI){

    float irho = 1.f / rho;
    int height = img.rows;
    int width = img.cols;
    size_t image_size = width*height*sizeof(unsigned char);
    int rho_max = width + height;
    int rho_min = -rho_max;

    // Compute the number of angles and rhos
    int num_angle = compute_angle(min_theta, max_theta, theta);
    int num_rho = round(((rho_max - rho_min)) *  irho);

    // Allocate for gpu:
    // Trygonometric tables, accumulator array
    // Copy image data from cpu to gpu
    int *accum = nullptr;
    float *sin_table = nullptr, *cos_table = nullptr;
    unsigned char *data = nullptr;
    int* counter = nullptr;
    line* lines = nullptr;

    cudaMallocManaged(&sin_table, num_angle*sizeof(float));
    cudaMallocManaged(&cos_table, num_angle*sizeof(float));
    cudaMallocManaged(&accum, (num_angle+2)*(num_rho+2)*sizeof(int));
    cudaMallocManaged(&data,image_size);
    cudaMallocManaged(&lines,(num_angle+2)*(num_rho+2)*sizeof(line));
    cudaMallocManaged(&counter, sizeof(int));
    
    memset(accum,0,(num_angle+2)*(num_rho+2)*sizeof(int));
    memcpy(data,img.ptr(),image_size);
    *counter = 0;
    
    int threadsPerBlock = 256;
    int blocksPerGridTrig = (num_angle + threadsPerBlock - 1) / threadsPerBlock;
    dim3 threadsPerBlockAccum(16, 16);
    dim3 blocksPerGridAccum((width + threadsPerBlockAccum.x - 1) / threadsPerBlockAccum.x,
                       (height + threadsPerBlockAccum.y - 1) / threadsPerBlockAccum.y);
    dim3 threadsPerBlockMaxims(16, 16);
    dim3 blocksPerGridMaxims(((num_angle+2) + threadsPerBlockMaxims.x - 1) / threadsPerBlockMaxims.x,
                       ((num_rho+2) + threadsPerBlockMaxims.y - 1) / threadsPerBlockMaxims.y);
                   

    auto start = omp_get_wtime();    

    fill_trig_tables<<<blocksPerGridTrig, threadsPerBlock>>>(sin_table, cos_table, min_theta, theta, num_angle, irho);    
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    fill_accum<<<blocksPerGridAccum, threadsPerBlockAccum>>>(accum, data, cos_table, sin_table, width, height, num_angle, num_rho);    
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    find_maxims<<<blocksPerGridMaxims, threadsPerBlockMaxims>>>(accum, num_angle, num_rho, threshold, min_theta, theta, rho, lines, counter);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    
    auto stop = omp_get_wtime();
    std::cout << "\n====GPU PARAMS===\nTime for calculations: " << stop-start << std::endl;
    
    int size = *counter;
    
    cudaFree(sin_table);
    cudaFree(cos_table);
    cudaFree(accum);
    cudaFree(data);
    cudaFree(counter);
    return std::pair<int,line*>(size,lines);
}


int main(int argc, char** argv){

    // check if cuda device available
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount == 0){
        std::cout << "No CUDA device found" << std::endl;
        return -1;
    }

    // get device properties
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
    cv::Mat img = cv::imread(path, 1);
    cv::Mat img_blur, img_dst;
    img_dst = img.clone();

    int biggest = std::max(img.rows, img.cols);

    cv::Mat dst;
    cv::blur(img, img_blur, cv::Size(5,5));
    cv::Canny(img_blur, dst, 100, 150, 3);
    cv::imwrite("../results/edges.jpg", dst);

    std::vector<cv::Vec2f> lines;
    auto start = omp_get_wtime();
    cv::HoughLines(dst, lines, 1, CV_PI/180, threshold);
    auto stop = omp_get_wtime();
    auto duration = stop-start;

    std::cout << "====CV PARAMS====\nTime taken: " << duration << "\nDetected lines: " << lines.size() << '\n';

    start = omp_get_wtime();
    std::pair<int,line*> result = hough_parallel(dst, threshold, 1, CV_PI/180);
    stop = omp_get_wtime();
    duration = stop-start;

    int size = result.first;
    line * lines_mine = nullptr;
    lines_mine = result.second;

    std::cout << "\nTime taken with copying: " << duration << "\nDetected lines: " << size << "\n\nDrawing lines for cv and gpu...";

    // for( size_t i = 0; i < lines.size(); i++ ){
    //     float rho = lines[i][0], theta = lines[i][1];
    //     cv::Point pt1, pt2;
    //     double a = cos(theta), b = sin(theta);
    //     double x0 = a*rho, y0 = b*rho;
    //     pt1.x = cvRound(x0 + biggest*(-b));
    //     pt1.y = cvRound(y0 + biggest*(a));
    //     pt2.x = cvRound(x0 - biggest*(-b));
    //     pt2.y = cvRound(y0 - biggest*(a));
    //     cv::line(img, pt1, pt2, cv::Scalar(0,0,255), 2, cv::LINE_AA);
    // }

    // for (int i  = 0; i < size ; i++) {
    //     float theta = lines_mine[i].theta;
    //     int rho = lines_mine[i].rho;
    //     double a = std::cos(theta);
    //     double b = std::sin(theta);
    //     double x0 = a * rho;
    //     double y0 = b * rho;
    //     cv::Point pt1(cvRound(x0 + biggest * (-b)), cvRound(y0 + biggest * (a)));
    //     cv::Point pt2(cvRound(x0 - biggest * (-b)), cvRound(y0 - biggest * (a)));
    //     cv::line(img_dst, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    // }
    std::cout << " done";

    std::cout << "\nSaving both results...";
    // cv::imwrite("../results/result_mine.png",img_dst);
    // cv::imwrite("../results/result_cv.png",img);
    std::cout << " done, exiting\n";

    cudaFree(lines_mine);

    return 0;

}
