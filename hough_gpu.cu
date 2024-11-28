#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <chrono>


struct line
{
    int rho;
    float theta;
};


int compute_angle(double min_theta, double max_theta, double theta_step){
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
            r+= (num_rho - 1) / 2;
            atomicAdd(&accum[(angle+1) * (num_rho+2) + r + 1], 1);
        }
    }
}

__global__ void find_maxims(int* accum, int num_angle, int num_rho, int threshold, double min_theta, double theta_step, double rho_step, line* lines, int *current_size){
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
                    float rho, float theta, double min_theta=0.0, double max_theta=CV_PI){
    
    float irho = 1 / rho;
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
    int *daccum = nullptr;
    float *dsin_table = nullptr, *dcos_table = nullptr;
    unsigned char *ddata = nullptr;
    cudaMalloc(&dsin_table, num_angle * sizeof(float));
    cudaMalloc(&dcos_table, num_angle * sizeof(float));
    cudaMalloc(&daccum, (num_angle+2) * (num_rho+2) * sizeof(int));
    cudaMemset(daccum, 0, (num_angle+2) * (num_rho+2) * sizeof(int));
    cudaMalloc(&ddata,image_size);
    cudaMemcpy(ddata, img.ptr(), image_size, cudaMemcpyHostToDevice);
    
    
    // ============================PRECOMPUTE TRIG TABLES=====================================

    int threadsPerBlock = 256;
    int blocksPerGridTrig = (num_angle + threadsPerBlock - 1) / threadsPerBlock;

    fill_trig_tables<<<blocksPerGridTrig, threadsPerBlock>>>(dsin_table, dcos_table, min_theta, theta, num_angle, irho);
    cudaDeviceSynchronize();

    // ============================FILL ACCUMULATOR==================================

    dim3 threadsPerBlockAccum(16, 16); 
    dim3 blocksPerGridAccum((width + threadsPerBlockAccum.x - 1) / threadsPerBlockAccum.x,
                       (height + threadsPerBlockAccum.y - 1) / threadsPerBlockAccum.y);

    fill_accum<<<blocksPerGridAccum, threadsPerBlockAccum>>>(daccum, ddata, dcos_table, dsin_table, width, height, num_angle, num_rho);
    cudaDeviceSynchronize();

    // ===========================FIND MAXIMS======================================
    dim3 threadsPerBlockMaxims(16, 16); 
    dim3 blocksPerGridMaxims(((num_angle+2) + threadsPerBlockMaxims.x - 1) / threadsPerBlockMaxims.x,
                       ((num_rho+2) + threadsPerBlockMaxims.y - 1) / threadsPerBlockMaxims.y);

    int *d_counter = nullptr;
    int counter = 0;
    line* lines = nullptr;
    cudaMalloc(&lines, (num_angle+2)*(num_rho+2)*sizeof(line));
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemcpy(d_counter, &counter, sizeof(int), cudaMemcpyHostToDevice);
    
    find_maxims<<<blocksPerGridMaxims, threadsPerBlockMaxims>>>(daccum, num_angle, num_rho, threshold, min_theta, theta, rho, lines, d_counter);
    
    cudaDeviceSynchronize();
    cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    line* lines_cpu = (line*)malloc(counter*sizeof(line));
    cudaMemcpy(lines_cpu, lines, counter*sizeof(line), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(dsin_table);
    cudaFree(dcos_table);
    cudaFree(daccum);
    cudaFree(ddata);
    cudaFree(lines);
    return std::pair<int,line*>(counter,lines_cpu);
}


int main(){
    std::string filename;
    std::cout << "Enter the filename: ";
    std::cin >> filename;
    std::string path ="../pictures/" + filename;
    cv::Mat img = cv::imread(path, 1);
    cv::Mat img_blur, img_dst; 
    img_dst = img.clone();
    int device = 0;
    cudaError_t error = cudaGetDeviceCount(&device);

    if (error != cudaSuccess) {
        std::cerr << "Error getting CUDA device count: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    
    std::cout << "CUDA Device Count: " << device << std::endl;


    if(img.empty()){
        std::cout << "Image not found" << std::endl;
        return -1;
    }

    int biggest = std::max(img.rows, img.cols);
    
    cv::Mat dst;
    cv::blur(img, img_blur, cv::Size(5,5));
    cv::Canny(img_blur, dst, 100, 150, 3);
    cv::imwrite("../results/edges.jpg", dst);

    std::vector<cv::Vec2f> lines;
    std::cout << "Starting cv sequential implementation" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    cv::HoughLines(dst, lines, 1, CV_PI/180, 100);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Done! \nTime taken: " << duration.count() << " microseconds" << '\n' << "Lines detected: " << lines.size() << std::endl;
    
    for( size_t i = 0; i < lines.size(); i++ ){
        float rho = lines[i][0], theta = lines[i][1];
        cv::Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + biggest*(-b));
        pt1.y = cvRound(y0 + biggest*(a));
        pt2.x = cvRound(x0 - biggest*(-b));
        pt2.y = cvRound(y0 - biggest*(a));
        cv::line(img, pt1, pt2, cv::Scalar(0,0,255), 2, cv::LINE_AA);
    }

    std::cout << "=======================================================\n";
    std::cout << "Starting my gpu implementation" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    std::pair<int,line*> result = hough_parallel(dst, 150, 1, CV_PI/180);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    int size = result.first;
    line * lines_mine = nullptr;
    lines_mine = result.second;
    
    std::cout << "Done! \nTime taken: " << duration.count() << " microseconds" << std::endl;
    std::cout << size << "\n";
    for (int i  = 0; i < size ; i++) {
        float theta = lines_mine[i].theta;
        int rho = lines_mine[i].rho;
        double a = std::cos(theta);
        double b = std::sin(theta);
        double x0 = a * rho;
        double y0 = b * rho;
        cv::Point pt1(cvRound(x0 + biggest * (-b)), cvRound(y0 + biggest * (a)));
        cv::Point pt2(cvRound(x0 - biggest * (-b)), cvRound(y0 - biggest * (a)));
        cv::line(img_dst, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    } 

    cv::imwrite("../results/result_mine.png",img_dst);
    cv::imwrite("../results/result_cv.png",img);

    delete [] lines_mine;

    return 0;

}
