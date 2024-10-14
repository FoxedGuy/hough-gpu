#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <chrono>

using namespace cv;

int compute_angle(double min_theta, double max_theta, double theta){
    int numangle = cvFloor((max_theta - min_theta) / theta) + 1;
    if ( numangle > 1 && fabs(CV_PI - (numangle-1)*theta) < theta/2 )
        --numangle;
    return numangle;
}


__global__ void fill_trig_tables(float *sin_table, float *cos_table, double min_theta, double theta, int num_angle){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < num_angle){
        double angle = min_theta + index * theta;
        sin_table[index] = sinf(angle);
        cos_table[index] = cosf(angle);
    }
}

__global__ void fill_accum(int *accum, uchar *data, float *cos_table, float *sin_table,
                           int width, int height, int num_angle, int num_rho){
    int y = blockDim.x * blockIdx.x + threadIdx.x;
    int x = blockDim.y * blockIdx.y + threadIdx.y;
    if(y < height && x < width){
        if(data[y * width + x] > 0){
            for(int angle = 0; angle < num_angle; angle++){
                int r = roundf(x * cos_table[angle] + y * sin_table[angle]);
                r+= roundf((num_rho - 1)) / 2;
                atomicAdd(&accum[angle * (num_rho+2) + r + 1], 1);
            }
        }
    }
}

void hough_parallel(Mat img, std::vector<std::pair<double,double>>& lines, int threshold,
                    double rho, double theta, double min_theta, double max_theta){
    
    float irho = 1.f / rho;
    int height = img.rows;
    int width = img.cols;
    int rho_max = width + height;
    int rho_min = -rho_max;

    int num_angle = compute_angle(min_theta, max_theta, theta);
    int num_rho = cvRound(((rho_max - rho_min) + 1) /  rho);

    int *accum = new int[num_angle * num_rho]();
    
    float *sin_table = new float[num_angle];
    float *cos_table = new float[num_angle];

    float *dsin_table = nullptr, *dcos_table = nullptr;

    cudaMalloc(&dsin_table, num_angle * sizeof(float));
    cudaMalloc(&dcos_table, num_angle * sizeof(float));

    int threadsPerBlock = 256;
    int blocksPerGridTrig = (num_angle + threadsPerBlock - 1) / threadsPerBlock;

    fill_trig_tables<<<blocksPerGridTrig, threadsPerBlock>>>(dsin_table, dcos_table, min_theta, theta, num_angle);

    cudaMemcpy(sin_table, dsin_table, num_angle * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cos_table, dcos_table, num_angle * sizeof(float), cudaMemcpyDeviceToHost);

    dim3 threadsPerBlockAccum(16, 16); 
    dim3 blocksPerGridAccum((height + threadsPerBlockAccum.x - 1) / threadsPerBlockAccum.x,
                       (width + threadsPerBlockAccum.y - 1) / threadsPerBlockAccum.y);
 
    // fill_accum<<<blocksPerGridAccum, threadsPerBlockAccum>>>(accum, img.data, dsin_table, dcos_table, width, height, num_angle, num_rho);


    cudaFree(dsin_table);
    cudaFree(dcos_table);
    delete sin_table;
    delete cos_table;
}


int main(){
    std::string filename;
    std::cout << "Enter the filename: ";
    std::cin >> filename;
    std::string path ="../pictures/" + filename;
    Mat img = imread(path, 0);
    
    if(img.empty()){
        std::cout << "Image not found" << std::endl;
        return -1;
    }
    
    Mat dst, cdst;
    Canny(img, dst, 50, 200, 3);
    std::vector<Vec2f> lines;
    std::cout << "Starting cv sequential implementation" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    HoughLines(dst, lines, 1, CV_PI/180, 100, 0, 0);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Done! \n Time taken: " << duration.count() << " microseconds" << std::endl;

    cdst = dst.clone();

    std::cout << "Starting my gpu implementation" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<double,double>> lines_mine;
    hough_parallel(cdst, lines_mine, 100, 1, CV_PI/180, 0, CV_PI);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Done! \n Time taken: " << duration.count() << " microseconds" << std::endl;

}
