#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/cudaimgproc.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include "omp.h"

struct line {
    float rho;
    float theta;
};

int computeNumangle(double min_theta, double max_theta, double theta_step) {
    int numangle = cvFloor((max_theta - min_theta) / theta_step) + 1;

    if (numangle > 1 && fabs(CV_PI - (numangle - 1) * theta_step) < theta_step / 2)
        --numangle;
    return numangle;
}

void saveAccumulatorImage(int* accu,
                          int numangle,
                          int numrho,
                          const std::string& filename) {
    cv::Mat accuImage(numangle, numrho, CV_32SC1);

    for (int angle = 0; angle < numangle; ++angle) {
        for (int rho = 0; rho < numrho; ++rho) {
            int value = accu[(angle + 1) * (numrho + 2) + (rho + 1)];
            accuImage.at<int>(angle, rho) = value;
        }
    }

    cv::Mat normalized;
    cv::normalize(accuImage, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::imwrite("../results/" + filename, normalized);

    int scaleFactor = 7;
    cv::Mat resized;
    cv::resize(normalized, resized, cv::Size(normalized.cols / scaleFactor, normalized.rows), cv::INTER_AREA);

    cv::imwrite("../results/accumulator_squeezed.png", resized);
}

void compute_trig_cpu(float* cosvalues,
                      float* sinvalues,
                      int numangle,
                      float min_theta,
                      float theta,
                      float irho) {
    for (int angle = 0; angle < numangle; angle++) {
        float ang = min_theta + angle * theta;
        cosvalues[angle] = std::cos(ang) * irho;
        sinvalues[angle] = std::sin(ang) * irho;
    }
}

void fill_accum_cpu(int* accu,
                    const std::vector<cv::Point>& points,
                    float* cosvalues,
                    float* sinvalues,
                    int numangle,
                    int numrho) {
    for (const auto& pt : points) {
        for (int angle = 0; angle < numangle; angle++) {
            int r = cvRound(pt.x * cosvalues[angle] + pt.y * sinvalues[angle]);
            r += (numrho - 1) / 2;
            accu[(angle + 1) * (numrho + 2) + (r + 1)]++;
        }
    }
}

void find_maxims_cpu(int* accu,
                     int numangle,
                     int numrho,
                     int threshold,
                     float min_theta,
                     float theta_step,
                     float rho_step,
                     std::vector<std::pair<float, float>>& lines) {
    for (int angle = 0; angle < numangle; angle++) {
        for (int r = 0; r < numrho; r++) {
            int ind = (angle + 1) * (numrho + 2) + r + 1;

            if (accu[ind] > threshold &&
                accu[ind] > accu[ind - 1] &&
                accu[ind] >= accu[ind + 1] &&
                accu[ind] > accu[ind - numrho - 2] &&
                accu[ind] >= accu[ind + numrho + 2]) {

                float rho_val = (r - (numrho - 1) * 0.5f) * rho_step;
                float angle_val = min_theta + angle * theta_step;

                lines.push_back(std::pair<float, float>(rho_val, angle_val));
            }
        }
    }
}

std::vector<std::pair<float, float>>
hough_transform_cpu(cv::Mat img,
                    int threshold,
                    float rho,
                    float theta,
                    float min_theta = 0.0,
                    float max_theta = CV_PI) {
    const float irho = 1.0f / rho;
    const int width = img.cols;
    const int height = img.rows;
    const int max_rho = width + height;

    const int numangle = computeNumangle(min_theta, max_theta, theta);
    const int numrho = cvRound((max_rho * 2 + 1) * irho);
    const int accu_size = (numangle + 2) * (numrho + 2);

    int* accu = new int[accu_size]{0};
    float* cosvalues = new float[numangle];
    float* sinvalues = new float[numangle];

    compute_trig_cpu(cosvalues, sinvalues, numangle, min_theta, theta, irho);

    std::vector<cv::Point> non_zero;
    cv::findNonZero(img, non_zero);

    fill_accum_cpu(accu, non_zero, cosvalues, sinvalues, numangle, numrho);

    std::vector<std::pair<float, float>> lines;
    find_maxims_cpu(accu, numangle, numrho, threshold, min_theta, theta, rho, lines);

    delete[] cosvalues;
    delete[] sinvalues;
    delete[] accu;

    return lines;
}

__global__ void extract_non_zero_packed(unsigned char* image,
                                        unsigned int* packed_coords,
                                        int* count,
                                        int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= N || y >= N)
        return;

    int idx = y * N + x;

    if (image[idx] != 0) {
        int index = atomicAdd(count, 1);
        packed_coords[index] = (y << 16) | x;
    }
}

__global__ void fill_accum(int* accum,
                           size_t step,
                           unsigned int* x_coords_packed,
                           int* num_points,
                           int numrho,
                           float irho,
                           float min_theta,
                           float theta_step) {
    int ang = blockIdx.x;
    float theta = min_theta + ang * theta_step;

    float cosv, sinv;
    sincosf(theta, &sinv, &cosv);
    sinv *= irho;
    cosv *= irho;

    int shift = (numrho - 1) >> 1;

    char* base = (char*)accum;
    int* row = (int*)(base + (ang + 1) * step);

    for (int i = threadIdx.x; i < *num_points; i += blockDim.x) {
        unsigned int packed = x_coords_packed[i];
        int x = packed & 0xFFFF;
        int y = (packed >> 16) & 0xFFFF;

        int r = __float2int_rn(x * cosv + y * sinv);
        r += shift;

        atomicAdd(&row[r + 1], 1);
    }
}

__global__ void fill_accum_packed(int* accum,
                                  size_t step,
                                  unsigned int* packed_coords,
                                  int* num_points,
                                  int numrho,
                                  float irho,
                                  float min_theta,
                                  float theta_step) {
    extern __shared__ int sh_accum[];

    int tid = threadIdx.x;

    for (int r = tid; r <= numrho + 2; r += blockDim.x) {
        sh_accum[r] = 0;
    }

    __syncthreads();

    int angle = blockIdx.x;
    float theta = min_theta + angle * theta_step;

    float cosv, sinv;
    __sincosf(theta, &sinv, &cosv);
    sinv *= irho;
    cosv *= irho;

    int shift = (numrho - 1) >> 1;

    for (int i = tid; i < *num_points; i += blockDim.x) {
        unsigned int packed = packed_coords[i];
        int x = packed & 0xFFFF;
        int y = (packed >> 16) & 0xFFFF;

        int r = __float2int_rn(x * cosv + y * sinv);
        r += shift;

        atomicAdd(&sh_accum[r + 1], 1);
    }

    __syncthreads();

    char* base = (char*)accum;
    int* row = (int*)(base + (angle + 1) * step);

    for (int i = tid; i < numrho + 1; i += blockDim.x)
        row[i] = sh_accum[i];
}

__global__ void find_maxims(int* accum,
                            size_t step,
                            int numangle,
                            int numrho,
                            int threshold,
                            float min_theta,
                            float theta_step,
                            float rho_step,
                            line* lines,
                            int* current_size) {
    int rho = blockDim.x * blockIdx.x + threadIdx.x;
    int angle = blockDim.y * blockIdx.y + threadIdx.y;

    if (angle >= numangle || rho >= numrho) return;

    int* row_center = (int*)((char*)accum + (angle + 1) * step);
    int* row_up = (int*)((char*)accum + angle * step);
    int* row_down = (int*)((char*)accum + (angle + 2) * step);

    int center = row_center[rho + 1];

    bool is_maximum =
        (center > threshold &&
         center > row_center[rho] &&
         center >= row_center[rho + 2] &&
         center > row_up[rho + 1] &&
         center >= row_down[rho + 1]);

    if (!is_maximum) return;

    int index = atomicAdd(current_size, 1);
    lines[index].rho = (rho - (numrho - 1) * 0.5f) * rho_step;
    lines[index].theta = min_theta + angle * theta_step;
}

std::pair<int, line*> hough_parallel(unsigned char* d_img,
                                     int N,
                                     int threshold,
                                     float rho,
                                     float theta_step,
                                     double* duration,
                                     size_t sharedMemPerBlock,
                                     float min_theta = 0.0,
                                     float max_theta = CV_PI) {
    unsigned char* input;
    cudaMalloc(&input, N * N);
    cudaMemcpy(input, d_img, N * N, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    const float irho = 1.f / rho;
    const int max_rho = N + N;
    const int numangle = cvRound(CV_PI / theta_step);
    const int numrho = cvRound(((max_rho * 2 + 1)) / rho);

    unsigned int* d_packed_coords;
    int* d_count;

    cudaMalloc(&d_packed_coords, N * N * sizeof(unsigned int));
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (N + 15) / 16);
    extract_non_zero_packed<<<gridDim, blockDim>>>(input, d_packed_coords, d_count, N);

    int* accum;
    size_t accu_pitch;
    cudaMallocPitch(&accum, &accu_pitch, (numrho + 2) * sizeof(int), numangle + 2);

    cudaMemset2D(accum, accu_pitch, 0, (numrho + 2) * sizeof(int), numangle + 2);

    dim3 block(1024);
    dim3 grid(numangle - 2);
    size_t smemSize = (numrho - 1) * sizeof(int);

    if (smemSize < sharedMemPerBlock - 1000) 
        fill_accum_packed<<<grid, block, smemSize>>>(accum, accu_pitch, d_packed_coords, d_count, numrho, irho, min_theta, theta_step);
    else 
        fill_accum<<<grid, block>>>(accum, accu_pitch, d_packed_coords, d_count, numrho, irho, min_theta, theta_step);

    line* d_lines;
    int* d_counter;
    int counter = 0;

    cudaMalloc(&d_lines, 4096 * sizeof(line));
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));

    dim3 block_max(16, 16);
    dim3 grid_max((numrho + 15) / 16,
                  (numangle + 15) / 16);

    find_maxims<<<grid_max, block_max>>>(accum, accu_pitch, numangle, numrho, threshold, min_theta, theta_step, rho, d_lines, d_counter);

    cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    line* result = (line*)malloc(counter * sizeof(line));
    cudaMemcpy(result, d_lines, counter * sizeof(line), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    if (duration)
        *duration += milliseconds;

    cudaFree(input);
    cudaFree(accum);
    cudaFree(d_lines);
    cudaFree(d_counter);
    cudaFree(d_packed_coords);
    cudaFree(d_count);

    return {counter, result};
}

void drawHoughLines(
    const std::vector<std::pair<float, float>>& houghLines,
    cv::Mat& outputImage,
    double scale = 1000.0,
    cv::Scalar color = cv::Scalar(0, 0, 255)) {

    for (long unsigned int i = 0; i < houghLines.size(); ++i) {
        float rho = houghLines[i].first;
        float theta = houghLines[i].second;

        double a = std::cos(theta);
        double b = std::sin(theta);
        double x0 = a * rho;
        double y0 = b * rho;

        cv::Point pt1(cvRound(x0 + scale * -b),
                      cvRound(y0 + scale * a));
        cv::Point pt2(cvRound(x0 - scale * -b),
                      cvRound(y0 - scale * a));

        cv::line(outputImage,
                 pt1,
                 pt2,
                 color,
                 1,
                 cv::LINE_AA);
    }
}

int main(int argc, char** argv) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No CUDA device found" << std::endl;
        return -1;
    }

    int device = 0;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Cuda device: " << prop.name << std::endl;

    if (argc < 4) {
        std::cout << "not enough parametrs" << std::endl;
        return -1;
    }

    std::string filename = argv[1];
    int threshold = std::stoi(argv[2]);
    int n = std::stoi(argv[3]);

    std::string path = "../pictures/" + filename;

    double total_time_cpu = 0.0;
    double total_time_opencv_gpu = 0.0;
    double total_time_mine_gpu_basic = 0.0;

    cv::Mat edges =
        cv::imread(path, cv::IMREAD_GRAYSCALE);

    if (edges.empty()) {
        std::cerr << "Failed to load image: "
                  << path << std::endl;
        return -1;
    }

    cv::Mat img_dst = edges.clone();
    cv::cvtColor(edges, img_dst, cv::COLOR_GRAY2BGR);

    cv::Mat img_dst_mine = edges.clone();
    cv::cvtColor(edges, img_dst_mine, cv::COLOR_GRAY2BGR);

    auto start_cpu = omp_get_wtime();
    std::vector<std::pair<float, float>> lines = hough_transform_cpu(edges, threshold, 1, CV_PI / 180);
    auto stop_cpu = omp_get_wtime();

    double duration_cpu = stop_cpu - start_cpu;

    int biggest = edges.rows + edges.rows;

    for (int experiment = 0; experiment < n; ++experiment) {
        cv::cuda::GpuMat img_gpu, img_lines_gpu;
        cv::Mat img_lines_cpu;

        auto cv_hough = cv::cuda::createHoughLinesDetector(1, CV_PI / 180, threshold);

        auto start_gpu_full = omp_get_wtime();
        img_gpu.upload(edges);

        auto start_gpu = omp_get_wtime();
        cv_hough->detect(img_gpu, img_lines_gpu);
        img_lines_gpu.download(img_lines_cpu);
        cudaDeviceSynchronize();
        auto stop_gpu = omp_get_wtime();

        double duration_gpu = stop_gpu - start_gpu;
        double duration_gpu_full =
            stop_gpu - start_gpu_full;

        total_time_opencv_gpu += duration_gpu;

        if (experiment == 0) {
            for (int i = 0; i < img_lines_cpu.cols; ++i) {
                float rho =
                    img_lines_cpu.at<float>(0, i * 2);
                float theta =
                    img_lines_cpu.at<float>(0,
                                             i * 2 + 1);

                double a = std::cos(theta);
                double b = std::sin(theta);
                double x0 = a * rho;
                double y0 = b * rho;

                cv::Point pt1(
                    cvRound(x0 + biggest * (-b)),
                    cvRound(y0 + biggest * (a)));
                cv::Point pt2(
                    cvRound(x0 - biggest * (-b)),
                    cvRound(y0 - biggest * (a)));

                cv::line(img_dst,
                         pt1,
                         pt2,
                         cv::Scalar(255, 0, 0),
                         1,
                         cv::LINE_AA);
            }
        }

        unsigned char* d_img = edges.ptr();
        int N = edges.rows;

        std::pair<int, line*> result_basic = hough_parallel(d_img, N, threshold, 1, CV_PI / 180, &total_time_mine_gpu_basic, prop.sharedMemPerBlock);

        if (experiment == 0) {
            for (int i = 0; i < result_basic.first; ++i) {
                line line = result_basic.second[i];

                float theta = line.theta;
                float rho = line.rho;

                double a = std::cos(theta);
                double b = std::sin(theta);
                double x0 = a * rho;
                double y0 = b * rho;

                cv::Point pt1(
                    cvRound(x0 + biggest * (-b)),
                    cvRound(y0 + biggest * (a)));
                cv::Point pt2(
                    cvRound(x0 - biggest * (-b)),
                    cvRound(y0 - biggest * (a)));

                cv::line(img_dst_mine,
                         pt1,
                         pt2,
                         cv::Scalar(0, 0, 255),
                         1,
                         cv::LINE_AA);
            }
        }

        delete[] result_basic.second;

        std::string output_path_gpu = "../results/lines/gpu/";
        std::string output_filename_gpu = "lines_gpu_opencv.png";
        std::string output_filename_mine_basic = "lines_mine_basic.png";

        cv::imwrite(output_path_gpu + output_filename_gpu, img_dst);
        cv::imwrite(output_path_gpu + output_filename_mine_basic, img_dst_mine);
    }

    std::cout << "\n=== RESULTS OVER "
              << n << " EXPERIMENTS ===\n";
    std::cout << "CPU: "
              << duration_cpu * 1000.0
              << "ms, "
              << "lines found: "
              << lines.size() << "\n";

    std::cout << "GPU:"
              << (total_time_mine_gpu_basic / n) 
              << "ms\n";

    std::cout << "OpenCV: "
              << (total_time_opencv_gpu / n) * 1000.0
              << " ms\n";

    

    return 0;
}

