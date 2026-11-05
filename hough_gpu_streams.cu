#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/cudaimgproc.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include "omp.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        std::abort(); \
    } \
} while (0)

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
    __sincosf(theta, &sinv, &cosv);
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

        if (r >= 0 && r < numrho) {
            atomicAdd(&row[r + 1], 1);
        }    

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

    for (int r = tid; r < numrho + 2; r += blockDim.x) {
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

        if (r >= 0 && r < numrho) {
            atomicAdd(&sh_accum[r + 1], 1);
        }
    }

    __syncthreads();

    char* base = (char*)accum;
    int* row = (int*)(base + (angle + 1) * step);

    for (int i = tid; i < numrho + 2; i += blockDim.x)
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

class HoughTransformGPU {
public:
    HoughTransformGPU(
        const cv::Mat& image,
        int threshold,
        float rho,
        float theta_step,
        size_t sharedMemPerBlock,
        float min_theta = 0.0f,
        float max_theta = CV_PI
    );

    ~HoughTransformGPU();

    HoughTransformGPU(const HoughTransformGPU&) = delete;
    HoughTransformGPU& operator=(const HoughTransformGPU&) = delete;

    void execute();

    std::vector<line> getResults();

private:
    void allocateMemory();
    void uploadImage(const cv::Mat& image);
    void reset();

    int N;
    int threshold;
    int numrho;
    int numangle;

    float rho;
    float irho;
    float theta_step;
    float min_theta;

    size_t sharedMemPerBlock;
    size_t accu_pitch = 0;

    unsigned char* d_input = nullptr;
    unsigned int* d_packed_coords = nullptr;
    int* d_count = nullptr;

    int* d_accum = nullptr;

    line* d_lines = nullptr;
    int* d_counter = nullptr;
};

HoughTransformGPU::HoughTransformGPU(
    const cv::Mat& image,
    int threshold,
    float rho,
    float theta_step,
    size_t sharedMemPerBlock,
    float min_theta,
    float max_theta
)
    : N(image.rows),
      threshold(threshold),
      rho(rho),
      irho(1.0f / rho),
      theta_step(theta_step),
      min_theta(min_theta),
      sharedMemPerBlock(sharedMemPerBlock)
{
    if (image.empty() ||
        image.rows != image.cols ||
        image.type() != CV_8UC1 ||
        !image.isContinuous()) {
        throw std::invalid_argument("Invalid input image");
    }

    if (N > 65536 || rho <= 0.0f ||
        theta_step <= 0.0f || max_theta <= min_theta) {
        throw std::invalid_argument("Invalid Hough parameters");
    }

    numangle = computeNumangle(
        min_theta, max_theta, theta_step
    );

    const int max_rho = N + N;
    numrho = cvRound((max_rho * 2 + 1) * irho);
    
    allocateMemory();
    uploadImage(image);
}

void HoughTransformGPU::allocateMemory() {
    const size_t image_size =
        static_cast<size_t>(N) * N;

    const size_t max_lines =
        static_cast<size_t>(numangle) * numrho;

    CUDA_CHECK(cudaMalloc(&d_input, image_size));

    CUDA_CHECK(cudaMalloc(
        &d_packed_coords,
        image_size * sizeof(unsigned int)
    ));

    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));

    CUDA_CHECK(cudaMallocPitch(
        &d_accum,
        &accu_pitch,
        (numrho + 2) * sizeof(int),
        numangle + 2
    ));

    CUDA_CHECK(cudaMalloc(
        &d_lines,
        max_lines * sizeof(line)
    ));

    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
}

void HoughTransformGPU::uploadImage(const cv::Mat& image) {
    const size_t image_size =
        static_cast<size_t>(N) * N;

    CUDA_CHECK(cudaMemcpy(
        d_input,
        image.ptr<unsigned char>(),
        image_size,
        cudaMemcpyHostToDevice
    ));
}

void HoughTransformGPU::execute() {
    reset();

    dim3 block_extract(16, 16);
    dim3 grid_extract(
        (N + 15) / 16,
        (N + 15) / 16
    );

    extract_non_zero_packed<<<grid_extract, block_extract>>>(
        d_input,
        d_packed_coords,
        d_count,
        N
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 block_vote(1024);
    dim3 grid_vote(numangle);

    size_t smemSize = (numrho + 2) * sizeof(int);

    if (sharedMemPerBlock > 1000 &&
        smemSize < sharedMemPerBlock - 1000)
    {
        fill_accum_packed<<<grid_vote, block_vote, smemSize>>>(
            d_accum,
            accu_pitch,
            d_packed_coords,
            d_count,
            numrho,
            irho,
            min_theta,
            theta_step
        );
    }
    else
    {
        fill_accum<<<grid_vote, block_vote>>>(
            d_accum,
            accu_pitch,
            d_packed_coords,
            d_count,
            numrho,
            irho,
            min_theta,
            theta_step
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 block_max(16, 16);
    dim3 grid_max(
        (numrho + 15) / 16,
        (numangle + 15) / 16
    );

    find_maxims<<<grid_max, block_max>>>(
        d_accum,
        accu_pitch,
        numangle,
        numrho,
        threshold,
        min_theta,
        theta_step,
        rho,
        d_lines,
        d_counter
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void HoughTransformGPU::reset() {
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    CUDA_CHECK(cudaMemset2D(
        d_accum,
        accu_pitch,
        0,
        (numrho + 2) * sizeof(int),
        numangle + 2
    ));
}

std::vector<line> HoughTransformGPU::getResults() {
    int count = 0;

    CUDA_CHECK(cudaMemcpy(
        &count,
        d_counter,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    const size_t max_lines =
        static_cast<size_t>(numangle) * numrho;

    if (count < 0 || static_cast<size_t>(count) > max_lines) {
        throw std::runtime_error("Invalid number of detected lines");
    }

    std::vector<line> results(count);

    if (count > 0) {
        CUDA_CHECK(cudaMemcpy(
            results.data(),
            d_lines,
            static_cast<size_t>(count) * sizeof(line),
            cudaMemcpyDeviceToHost
        ));
    }

    return results;
}

HoughTransformGPU::~HoughTransformGPU() {
    cudaFree(d_input);
    cudaFree(d_packed_coords);
    cudaFree(d_count);
    cudaFree(d_accum);
    cudaFree(d_lines);
    cudaFree(d_counter);
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

    if (argc < 5) {
        std::cerr << "Usage: ./hough <image> <threshold> <iterations> <opencv|custom>\n";
        return 1;
    }

    

    std::string filename = argv[1];
    int threshold = std::stoi(argv[2]);
    int n = std::stoi(argv[3]);
    std::string implementation = argv[4];

    if (implementation != "opencv" && implementation != "custom") {
        std::cerr << "Invalid implementation\n";
        return 1;
    }

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

    if (edges.rows != edges.cols) {
    throw std::runtime_error("Image must be square");
    }

    if (edges.type() != CV_8UC1) {
        throw std::runtime_error("Image must be 8-bit grayscale");
    }

    if (!edges.isContinuous()) {
        edges = edges.clone();
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
    
    auto cv_hough = cv::cuda::createHoughLinesDetector(1, CV_PI / 180, threshold);
    
    HoughTransformGPU hough(
        edges,
        threshold,
        1.0f,
        CV_PI / 180,
        prop.sharedMemPerBlock
    ); 
    cv::cuda::GpuMat img_gpu, img_lines_gpu;
    cv::Mat img_lines_cpu;
    img_gpu.upload(edges);

    for (int i = 0; i < 5; ++i) {
        if (implementation == "opencv") {
            cv_hough->detect(img_gpu, img_lines_gpu);
        } else {
            hough.execute();
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < n; ++i) {
        if (implementation == "opencv") {
            cv_hough->detect(img_gpu, img_lines_gpu);
        } else {
            hough.execute();
        }
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_ms, start, stop
    ));

    std::cout << "Implementation: " << implementation << "\n";
    std::cout << "Total time: " << total_ms << " ms\n";
    std::cout << "Average time: " << total_ms / n << " ms\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    img_lines_gpu.download(img_lines_cpu);
    std::vector<line> result_basic = hough.getResults();

    return 0;
}

