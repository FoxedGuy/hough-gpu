#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<omp.h>

std::vector<cv::Vec3i> hough_circles(cv::Mat img, int height, int width, int radius_min, int radius_max, unsigned int threshold){    
    
    const int numradii = radius_max - radius_min + 1;
    const int accu_size = height * width * numradii;
    unsigned int *accu = new unsigned int[accu_size];
    memset(accu, 0, accu_size*sizeof(unsigned int));

    float *sinvalues = new float[360];
    float *cosvalues = new float[360];

    for (int i  = 0; i < 360; i++){
        sinvalues[i] = sin(i);
        cosvalues[i] = cos(i);
    }

    std::vector<cv::Point> non_zero;
    cv::findNonZero(img, non_zero);

    for (const auto& pt : non_zero){
        for (int r = radius_min; r <= radius_max; r++){
            int radius_idx = r - radius_min;
            int base_idx = radius_idx * height * width;
            for (int i = 0; i < 360; i++){
                int a = (pt.x - r * cosvalues[i]);
                int b = (pt.y - r * sinvalues[i]);
                if (a >= 0 && a < width && b >= 0 && b < height){
                    accu[base_idx + b * width + a]++;
                }
            }
        }
    }

    const int offsets[8][2] = {
        {-1, -1}, {-1, 0}, {-1, 1},
        { 0, -1},          { 0, 1},
        { 1, -1}, { 1, 0}, { 1, 1}
    };

    std::vector<cv::Vec3i> centers;
    for (int r = radius_min; r <= radius_max; r++){
        int radius_idx = r - radius_min;
        for (int y = 0; y < height; y++){
            for (int x = 0; x < width; x++){
                int idx = radius_idx * height * width + y * width + x;
                int center_val = accu[idx];
                
                if (center_val <= threshold) continue;

                bool is_local_max = true;
                 for (int k = 0; k < 8; ++k) {
                    int nx = x + offsets[k][0];
                    int ny = y + offsets[k][1];
                    int neighbor_idx = radius_idx * height * width + ny * width + nx;
                    if (accu[neighbor_idx] > center_val) {
                        is_local_max = false;
                        break;
                    }
                
                }
                if (is_local_max){
                    centers.emplace_back(x,y,r);
                }
            }
        }
    }

    std::cout << centers.size() << " circles found" << std::endl;
    delete[] accu;
    delete[] sinvalues;
    delete[] cosvalues;
    return centers;
}

int main(int argc, char** argv){
    
    std::string img_name = argv[1];
    int radius_min = std::stoi(argv[2]);
    int radius_max = std::stoi(argv[3]);
    
    std::string path = "../pictures_circles/";
    std::string img_path = path + img_name;
    cv::Mat img = cv::imread(img_path, 1);
    
    if (!img.data){
        printf("No image data \n");
        return -1;
    }

    cv::Mat img_result = img.clone();
    cv::Mat blur;
    cv::Mat edges;
    cv::GaussianBlur(img, blur, cv::Size(5,5), 2, 2);
    cv::Canny(blur, edges, 100, 200);

    auto start = omp_get_wtime();
    std::vector<cv::Vec3i> centers = hough_circles(edges, edges.rows, edges.cols, radius_min, radius_max,200);
    auto end = omp_get_wtime();
    std::cout << "Time: " << end - start << std::endl;
    
    for (auto center: centers){
        cv::circle(img_result, cv::Point(center[0], center[1]), center[2], cv::Scalar(255, 0, 255), 2);
    }

    cv::Mat img_concat = cv::Mat::zeros(img.rows, img.cols + edges.cols, CV_8UC3);
    cv::cvtColor(edges, edges, cv::COLOR_GRAY2BGR);
    cv::hconcat(img, edges, img_concat);
    cv::hconcat(img_concat, img_result, img_concat);
    cv::imshow("Original image, edges and results", img_concat);
    cv::imwrite("../results/lines/cpu/result.jpg", img_concat);
    cv::waitKey(0);


    return 0;
}