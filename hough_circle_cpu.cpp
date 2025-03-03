#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<omp.h>

void calculate_trig(float *cos, float *sin){
    for (int i  = 0; i < 360; i++){
        cos[i] = std::cos(i);
        sin[i] = std::sin(i);
    }
}

std::vector<std::tuple<int,int,int>> hough_circles(unsigned char* data, int height, int width, int radius_min, int radius_max, unsigned int threshold){
    
    // accum 
    unsigned int *acc = new unsigned int[height*width*(radius_max-radius_min+1)];
    memset(acc, 0, height*width*(radius_max-radius_min+1)*sizeof(unsigned int));

    float *sin_tab = new float[360];
    float *cos_tab = new float[360];

    calculate_trig(cos_tab, sin_tab);

    // fill accum
    for(int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            if (data[y*width + x] > 0){
                // draw circles in accum
                for (int r = radius_min; r <= radius_max; r++){
                    for (int i = 0; i < 360; i++){
                        int a = (float)x - r * cos_tab[i];
                        int b = (float)y - r * sin_tab[i];
                        if (a >= 0 && a < width && b >= 0 && b < height){
                            acc[(r - radius_min)*height*width + b*width + a]++;
                        }
                    }
                }
            }
        }
    }

    std::vector<std::tuple<int,int,int>> centers;
    int circles = 0;
    for (int r = radius_min; r <= radius_max; r++){
        for (int y = 0; y < height; y++){
            for (int x = 0; x < width; x++){
                // non-maximum suppression
                if (acc[(r - radius_min)*height*width + y*width + x] > threshold){
                    bool is_local_max = true;
                    for (int dy = -1; dy <= 1; dy++){
                        for (int dx = -1; dx <= 1; dx++){
                            if (dy == 0 && dx == 0) continue;
                            int ny = y + dy;
                            int nx = x + dx;
                            if (ny >= 0 && ny < height && nx >= 0 && nx < width){
                                if (acc[(r - radius_min)*height*width + ny*width + nx] > acc[(r - radius_min)*height*width + y*width + x]){
                                    is_local_max = false;
                                    break;
                                }
                            }
                        }
                        if (!is_local_max) break;
                    }
                    if (is_local_max){
                        centers.push_back(std::make_tuple(x,y,r));
                        circles++;
                    }
                }
            }
        }
    }

    std::cout << circles << " circles found" << std::endl;
    delete[] acc;
    delete[] sin_tab;
    delete[] cos_tab;
    return centers;
}

int main(int argc, char** argv){
    
    std::string img_name = argv[1];
    int radius_min = std::stoi(argv[2]);
    int radius_max = std::stoi(argv[3]);
    
    std::string path = "pictures_circles/";
    std::string img_path = path + img_name;
    cv::Mat img = cv::imread(img_path, 1);
    
    if (!img.data){
        printf("No image data \n");
        return -1;
    }

    cv::Mat img_result = img.clone();
    cv::Mat blur;
    cv::Mat edges;
    cv::GaussianBlur(img, blur, cv::Size(9,9), 2, 2);
    cv::Canny(blur, edges, 100, 200);

    auto start = omp_get_wtime();
    std::vector<std::tuple<int,int,int>> centers = hough_circles(edges.data, edges.rows, edges.cols, radius_min, radius_max,200);
    auto end = omp_get_wtime();
    std::cout << "Time: " << end - start << std::endl;
    
    for (auto center: centers){
        cv::circle(img_result, cv::Point(std::get<0>(center), std::get<1>(center)), std::get<2>(center), cv::Scalar(255, 0, 255), 2);
    }

    cv::Mat img_concat = cv::Mat::zeros(img.rows, img.cols + edges.cols, CV_8UC3);
    cv::cvtColor(edges, edges, cv::COLOR_GRAY2BGR);
    cv::hconcat(img, edges, img_concat);
    cv::hconcat(img_concat, img_result, img_concat);
    cv::imshow("Original image, edges and results", img_concat);
    cv::imwrite("results/lines/cpu/result.jpg", img_concat);
    cv::waitKey(0);


    return 0;
}