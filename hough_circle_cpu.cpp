#include<iostream>
#include<opencv2/opencv.hpp>

std::vector<std::tuple<int,int,int>> hough_circles(unsigned char* data, int height, int width, int radius_min, int radius_max, unsigned int threshold){
    
    // accum 
    unsigned int *acc = new unsigned int[height*width*(radius_max-radius_min+1)];
    memset(acc, 0, height*width*(radius_max-radius_min+1)*sizeof(unsigned int));

    // fill accum
    for(int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            if (data[y*width + x] > 0){
                // draw circles in accum
                for (int r = radius_min; r <= radius_max; r++){
                    for (int i = 0; i < 360; i++){
                        int a = (float)x - r * cos(i);
                        int b = (float)y - r * sin(i);
                        if (a >= 0 && a < width && b >= 0 && b < height){
                            acc[(r - radius_min)*height*width + b*width + a]++;
                        }
                    }
                }
            }
        }
    }

    std::vector<std::tuple<int,int,int>> centers;
    for (int r = radius_min; r <= radius_max; r++){
        for (int y = 0; y < height; y++){
            for (int x = 0; x < width; x++){
                if (acc[(r - radius_min)*height*width + y*width + x] > threshold){
                    centers.push_back(std::make_tuple(x,y,r));
                }
            }
        }
    }

    return centers;
}

int main(int argc, char** argv){
    std::string path = "../pictures_circles/";
    
    std::string img_name = argv[1];
    int radius_min = std::stoi(argv[2]);
    int radius_max = std::stoi(argv[3]);
    std::string img_path = path + img_name;
    cv::Mat img = cv::imread(img_path, 1);
    
    if (!img.data){
        printf("No image data \n");
        return -1;
    }

    cv::Mat blur;
    cv::Mat edges;
    cv::GaussianBlur(img, blur, cv::Size(9,9), 2, 2);
    cv::Canny(blur, edges, 100, 200);

    std::vector<std::tuple<int,int,int>> centers = hough_circles(edges.data, edges.rows, edges.cols, radius_min, radius_max,200);

    for (auto center: centers){
        cv::circle(img, cv::Point(std::get<0>(center), std::get<1>(center)), std::get<2>(center), cv::Scalar(255, 0, 255), 2);
    }

    cv::imshow("Hough Circle", img);
    cv::waitKey(0);

    return 0;
}