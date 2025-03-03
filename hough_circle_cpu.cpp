#include<iostream>
#include<opencv2/opencv.hpp>

std::vector<std::pair<int,int>> hough_circles(unsigned char* data, int height, int width, int radius, unsigned int threshold){
    
    // accum 
    unsigned int *acc = new unsigned int[height*width];
    memset(acc, 0, height*width*sizeof(unsigned int));

    // fill accum
    for(int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            if (data[y*width + x] > 0){
                // draw circle in accum
                for (int i = 0; i < 360; i++){
                    int a = (float)x - radius * cos(i);
                    int b = (float)y - radius * sin(i);
                    if (a >= 0 && a < width && b >= 0 && b < height){
                        acc[b*width + a]++;
                    }
                }
            }
        }
    }

    std::vector<std::pair<int,int>> centers;
    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            if (acc[y*width + x] > threshold){
                centers.push_back(std::pair<int,int>(x,y));
            }
        }
    }

    return centers;
}

int main(int argc, char** argv){
    std::string path = "../pictures_circles/";
    
    std::string img_name = argv[1];
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

    std::vector<std::pair<int,int>> centers = hough_circles(edges.data, edges.rows, edges.cols, 100,200);

    for (auto center: centers){
        cv::circle(img, cv::Point(center.first, center.second), 100, cv::Scalar(0, 0, 255), 2);
    }

    cv::imshow("Hough Circle", img);
    cv::waitKey(0);

    return 0;
}