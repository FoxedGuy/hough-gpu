#include<cmath>
#include "opencv2/opencv.hpp"
#include "opencv2/opencv.hpp"
#define DEG2RAD 0.017453293f


std::vector<std::pair<int,int>> hough_transform(unsigned char* edges, int w, int h, int threshold){
    //getting the width and height of accumulator;
    double hough_h = sqrt(w*w+h*h);
    int accu_h = hough_h*2; // (-r, r)
    int accu_w = 180; 
    std::vector<std::vector<int>>accu(accu_h, std::vector<int>(accu_w,0)); 

    double center_x = w/2;
    int maxa = -1;
    double center_y = h/2;
    for (double x = 0; x < w; x++){
        for (double y = 0; y < h; y++){
            if (edges[((int)y*(int)w)+(int)x] > 250){
                for (int alpha = 0; alpha < 180; alpha++){
                    double ang = (double)alpha * 0.017453293f;
                    double r = x*cos(ang)+y*sin(ang);
                    int ind =round(r + hough_h); // compute index of rho in accu
                    accu[ind][alpha]++;
                    if(accu[ind][alpha] > maxa)  
                        maxa = accu[ind][alpha];  
                }
            }
        }
    }
    cv::Mat img_accu(accu_h, accu_w, CV_8UC3);  
    double coef = 255.0/maxa;
    for (int y = 0; y < accu_h; y++){
        for (int x = 0; x < accu_w; x++){  
            int p = y*180+x;
            unsigned char c = (double)accu[y][x] * coef < 255.0 ? (double)accu[y][x] * coef : 255.0;  
            img_accu.data[(p*3)+0] = 255;  
            img_accu.data[(p*3)+1] = 255-c;  
            img_accu.data[(p*3)+2] = 255-c;  
        }  
    }

    cv::imwrite("../accu.png", img_accu);
    
    // TODO: same line is detected multiple times, resulting in bad computing times
    // make this loop check surroundings of index, before adding line to vector 
    std::vector<std::pair<int,int>> lines;
    for (int rho = 0; rho < accu_h; ++rho) {
        for (int theta = 0; theta < accu_w ; ++theta){
            if ((int)(accu[rho][theta]) > threshold){
                lines.push_back({rho-hough_h,theta}); // don't forget to calculate real rho!
            }
        }
    }	
    return lines;   
}

int main(){
    cv::Mat img_edge;
    cv::Mat img_dst;
    cv::Mat img_blur;
    cv::Mat img = cv::imread("../pictures/fence.png", 1);
    if (!img.data){
        printf("No image data \n");
        return -1;
    }
    img_dst = img.clone();
    cv::blur(img, img_blur,cv::Size(5,5));
	cv::Canny(img_blur, img_edge, 100, 150, 3);
    cv::imwrite("edges.jpg", img_edge);

    auto lines = hough_transform(img_edge.data, img_edge.cols, img_edge.rows, 190);
    printf("Lines: %d", lines.size());
    for (const auto& line : lines) {
        double theta = line.second * CV_PI / 180.0;
        double rho = line.first;
        double a = std::cos(theta);
        double b = std::sin(theta);
        double x0 = a * rho;
        double y0 = b * rho;
        cv::Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
        cv::Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
        cv::line(img_dst, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    } 

    cv::imwrite("result.png",img_dst);
    return 0;
}