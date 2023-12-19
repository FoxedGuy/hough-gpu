#include "opencv2/opencv.hpp"
#include <vector>
#include <cmath>

unsigned int* hough_transform(unsigned char* edges, int h, int w, int threshold){
    //getting the width and height of accumulator;
    double hough_h = std::sqrt(std::pow(h,2)+std::pow(w,2));  
    int accu_h = hough_h * 2.0; // -r -> +r 
    int accu_width = 180; 
    unsigned int* accu = (unsigned int*)malloc(sizeof(unsigned int)*(accu_h*accu_width));

    int maxa = 0;
    for (int x = 0; x < w; x++){
        for (int y = 0; y < h; y++){
            if (edges[(y*w)+x] > 250){
                for (int alpha = 0; alpha < 180; alpha++){
                    double ang = (double)alpha * CV_PI/180;
                    double r = x*cos((double)ang)+y*sin((double)ang);
                    
                    accu[(int)(r*180+ang)]++;
                    if (maxa < accu[(int)((int)(r+accu_h)*180+alpha)]){
                        maxa = accu[(int)((int)(r+accu_h)*180+alpha)];
                    }
                }
            }
        }
    }

    cv::Mat vis (accu_h,accu_width, CV_8UC3);
    double coef = 255.0 / (double)maxa;
    for (int p = 0; p < (accu_h*accu_width); p++){
        unsigned char c = (double)accu[p] * coef < 255.0 ? (double)accu[p] * coef : 255.0;
        vis.data[(p*3)+0] = 255;
        vis.data[(p*3)+1] = 255-c;
        vis.data[(p*3)+2] = 255-c;
    }

    cv::imwrite("../accu.png",vis);
    std::vector<std::pair<int,int>> lines;
    for (int rho = 0; rho < accu_h; ++rho) {
        for (int theta = 0; theta < accu_width ; ++theta) {
	    if (accu[(int)rho*180+theta] > threshold) {
                lines.push_back({rho,theta});
            }
        }
    }	  
    return accu;   
}

int main(){
    cv::Mat src,edges,dst;
    src = cv::imread("../pictures/sudoku.png");
    dst = src.clone();
    
    cv::Canny(src,edges,50,200,3);
    cv::imwrite("../edges.png",edges);
    
    unsigned int* full_accu = hough_transform(edges.data,edges.rows,edges.cols,250);

    free(full_accu);
    
    return 0;
}
