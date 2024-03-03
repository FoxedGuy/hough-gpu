#include <cmath>
#include <chrono>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// precompute trigonometric values from 0 to angles_range
void precompute_tryg(double * cosarr, double * sinarr, int angles_range){
    for (int angle = 0; angle < angles_range; angle++){
        cosarr[angle] = std::cos(angle * 0.017453293f);
        sinarr[angle] = std::sin(angle * 0.017453293f);
    }
}

// find local maximums in accuulator, write them to vector of lines if above threshold
void find_maxims(int * accu, std::vector<std::pair<int,int>> & lines, int rho_range, int theta_range, int threshold, int offset){
    for (int rho = 0; rho < rho_range; ++rho) {
        int r = rho-offset;
        for (int theta = 0; theta < theta_range; ++theta){
            if (accu[rho*theta_range + theta] > threshold){
                int current_ind = rho * (theta_range) + theta;
                if( accu[current_ind] > threshold &&
                    accu[current_ind] > accu[current_ind - 1] && accu[current_ind] >= accu[current_ind + 1] &&
                    accu[current_ind] > accu[current_ind - theta_range ] && accu[current_ind] >= accu[current_ind + theta_range])
                    lines.push_back(std::make_pair(r,theta));
            }
        }
    }	
}


std::vector<std::pair<int,int>> hough_transform(unsigned char* edges, int w, int h, int threshold){
    double diagonal = sqrt(w*w+h*h);
    int accu_h = diagonal*2; // (-r, r)
    int accu_w = 180; 

    double *cosvalues = new double[180];
    double *sinvalues = new double[180];
    int * accu = new int[accu_h*accu_w]();

    precompute_tryg(cosvalues,sinvalues,180);

    for (int y = 0; y < h; y++){
        for (int x = 0; x < w; x++){
            if (edges[(y*w)+x] != 0){
                for (int theta = 0; theta < 180; theta++){
                    int r =  (x*cosvalues[theta]+y*sinvalues[theta]) + diagonal;
                    accu[r*accu_w+theta]++;
                }
            }
        }
    }

    std::vector<std::pair<int,int>> lines;
    find_maxims(accu,lines,accu_h,accu_w,threshold,diagonal);

    delete [] cosvalues;
    delete [] sinvalues;
    delete [] accu;

    return lines;   
}

int main(){
    cv::Mat img_edge;
    cv::Mat img_dst;
    cv::Mat img_blur;
    cv::Mat img = cv::imread("../pictures/house2.jpg", 1);
    if (!img.data){
        printf("No image data \n");
        return -1;
    }
    img_dst = img.clone();
    cv::blur(img, img_blur,cv::Size(5,5));
	cv::Canny(img_blur, img_edge, 100, 150, 3);
    cv::imwrite("edges.jpg", img_edge);

    auto start = std::chrono::high_resolution_clock::now();
    auto lines = hough_transform(img_edge.data, img_edge.cols, img_edge.rows, 350);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    printf("mine:\n");
    printf("1. lines:  %d \n", lines.size());
    printf("2. time:  %d \n", duration.count());
    
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

    std::vector<cv::Vec2f> lines_cv;
    
    start = std::chrono::high_resolution_clock::now();
    cv::HoughLines(img_edge, lines_cv, 1, CV_PI/180, 350, 0, 0);
    stop = std::chrono::high_resolution_clock::now();
    auto duration_cv = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    printf("cv:\n");
    printf("1. lines:  %d \n", lines_cv.size());
    printf("2. time:  %d \n", duration_cv.count());

    cv::Mat img_dst2 = img.clone();
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines_cv[i][0], theta = lines_cv[i][1];
        cv::Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line( img_dst2, pt1, pt2, cv::Scalar(0,0,255), 2, cv::LINE_AA);
    }

    cv::imwrite("../results/result_mine.png",img_dst);
    cv::imwrite("../results/result_cv.png",img_dst2);
    return 0;
}