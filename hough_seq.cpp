#include <cmath>
#include <chrono>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "omp.h"

int compute_angle(double min_theta, double max_theta, double theta_step){
    int numangle = cvFloor((max_theta - min_theta) / theta_step) + 1;
    if ( numangle > 1 && fabs(CV_PI - (numangle-1)*theta_step) < theta_step/2 )
        --numangle;
    return numangle;
}

// precompute trigonometric values from 0 to angles_range
void precompute_tryg(double * cosarr, double * sinarr, int numangle, double min_theta, double theta_step, float irho){
    float ang = min_theta;
    for (int angle = 0; angle < numangle;ang+=(float)theta_step, angle++){
        cosarr[angle] = std::cos(ang) *irho;
        sinarr[angle] = std::sin(ang) *irho;
    }
}

// find local maximums in accuulator, write them to vector of lines if above threshold
void find_maxims(int * accu, std::vector<std::pair<float,float>> & lines, int numrho, int numangle, int threshold, 
                 double min_theta, float theta_step, float rho_step){
    for (int r = 0; r < numrho; ++r) {
        for (int theta = 0; theta < numangle; ++theta){
            int current_ind = (theta+1)*(numrho+2)+r+1;
            if( accu[current_ind] > threshold &&
                accu[current_ind] > accu[current_ind - 1] && accu[current_ind] >= accu[current_ind + 1] &&
                accu[current_ind] > accu[current_ind - numrho - 2 ] && accu[current_ind] >= accu[current_ind + numrho + 2] ){
                float rho = (r - (numrho-1)*0.5f) * rho_step;
                float angle = min_theta + theta * theta_step;
                lines.push_back(std::pair<float,float>(rho,angle));
            }
        }	
    }
}

std::vector<std::pair<float,float>> hough_transform(cv::Mat img, int threshold, float rho, float theta, double min_theta=0.0, double max_theta=CV_PI){
    float irho = 1./rho;
    
    unsigned char *data = img.ptr();
    int width = img.cols;
    int height = img.rows;

    int max_rho = width+height;
    int min_rho = -max_rho;

    int numangle = compute_angle(min_theta, max_theta, theta);
    int numrho = cvRound((max_rho - min_rho) * irho);

    int*accu = new int[(numangle+2)*(numrho+2)]{0};

    double *cosvalues = new double[numangle];
    double *sinvalues = new double[numangle];

    precompute_tryg(cosvalues,sinvalues,numangle, min_theta, theta, irho);

    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            if (data[(y*width)+x] != 0){
                for (int angle = 0; angle < numangle; angle++){
                    int r =  round(x*cosvalues[angle]+y*sinvalues[angle]);
                    r += (numrho-1)/2;
                    accu[(angle+1)*(numrho+2)+r+1]++;
                }
            }
        }
    }

    std::vector<std::pair<float,float>> lines;
    find_maxims(accu,lines,numrho,numangle,threshold,min_theta,theta,rho);

    delete [] cosvalues;
    delete [] sinvalues;
    delete [] accu;

    return lines;   
}

int main(){
    cv::Mat img_edge;
    cv::Mat img_dst;
    cv::Mat img_blur;
    std::string filename;
    std::cout << "Enter the name of the file: ";
    std::cin >> filename;
    std::string path = "../pictures/" + filename;
    cv::Mat img = cv::imread(path, 1);
    if (!img.data){
        printf("No image data \n");
        return -1;
    }

    int biggest = std::max(img.rows, img.cols);

    img_dst = img.clone();
    cv::blur(img, img_blur,cv::Size(5,5));
	cv::Canny(img_blur, img_edge, 100, 200, 3);
    cv::imwrite("../results/edges.jpg", img_edge);

    cv::Mat img_edge_cv = img_edge.clone();
    auto start = omp_get_wtime();
    auto lines = hough_transform(img_edge,50, 1,CV_PI/180);
    auto stop = omp_get_wtime();
    auto duration = stop - start;
    
    printf("mine:\n");
    printf("1. lines:  %ld \n", lines.size());
    printf("2. time:  %lf \n", duration);
    
    std::vector<cv::Vec2f> lines_cv;
    
    start = omp_get_wtime();
    cv::HoughLines(img_edge_cv, lines_cv, 50, CV_PI/180, 1, 0, 0);
    stop = omp_get_wtime();
    auto duration_cv = stop-start;

    for (const auto& line : lines) {
        double theta = line.second;
        double rho = line.first;
        double a = std::cos(theta);
        double b = std::sin(theta);
        double x0 = a * rho;
        double y0 = b * rho;
        cv::Point pt1(cvRound(x0 + biggest * (-b)), cvRound(y0 + biggest * (a)));
        cv::Point pt2(cvRound(x0 - biggest * (-b)), cvRound(y0 - biggest * (a)));
        cv::line(img_dst, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    } 
    
    printf("cv:\n");
    printf("1. lines:  %ld \n", lines_cv.size());
    printf("2. time:  %lf \n", duration_cv);

    cv::Mat img_dst2 = img.clone();
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines_cv[i][0], theta = lines_cv[i][1];
        cv::Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + biggest*(-b));
        pt1.y = cvRound(y0 + biggest*(a));
        pt2.x = cvRound(x0 - biggest*(-b));
        pt2.y = cvRound(y0 - biggest*(a));
        line( img_dst2, pt1, pt2, cv::Scalar(0,0,255), 2, cv::LINE_AA);
    }

    cv::imwrite("../results/result_mine.png",img_dst);
    cv::imwrite("../results/result_cv.png",img_dst2);
    return 0;
}