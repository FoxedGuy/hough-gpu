#include <cmath>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "omp.h"

void saveAccumulatorImage(int* accu, int numangle, int numrho, const std::string& filename) {
    cv::Mat accuImage(numangle, numrho, CV_32SC1);
    for (int angle = 0; angle < numangle; ++angle) {
        for (int rho = 0; rho < numrho; ++rho) {
            int value = accu[(angle + 1) * (numrho + 2) + (rho + 1)];
            accuImage.at<int>(angle, rho) = value;
        }
    }

    cv::Mat normalized;
    cv::normalize(accuImage, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::imwrite(filename, normalized);

    int scaleFactor = 7; 
    cv::Mat resized;
    cv::resize(normalized, resized, cv::Size(normalized.cols / scaleFactor, normalized.rows), cv::INTER_AREA);

    cv::imwrite("../results/accumulator_squeezed.png", resized);
}

std::vector<std::pair<float,float>> hough_transform(cv::Mat img, int threshold, float rho, float theta, float min_theta=0.0, float max_theta=CV_PI){
    
    const float irho = 1./rho;
    
    const int width = img.cols;
    const int height = img.rows;
    const int max_rho = width+height;

    const int numangle = cvFloor((max_theta - min_theta) / theta) + 1;
    const int numrho = cvRound((max_rho*2+1)*irho);
    const int accu_size = (numangle+2)*(numrho+2);

    int *accu = new int[accu_size]{0};
    float *cosvalues = new float[numangle];
    float *sinvalues = new float[numangle];

    for (int angle = 0; angle < numangle; angle++){
        float ang = min_theta + angle * theta;
        cosvalues[angle] = std::cos(ang) * irho;
        sinvalues[angle] = std::sin(ang) * irho;
    }

    std::vector<cv::Point> non_zero;
    cv::findNonZero(img, non_zero);

    for (const auto& pt : non_zero){
        for (int angle = 0; angle < numangle; angle++){
            int r = cvRound(pt.x * cosvalues[angle] + pt.y * sinvalues[angle]);
            r += (numrho - 1) / 2;
            accu[(angle + 1) * (numrho + 2) + (r + 1)]++;
        }
    }

    std::vector<std::pair<float,float>> lines;
    for (int r = 0; r < numrho; ++r) {
        for (int angle = 0; angle < numangle; ++angle){
            int ind = (angle+1)*(numrho+2)+r+1;
            if( accu[ind] > threshold &&
                accu[ind] > accu[ind - 1] && accu[ind] >= accu[ind + 1] &&
                accu[ind] > accu[ind - numrho - 2 ] && accu[ind] >= accu[ind + numrho + 2] ){
                float rho_val = (r - (numrho-1)*0.5f) * rho;
                float angle_val = min_theta + angle * theta;
                lines.push_back(std::pair<float,float>(rho_val,angle_val));
            }
        }	
    }

    delete [] cosvalues;
    delete [] sinvalues;
    delete [] accu;

    return lines;   
}

int main(int argc, char** argv) {
    cv::Mat img_edge;
    cv::Mat img_dst;
    cv::Mat img_blur;
    cv::Mat img;
    
    int threshold;

    if (argc == 3){
        std::string filename = argv[1];
        std::string path = "../pictures/" + filename;

        img = cv::imread(path, 1);
        if (img.empty()) {
            std::cerr << "Failed to load image: " << path << std::endl;
            return -1;
        }
        threshold = std::stoi(argv[2]);
    }else{
        std::cout << "Usage: " << argv[0] << " <image_filename> <threshold>\n";
        return -1;
    }

    int biggest = std::max(img.rows, img.cols);

    img_dst = img.clone();
    cv::blur(img, img_blur,cv::Size(5,5));
	cv::Canny(img_blur, img_edge, 50, 150, 3);
    cv::imwrite("../results/edges.jpg", img_edge);

    cv::Mat img_edge_cv = img_edge.clone();
    auto start = omp_get_wtime();
    auto lines = hough_transform(img_edge,threshold, 1,CV_PI/180);
    auto stop = omp_get_wtime();
    auto duration = stop - start;
    
    printf("mine:\n");
    printf("1. lines:  %ld \n", lines.size());
    printf("2. time:  %lf \n", duration);
    
    std::vector<cv::Vec2f> lines_cv;
    
    start = omp_get_wtime();
    cv::HoughLines(img_edge_cv, lines_cv, 1, CV_PI/180, threshold);
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