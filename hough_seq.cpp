#include<cmath>
#include "opencv2/opencv.hpp"
#include "opencv2/opencv.hpp"
#define DEG2RAD 0.017453293f

class Hough{
    
    unsigned int* accu;
    int accu_h, accu_w;
    int img_h, img_w;

public:

    Hough(){
        accu = 0;
    }

    ~Hough(){
        if (accu){
            free(accu);
        }
    }

    int hough_transform(unsigned char* img, int w, int h){
        img_h = h; img_w = w;
        double hough_h = ((sqrt(2.0) * (double)(h>w?h:w)) / 2.0);  
        accu_h = hough_h * 2.0; // -r -> +r 
        accu_w = 180;

        accu = (unsigned int*)calloc(accu_h*accu_w,sizeof(unsigned int)); 
        
        double center_x = w/2;
        double center_y = h/2;

        for(int y = 0; y < h; y++){
            for(int x = 0; x < w; x++){
                if (img[(y*w)+x] > 250){
                    for(int t=0; t<180; t++){
                        double r = (((double)x - center_x) * cos((double)t *DEG2RAD)) + (((double)y - center_y) * sin((double)t * DEG2RAD));
                        accu[(int)((round(r+hough_h)*180.0))+t]++;
                    }
                }
            }
        }
        return 0;
    }

    std::vector<std::pair<cv::Vec2d,cv::Vec2d>> get_lines(int threshold)
    {
        std::vector< std::pair< cv::Vec2d, cv::Vec2d>> lines;
        
        if(accu == 0)
            return lines;

        for(int r=0;r<accu_h;r++)
        {
            for(int t=0;t<accu_w;t++)
            {
                if((int)accu[(r*accu_w) + t] >= threshold)
                {
                    int max = accu[(r*accu_w) + t];
                    for(int ly=-4;ly<=4;ly++)
                    {
                        for(int lx=-4;lx<=4;lx++)
                        {
                            if((ly+r>=0 && ly+r<accu_h) && (lx+t>=0 && lx+t<accu_w))
                            {
                                if( (int)accu[( (r+ly)*accu_w) + (t+lx)] > max )
                                {
                                    max = accu[( (r+ly)*accu_w) + (t+lx)];
                                    ly = lx = 5;
                                }
                            }
                        }
                    }

                    if(max > (int)accu[(r*accu_w) + t])
                        continue;

                    int x1, y1, x2, y2;
                    x1 = y1 = x2 = y2 = 0;

                    if(t >= 45 && t <= 135)
                    {
                        x1 = 0;
                        y1 = ((double)(r-(accu_h/2)) - ((x1 - (img_w/2) ) * cos(t * DEG2RAD))) / sin(t * DEG2RAD) + (img_h / 2);
                        x2 = img_w - 0;
                        y2 = ((double)(r-(accu_h/2)) - ((x2 - (img_w/2) ) * cos(t * DEG2RAD))) / sin(t * DEG2RAD) + (img_h / 2);
                    }
                    else
                    {
                        y1 = 0;
                        x1 = ((double)(r-(accu_h/2)) - ((y1 - (img_h/2) ) * sin(t * DEG2RAD))) / cos(t * DEG2RAD) + (img_w / 2);
                        y2 = img_h - 0;
                        x2 = ((double)(r-(accu_h/2)) - ((y2 - (img_h/2) ) * sin(t * DEG2RAD))) / cos(t * DEG2RAD) + (img_w / 2);
                    }

                    lines.push_back(std::pair< cv::Vec2d, cv::Vec2d >(cv::Vec2d(x1,y1), cv::Vec2d(x2,y2)));

                }
            }
        }
        std::cout << "lines: " << lines.size() << " " << threshold << std::endl;
        return lines;
    }
};

int main(){
    
    cv::Mat img_edge;
	cv::Mat img_blur;

    cv::Mat img = cv::imread("../pictures/sudoku.png", 1);
    if (!img.data){
        printf("No image data \n");
        return -1;
    }
	cv::Canny(img, img_edge, 100, 150, 3);
    cv::imwrite("edges.jpg", img_edge);

    Hough h;
    h.hough_transform(img_edge.data, img_edge.cols, img_edge.rows);
    cv::Mat img_res = img.clone();

    std::vector< std::pair<cv::Vec2d,cv::Vec2d>> lines = h.get_lines(160);
    std::vector<std::pair<cv::Vec2d,cv::Vec2d>>::iterator it;
    for(it=lines.begin();it!=lines.end();it++){
        cv::line(img_res, cv::Point(it->first[0], it->first[1]), cv::Point(it->second[0], it->second[1]), cv::Scalar( 0, 0, 255), 2, 8);
    }
    cv::imwrite("result.jpg",img_res);
    return 0;
}