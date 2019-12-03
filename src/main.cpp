/** 
 * clone image
 * */

#include "opencv2/opencv.hpp"
#include "seamless_cloning.hpp"
#include <iostream>

using namespace cv;

int main( int argc, const char** argv )
{
    Mat src = imread("../data/src_new.jpg");
    Mat dst = imread("../data/dst.jpg");
    // Create an all white mask
    // Mat src_mask = 255 * Mat::ones(src.rows, src.cols, src.depth());
    Mat src_mask = imread("../data/mask_new.jpg");
    // The location of the center of the src in the dst
    Point center(dst.cols/2,dst.rows/2);
    // Seamlessly clone src into dst and put the results in output
    Mat normal_clone;
    Mat mixed_clone;
    Mat monochrome_clone;
    float r, g, b = 255; 255; 255;
    customCV::seamlessClone(src, dst, src_mask, center, normal_clone, NORMAL_CLONE);
    // customCV::seamlessClone(src, dst, src_mask, center, mixed_clone, MIXED_CLONE);
    // customCV::seamlessClone(src, dst, src_mask, center, monochrome_clone, MONOCHROME_TRANSFER);
    // customCV::colorChange(src, src_mask, dst, r, g, b);
    cv::imwrite("../data/normal_clone.jpg", normal_clone);
    // cv::imwrite("../data/mixed_clone.jpg", mixed_clone);
    // cv::imwrite("../data/monochrome_clone.jpg", monochrome_clone);
    std::cout << "The end of clone" << std::endl;
    // imshow("colorchange",dst);
    // cv::imshow("normal_clone",normal_clone);
    // imshow("minxed_clone",mixed_clone);
    // imshow("monochrome_clone",monochrome_clone);
    // imshow("wood",dst);
    // imshow("lovepapa",src);
    cv::waitKey(0);
    // cv::destroyAllWindows();
    return 0;
}