#include <iostream>
#include <vector>
#include "nanodet.h"
#include "net.h"
#include "simpleocv.h"

int main(int argc, char** argv)
{
    // Nanodet 
    Nanodet nanodet;
    // nanodet.load_param("../config/nanodet-plus-m_416-int8.json");
    nanodet.load_param("../config/nanodet-plus-m-1.5x_416_int8.json");
    // nanodet.load_param("../config/nanodet-plus-m-1.5x_416.json");
    // nanodet.load_param("../config/nanodet-plus-m_416.json");
    // nanodet.load_param("../config/nanodet-plus-m-1.5x_416-opt.json");


    cv::Mat bgr = cv::imread("../image/1 EYFejGUjvjPcc4PZTwoufw.jpg", 1);
    // cv::Mat bgr = cv::imread("../image/test.jpg", 1);
    ncnn::Mat input = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows);

    nanodet.inference_test();
    std::vector<BoxInfo> boxxes = nanodet.detect(input);

    nanodet.draw_boxxes(bgr, boxxes);
    cv::imwrite("../output/test_output.png", bgr);
    return 0;
}