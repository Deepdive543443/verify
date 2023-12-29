#include <iostream>
#include <vector>
#include "fastestdet.h"
#include "net.h"
#include "simpleocv.h"

int main(int argc, char** argv)
{
    // Fastest Det
    FastestDet model;
    model.load_param("../config/fastestdet.json");

    model.inference_test();

    cv::Mat bgr = cv::imread("../image/1 EYFejGUjvjPcc4PZTwoufw.jpg", 1);
    // cv::Mat bgr = cv::imread("../image/3.jpg", 1);
    ncnn::Mat input = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows);
    
    cv::imwrite("../output/source.png", bgr);
    std::vector<BoxInfo> boxxes = model.detect(input);

    model.draw_boxxes(bgr, boxxes);
    cv::imwrite("../output/test_output.png", bgr);
    return 0;
}