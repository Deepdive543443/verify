#include <iostream>
#include <vector>
#include "nanodet.h"
// #include "detector.h"
#include "net.h"
#include "simpleocv.h"

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filereadstream.h"

#include "vision.h"


int main(int argc, char** argv)
{
    // Nanodet 
    Nanodet nanodet;
    nanodet.load_param("../config/nanodet-plus-m_416-int8.json");

    cv::Mat bgr = cv::imread("../image/1 EYFejGUjvjPcc4PZTwoufw.jpg", 1);
    // cv::Mat bgr = cv::imread("../image/3.jpg", 1);
    ncnn::Mat input = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows);

    nanodet.inference_test();
    std::vector<BoxInfo> boxxes = nanodet.detect(input);

    int inference_size = nanodet.input_size[0];
    int drawing_coor = ((float) inference_size / 2 ) - (inference_size / 2);
    float scale = (float) input.w / inference_size;

    draw_bboxes(bgr, boxxes, drawing_coor, scale);

    // cv::Mat resized_cv(to_size, to_size, 3);
    // bordered_resize(input, resized, to_size);
    // resized.to_pixels(resized_cv.data, ncnn::Mat::PIXEL_BGR);
    // // cv::imwrite("../Resized_test.png", resized_cv);

    // // Drawing boxxes
    // std::vector<BoxInfo> boxxes = nanodet.detect(resized);
    // draw_bboxes(resized_cv, boxxes);
    cv::imwrite("../output/test_output.png", bgr);

    // std::cout << "Hello, great detector" << std::endl;
    return 0;
}