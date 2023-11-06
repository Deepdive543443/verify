#include <iostream>
#include <vector>
#include "fastestdet.h"
#include "net.h"
#include "simpleocv.h"

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filereadstream.h"

#include "vision.h"


int main(int argc, char** argv)
{
    // Fastest Det
    FastestDet model;
    model.load_param("../config/fastestdet.json");

    model.inference_test();

    // cv::Mat bgr = cv::imread("../image/1 EYFejGUjvjPcc4PZTwoufw.jpg", 1);
    cv::Mat bgr = cv::imread("../image/3.jpg", 1);
    ncnn::Mat input = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows);
    
    cv::imwrite("../output/source.png", bgr);
    std::vector<BoxInfo> boxxes = model.detect(input);

    float x_scale = (float) input.w / model.input_size[0];
    float y_scale = (float) input.h / model.input_size[1];

    draw_bboxes(bgr, boxxes, 0, 1, 1);
    cv::imwrite("../output/test_output.png", bgr);
    return 0;
}