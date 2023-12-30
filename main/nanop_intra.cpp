#include <iostream>
#include "Nanop.h"

#include <net.h>

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filereadstream.h"

void load_param(const char* json_file, NanodetP &nanodetP)
{
    FILE* fp = fopen(json_file, "rb"); 
    char readBuffer[4000];
    rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer)); 
    rapidjson::Document doc; 
    doc.ParseStream(is); 
    fclose(fp);

    float mean_vals[3];
    float norm_vals[3];

    for (int i = 0; i < 3; i++)
    {
        mean_vals[i] = doc["config"]["mean_vals"][i].GetFloat();
        norm_vals[i] = doc["config"]["norm_vals"][i].GetFloat();
    }

    nanodetP.load(
        doc["name"].GetString(),
        doc["config"]["input_shape"].GetInt(),
        mean_vals,
        norm_vals,
        0
    );
}

int main(int argc, char** argv)
{
    NanodetP nanodetp;
    load_param("../config/nanodet-plus-m-1.5x_416_new_infra.json", nanodetp);
    cv::Mat bgr = cv::imread("../image/1 EYFejGUjvjPcc4PZTwoufw.jpg", 1);
    std::cout <<" Hello Nanoi" << std::endl;

    std::vector<Object> objects;
    nanodetp.detect(bgr, objects);
    nanodetp.draw(bgr, objects);
    cv::imwrite("Testing/test_output.png", bgr);
    return 0;
}