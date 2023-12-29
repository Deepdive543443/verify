#include <iostream>
#include <vector>
#include <random>
#include "nanodet.h"
#include "net.h"
#include "simpleocv.h"


static std::default_random_engine generator;
static std::normal_distribution<float> distribution(0.0, 1.0);


void print_shape(const char* info, ncnn::Mat &output)
{
    std::cout << info << std::endl;
    std::cout << "C: " << output.c << std::endl;
    std::cout << "H: " << output.h << std::endl;
    std::cout << "W: " << output.w << std::endl;
}

void randn_ncnn(ncnn::Mat &mat, int w, int h, int c)
{
    mat.create(w, h, c, (size_t) 4);

    memset(mat.data, 0.f, w * h * c * 4);

    #pragma omp parallel for num_threads(3)
    for (int k = 0; k < c; k++)
    {
        float *c_ptr = mat.channel(k);
        for (int j = 0; j < h; j++)
        {
            for (int i = 0; i < w; i++)
            {
                c_ptr[0] = distribution(generator);
                c_ptr++;
            }
        } 
    }
}

int main(int argc, char** argv)
{
    Nanodet nanodet;
    // nanodet.load_param("../config/nanodet-plus-m_416-int8.json");
    nanodet.load_param("../config/nanodet-plus-m-1.5x_416_int8.json");
    // nanodet.load_param("../config/nanodet-plus-m-1.5x_416.json");
    // nanodet.load_param("../config/nanodet-plus-m_416.json");

    ncnn::Mat input;
    randn_ncnn(input, 416, 416, 3);
    print_shape("data", input);

    static const char* output_names[] = {
        "output", 
        "/head/Concat_output_0",
        "/head/Concat_2_output_0",
        "/head/Concat_4_output_0",
        "/head/Concat_6_output_0"
    };

    for (int i = 0; i < 5; i++)
    {
        ncnn::Extractor ex = nanodet.detector.create_extractor();
        ex.input("data", input);
        ncnn::Mat out;
        ex.extract(output_names[i], out);
        print_shape(output_names[i], out);
    }

    // {
    //     ncnn::Extractor ex = nanodet.detector.create_extractor();
    //     ex.input("data", input);
    //     ncnn::Mat out;
    //     ex.extract("output", out);
    //     print_shape("output", out);
    // }

    // {
    //     ncnn::Extractor ex = nanodet.detector.create_extractor();
    //     ex.input("data", input);
    //     ncnn::Mat stride_out;
    //     ex.extract("/head/Concat_output_0", stride_out);
    //     print_shape("/head/Concat_output_0", stride_out);
    // }

    // {
    //     ncnn::Extractor ex = nanodet.detector.create_extractor();
    //     ex.input("data", input);
    //     ncnn::Mat stride_out;
    //     ex.extract("/head/Concat_2_output_0", stride_out);
    //     print_shape("/head/Concat_2_output_0", stride_out);
    // }

    // {
    //     ncnn::Extractor ex = nanodet.detector.create_extractor();
    //     ex.input("data", input);
    //     ncnn::Mat stride_out;
    //     ex.extract("/head/Concat_4_output_0", stride_out);
    //     print_shape("/head/Concat_4_output_0", stride_out);
    // }


    // {
    //     ncnn::Extractor ex = nanodet.detector.create_extractor();
    //     ex.input("data", input);
    //     ncnn::Mat stride_out;
    //     ex.extract("/head/Concat_6_output_0", stride_out);
    //     print_shape("/head/Concat_6_output_0", stride_out);
    // }


    return 0;
}