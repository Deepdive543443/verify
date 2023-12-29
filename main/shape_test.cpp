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
    // nanodet.load_param("../config/nanodet-plus-m-1.5x_416_int8.json");
    // nanodet.load_param("../config/nanodet-plus-m-1.5x_416.json");
    nanodet.load_param("../config/nanodet-plus-m-1.5x_416_multi_output.json");
    // nanodet.load_param("../config/nanodet-ELite1_416.json");
    // nanodet.load_param("../config/nanodet-plus-m_416.json");

    ncnn::Mat input;
    randn_ncnn(input, 416, 320, 3);
    print_shape("data", input);

    const std::vector<const char*>& output_names = nanodet.detector.output_names();
    ncnn::Extractor ex = nanodet.detector.create_extractor();
    ex.input("data", input);
    for (const char* name : output_names)
    {
        ncnn::Mat out;
        ex.extract(name, out);
        out = out.reshape(out.c, out.w, out.h);
        print_shape(name, out);
    }
    return 0;
}