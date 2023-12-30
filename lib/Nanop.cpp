#include <iostream>
#include "Nanop.h"

#include "cpu.h"

NanodetP::NanodetP()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int NanodetP::load(const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    Nanodet_plus.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    Nanodet_plus.opt = ncnn::Option();

#if NCNN_VULKAN
    Nanodet_plus.opt.use_vulkan_compute = use_gpu;
#endif

    Nanodet_plus.opt.num_threads = ncnn::get_big_cpu_count();
    Nanodet_plus.opt.blob_allocator = &blob_pool_allocator;
    Nanodet_plus.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "../models/nanodet-%s.param", modeltype);
    sprintf(modelpath, "../models/nanodet-%s.bin", modeltype);

    Nanodet_plus.load_param(parampath);
    Nanodet_plus.load_model(modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int NanodetP::detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold, float nms_threshold)
{
    // Original width and height
    int width = rgb.cols;
    int height = rgb.rows;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    // Resize and make border
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, width, height, w, h);
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    in_pad.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Extractor ex = Nanodet_plus.create_extractor();
    ex.input("data", in_pad);

    std::vector<Object> proposals; // All objects
    // stride 8
    {
        ncnn::Mat cls_pred;
        ncnn::Mat dis_pred;
        ex.extract("cls8", cls_pred);
        ex.extract("dis8", dis_pred);
        cls_pred = cls_pred.reshape(cls_pred.c, cls_pred.w, cls_pred.h);
        dis_pred = dis_pred.reshape(dis_pred.c, dis_pred.w, dis_pred.h);

        std::cout << "cls8: c " << cls_pred.c << std::endl;
        std::cout << "cls8: h " << cls_pred.h << std::endl;
        std::cout << "cls8: w " << cls_pred.w << std::endl;
        std::cout << "dis8: c " << dis_pred.c << std::endl;
        std::cout << "dis8: h " << dis_pred.h << std::endl;
        std::cout << "dis8: w " << dis_pred.w << std::endl;
        std::cout << std::endl;
    }

    // stride 16
    {
        ncnn::Mat cls_pred;
        ncnn::Mat dis_pred;
        ex.extract("cls16", cls_pred);
        ex.extract("dis16", dis_pred);
        cls_pred = cls_pred.reshape(cls_pred.c, cls_pred.w, cls_pred.h);
        dis_pred = dis_pred.reshape(dis_pred.c, dis_pred.w, dis_pred.h);

        std::cout << "cls16: c " << cls_pred.c << std::endl;
        std::cout << "cls16: h " << cls_pred.h << std::endl;
        std::cout << "cls16: w " << cls_pred.w << std::endl;
        std::cout << "dis16: c " << dis_pred.c << std::endl;
        std::cout << "dis16: h " << dis_pred.h << std::endl;
        std::cout << "dis16: w " << dis_pred.w << std::endl;
        std::cout << std::endl;
    }

    // stride 32
    {
        ncnn::Mat cls_pred;
        ncnn::Mat dis_pred;
        ex.extract("cls32", cls_pred);
        ex.extract("dis32", dis_pred);
        cls_pred = cls_pred.reshape(cls_pred.c, cls_pred.w, cls_pred.h);
        dis_pred = dis_pred.reshape(dis_pred.c, dis_pred.w, dis_pred.h);

        std::cout << "cls32: c " << cls_pred.c << std::endl;
        std::cout << "cls32: h " << cls_pred.h << std::endl;
        std::cout << "cls32: w " << cls_pred.w << std::endl;
        std::cout << "dis32: c " << dis_pred.c << std::endl;
        std::cout << "dis32: h " << dis_pred.h << std::endl;
        std::cout << "dis32: w " << dis_pred.w << std::endl;
        std::cout << std::endl;
    }

    // stride 64
    {
        ncnn::Mat cls_pred;
        ncnn::Mat dis_pred;
        ex.extract("cls64", cls_pred);
        ex.extract("dis64", dis_pred);
        cls_pred = cls_pred.reshape(cls_pred.c, cls_pred.w, cls_pred.h);
        dis_pred = dis_pred.reshape(dis_pred.c, dis_pred.w, dis_pred.h);

        std::cout << "cls64: c " << cls_pred.c << std::endl;
        std::cout << "cls64: h " << cls_pred.h << std::endl;
        std::cout << "cls64: w " << cls_pred.w << std::endl;
        std::cout << "dis64: c " << dis_pred.c << std::endl;
        std::cout << "dis64: h " << dis_pred.h << std::endl;
        std::cout << "dis64: w " << dis_pred.w << std::endl;
        std::cout << std::endl;
    }

    return 0;
}