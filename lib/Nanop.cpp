#include <iostream>
#include "Nanop.h"

#include "cpu.h"

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}


static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}


static void generate_proposals(ncnn::Mat& cls_pred, ncnn::Mat& dis_pred, int stride, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    // cls_pred = cls_pred.reshape(cls_pred.c, cls_pred.w, cls_pred.h);
    // dis_pred = dis_pred.reshape(dis_pred.c, dis_pred.w, dis_pred.h); // 2 hwc

    const int num_grid_x = cls_pred.w;
    const int num_grid_y = cls_pred.h;
    const int num_class = cls_pred.c;
    const int cstep_cls = cls_pred.cstep;

    const int reg_max_1 = dis_pred.w / 4;
    const int hstep_dis = dis_pred.cstep;

    // std::cout << "num_grid_x " << num_grid_x << std::endl;
    // std::cout << "num_grid_y " << num_grid_y << std::endl;
    // std::cout << "num_class " << num_class << std::endl;
    // std::cout << "cstep_cls " << cstep_cls << std::endl;
    // std::cout << "reg_max_1 " << reg_max_1 << std::endl;
    // std::cout << "hstep_dis " << hstep_dis << std::endl;    

    for (int i = 0; i < num_grid_y; i++)
    {
        for (int j = 0; j < num_grid_x; j++)
        {
            float *score_ptr = cls_pred.row(i) + j;
            float max_score = -FLT_MAX;
            int max_label = -1;

            for (int cls = 0; cls < num_class; cls++)
            {
                if (score_ptr[cls * cstep_cls] > max_score)
                {
                    max_score = score_ptr[cls * cstep_cls];
                    max_label = cls;
                }
            }

            if (max_score >= prob_threshold)
            {
                ncnn::Mat bbox_pred(reg_max_1, 4, (void*) dis_pred.row(j) + (i * hstep_dis));
                // std::cout << "max_score " << max_score << std::endl;
                // std::cout << "max_label " << max_label << std::endl;
                {
                    ncnn::Layer* softmax = ncnn::create_layer("Softmax");

                    ncnn::ParamDict pd;
                    pd.set(0, 1); // axis
                    pd.set(1, 1);
                    softmax->load_param(pd);

                    ncnn::Option opt;
                    opt.num_threads = 1;
                    opt.use_packing_layout = false;

                    softmax->create_pipeline(opt);
                    softmax->forward_inplace(bbox_pred, opt);
                    softmax->destroy_pipeline(opt);

                    delete softmax;
                }

                float pred_ltrb[4];
                for (int k = 0; k < 4; k++)
                {
                    float dis = 0.f;
                    const float* dis_after_sm = bbox_pred.row(k);
                    for (int l = 0; l < reg_max_1; l++)
                    {
                        dis += l * dis_after_sm[l];
                    }
                    pred_ltrb[k] = dis * stride;
                }

                float x_center = j * stride;
                float y_center = i * stride;

                // float xmin = x_center - pred_ltrb[0];
                // float ymin = y_center - pred_ltrb[1];
                // float xmax = x_center + pred_ltrb[2];
                // float ymax = y_center + pred_ltrb[3];

                Object obj;
                obj.rect.x = x_center - pred_ltrb[0];
                obj.rect.y = y_center - pred_ltrb[1];
                obj.rect.width =  pred_ltrb[2] + pred_ltrb[0];
                obj.rect.height = pred_ltrb[3] + pred_ltrb[1];
                // obj.rect.x = xmin;
                // obj.rect.y = ymin;
                // obj.rect.width = xmax - xmin;
                // obj.rect.height = ymax - ymin;
                obj.label = max_label;
                obj.prob = max_score;
                objects.push_back(obj);

                // std::cout << "obj.rect.x " << obj.rect.x << std::endl;
                // std::cout << "obj.rect.y " << obj.rect.y << std::endl;
                // std::cout << "obj.rect.width " << obj.rect.width << std::endl;
                // std::cout << "obj.rect.height " << obj.rect.height << std::endl;
                // std::cout << "obj.label " << obj.label << std::endl;
                // std::cout << "obj.prob " << obj.prob << std::endl;
            }
        }
    }
}


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
        scale = (float) target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float) target_size / h;
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
        // cls_pred = cls_pred.reshape(cls_pred.c, cls_pred.w, cls_pred.h);
        // dis_pred = dis_pred.reshape(dis_pred.c, dis_pred.w, dis_pred.h);

        // std::cout << "cls8: c " << cls_pred.c << std::endl;
        // std::cout << "cls8: h " << cls_pred.h << std::endl;
        // std::cout << "cls8: w " << cls_pred.w << std::endl;
        // std::cout << "dis8: c " << dis_pred.c << std::endl;
        // std::cout << "dis8: h " << dis_pred.h << std::endl;
        // std::cout << "dis8: w " << dis_pred.w << std::endl;
        std::vector<Object> obj8;
        generate_proposals(cls_pred, dis_pred, 8, in_pad, prob_threshold, obj8);
        proposals.insert(proposals.end(), obj8.begin(), obj8.end());
    }

    // stride 16
    {
        ncnn::Mat cls_pred;
        ncnn::Mat dis_pred;
        ex.extract("cls16", cls_pred);
        ex.extract("dis16", dis_pred);
        // cls_pred = cls_pred.reshape(cls_pred.c, cls_pred.w, cls_pred.h);
        // dis_pred = dis_pred.reshape(dis_pred.c, dis_pred.w, dis_pred.h);

        // std::cout << "cls16: c " << cls_pred.c << std::endl;
        // std::cout << "cls16: h " << cls_pred.h << std::endl;
        // std::cout << "cls16: w " << cls_pred.w << std::endl;
        // std::cout << "dis16: c " << dis_pred.c << std::endl;
        // std::cout << "dis16: h " << dis_pred.h << std::endl;
        // std::cout << "dis16: w " << dis_pred.w << std::endl;
        std::vector<Object> obj16;
        generate_proposals(cls_pred, dis_pred, 16, in_pad, prob_threshold, obj16);
        proposals.insert(proposals.end(), obj16.begin(), obj16.end());
    }

    // stride 32
    {
        ncnn::Mat cls_pred;
        ncnn::Mat dis_pred;
        ex.extract("cls32", cls_pred);
        ex.extract("dis32", dis_pred);
        // cls_pred = cls_pred.reshape(cls_pred.c, cls_pred.w, cls_pred.h);
        // dis_pred = dis_pred.reshape(dis_pred.c, dis_pred.w, dis_pred.h);

        // std::cout << "cls32: c " << cls_pred.c << std::endl;
        // std::cout << "cls32: h " << cls_pred.h << std::endl;
        // std::cout << "cls32: w " << cls_pred.w << std::endl;
        // std::cout << "dis32: c " << dis_pred.c << std::endl;
        // std::cout << "dis32: h " << dis_pred.h << std::endl;
        // std::cout << "dis32: w " << dis_pred.w << std::endl;
        std::vector<Object> obj32;
        generate_proposals(cls_pred, dis_pred, 32, in_pad, prob_threshold, obj32);
        proposals.insert(proposals.end(), obj32.begin(), obj32.end());
        
    }

    // stride 64
    {
        ncnn::Mat cls_pred;
        ncnn::Mat dis_pred;
        ex.extract("cls64", cls_pred);
        ex.extract("dis64", dis_pred);
        // cls_pred = cls_pred.reshape(cls_pred.c, cls_pred.w, cls_pred.h);
        // dis_pred = dis_pred.reshape(dis_pred.c, dis_pred.w, dis_pred.h);

        // std::cout << "cls64: c " << cls_pred.c << std::endl;
        // std::cout << "cls64: h " << cls_pred.h << std::endl;
        // std::cout << "cls64: w " << cls_pred.w << std::endl;
        // std::cout << "dis64: c " << dis_pred.c << std::endl;
        // std::cout << "dis64: h " << dis_pred.h << std::endl;
        // std::cout << "dis64: w " << dis_pred.w << std::endl;
        std::vector<Object> obj64;
        generate_proposals(cls_pred, dis_pred, 64, in_pad, prob_threshold, obj64);
        proposals.insert(proposals.end(), obj64.begin(), obj64.end());
    }

    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
    struct
    {
        bool operator()(const Object& a, const Object& b) const
        {
            return a.rect.area() > b.rect.area();
        }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);

    // for (Object obj : objects)
    // {
    //     std::cout << "obj.rect.x " << obj.rect.x << std::endl;
    //     std::cout << "obj.rect.y " << obj.rect.y << std::endl;
    //     std::cout << "obj.rect.width " << obj.rect.width << std::endl;
    //     std::cout << "obj.rect.height " << obj.rect.height << std::endl;
    //     std::cout << "obj.label " << obj.label << std::endl;
    //     std::cout << "obj.prob " << obj.prob << std::endl;
    // }

    return 0;
}

int NanodetP::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    static const unsigned char colors[19][3] = {
        { 54,  67, 244},
        { 99,  30, 233},
        {176,  39, 156},
        {183,  58, 103},
        {181,  81,  63},
        {243, 150,  33},
        {244, 169,   3},
        {212, 188,   0},
        {136, 150,   0},
        { 80, 175,  76},
        { 74, 195, 139},
        { 57, 220, 205},
        { 59, 235, 255},
        {  7, 193, 255},
        {  0, 152, 255},
        { 34,  87, 255},
        { 72,  85, 121},
        {158, 158, 158},
        {139, 125,  96}
    };

    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

//         fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
//                 obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        const unsigned char* color = colors[color_index % 19];
        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);

        cv::rectangle(rgb, obj.rect, cc, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cc, -1);

        cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc, 1);
    }

    return 0;
}