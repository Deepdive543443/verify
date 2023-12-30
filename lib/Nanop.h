# ifndef NANOP_H
# define NANOP_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <net.h>

typedef struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
} Object;

class NanodetP
{
    public:
        NanodetP();
        virtual int load(const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);
        // virtual int load(AAssetManager* mgr, const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);
        virtual int detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold = 0.4f, float nms_threshold = 0.5f);
        // virtual int draw(cv::Mat& rgb, const std::vector<Object>& objects);

    private:
        ncnn::Net Nanodet_plus;
        int target_size;
        float mean_vals[3];
        float norm_vals[3];
        ncnn::UnlockedPoolAllocator blob_pool_allocator;
        ncnn::PoolAllocator workspace_pool_allocator;
};

# endif // NANOP_H