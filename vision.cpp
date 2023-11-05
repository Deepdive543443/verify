#include "vision.h"

void bordered_resize(ncnn::Mat &src, ncnn::Mat &dst, int dst_w, int draw_coor)
{
    int dst_h = src.h * ((float) dst_w / src.w);
    
    ncnn::Mat resized(dst_w, dst_h, 3);
    ncnn::resize_bilinear(
        src, 
        resized, 
        dst_w, 
        dst_h
    );

    float *dst_ptr = (float *) dst.data;
    float *resized_src_ptr = (float *) resized.data;

    int src_cstep = resized.cstep;
    int dst_cstep = dst.cstep;

    memset(dst_ptr, 0.f, dst_w * dst_w * 3);
    
    for (int c=0; c<3; c++)
    {
        dst_ptr = dst.row(draw_coor) + (c * dst_cstep);
        resized_src_ptr = resized.row(0) + (c * src_cstep);
        for (int j = 0; j < dst_h; j++)
        {
            for (int i = 0; i < dst_w; i++)
            {
                dst_ptr[0] = resized_src_ptr[0];
                dst_ptr++;
                resized_src_ptr++;
            }
        }
    }
}

void draw_bboxes(const cv::Mat& image, const std::vector<BoxInfo>& bboxes, int v_shift, float x_scaler, float y_scaler)
{
    printf("=======================================\n");
    printf("% 12s % 2s % ", "Label", "Score");
    printf("% 4s % 4s % 4s % 4s\n\n","x1", "y1", "x2", "y2");
    for (size_t i = 0; i < bboxes.size(); i++)
    {
        const BoxInfo& bbox = bboxes[i];

        float x1 = bbox.x1;
        float y1 = bbox.y1;
        float w = (bbox.x2 - x1);
        float h = (bbox.y2 - y1);

        y1 += v_shift;
        x1 *= x_scaler;
        y1 *= y_scaler;
        w *= x_scaler;
        h *= y_scaler;

        uint8_t *rgba = (uint8_t *) malloc(4);
        
        rgba[0] = color_list[bbox.label][0];
        rgba[1] = color_list[bbox.label][1];
        rgba[2] = color_list[bbox.label][2];
        rgba[3] = 255;

        printf("% 12s %02.3f % ", class_names[bbox.label], bbox.score);
        printf("% 4d % 4d % 4d % 4d\n",(int) x1, (int) y1, (int) w, (int) h);

        int *color = (int *) rgba;

        ncnn::draw_rectangle_c3(
            image.data,
            image.cols,
            image.rows,
            (int) x1,
            (int) y1,
            (int) w,
            (int) h,
            color[0], 
            2
        );

        ncnn::draw_text_c3(
            image.data,
            image.cols,
            image.rows, 
            class_names[bbox.label],
            (int) x1 + 1, 
            (int) y1 + 1,
            7, 
            color[0]
        );

        free(rgba);
    }
    printf("=======================================\n");
}