#ifndef FASTEST_DET_H
#define FASTEST_DET_H

#include "detector.h"
#include "net.h"

class FastestDet : public Detector
{
    public:
        virtual void load_param(const char* json_file);
        virtual std::vector<BoxInfo> detect(ncnn::Mat &input);
};
#endif // FASTEST_DET_H