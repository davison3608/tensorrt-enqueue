#ifndef CUDA_DYNAMIC_OP
#define CUDA_DYNAMIC_OP
#include "core.hpp"
#include "cuda_core.h"


class BasePluginDynamic: public nvinfer1::IPluginV2DynamicExt
{
private:
    /* data */
public:
    BasePluginDynamic(/* args */);
    ~BasePluginDynamic();
};

#ifndef PLUGINV2_DYNAMIC_REGISTER
#define PLUGINV2_DYNAMIC_REGISTER

#endif

#endif