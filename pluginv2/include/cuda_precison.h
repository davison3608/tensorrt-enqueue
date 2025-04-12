#ifndef PRECISON_DYNAMIC
#define PRECISON_DYNAMIC
#include "/codes/cv/include/core.hpp"
    //半精度压缩
namespace AccHalf
{
    //传递已开辟float内存 未开辟half指针 矩阵元素数目
void Accuracyhalf(float *inputs, void **outputs, int32_t nbelements);

    //传递已开辟half内存 未开辟float指针 矩阵元素数目
void Accuracyhalfdeser(half *inputs, void **outputs, int32_t nbelements);
} // namespace AccHalf

__global__ void kernel_kflo2khal
(void *inputs, void *outputs, int32_t nbelements);

__global__ void kernel_kflo2khaldeser
(void *inputs, void *outputs, int32_t nbelements);

    //int8精度量化
namespace AccInt8
{
    
} // namespace AccInt8

__global__ void kernel_dataprint(void *ptr, int size);

    //数据复制打印
template <typename type>
void cudaMemcpydataShow(std::vector<type> vec_datas);

#endif