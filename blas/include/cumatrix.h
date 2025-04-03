#ifndef CUMATRIX
#define CUMATRIX
#include "/codes/cv/include/core.hpp"
    /*一维矩阵数据*/
class Matrixlevel1datas
{
private:
public:
    Matrixlevel1datas() = delete;

    explicit Matrixlevel1datas
    (void *datas, nvinfer1::DataType type, std::size_t size, std::size_t shape);

    ~Matrixlevel1datas();
        //矩阵数据    
    void *hos_datas=nullptr;
        //矩阵类型
    nvinfer1::DataType hos_type;
        //内存字节
    std::size_t memorysize;
        //元素数量
    std::size_t memoryelements;
        //一维长度
    std::size_t level1shape=0;
};
    /*二维矩阵数据*/
class Matrixlevel2datas
{
private:
    /* data */
public:
    Matrixlevel2datas() = delete;

    explicit Matrixlevel2datas
    (void *datas, nvinfer1::DataType type, std::size_t size, nvinfer1::Dims2 shape, bool T); 

    ~Matrixlevel2datas();

    void *hos_datas=nullptr;
        //矩阵类型
    nvinfer1::DataType hos_type;
        //内存字节
    std::size_t memorysize;
        //元素数量
    std::size_t memoryelements;
        //二维行列大小
    std::size_t level2rows=0;
    std::size_t level2cols=0;
        //是否行优先
    bool T;
};

class BlasOperator
{
private:
    cublasHandle_t cubl_hand_;
public:
    explicit BlasOperator();
    ~BlasOperator();

    template<typename BlasMatrix>
    void executeV1
    (BlasMatrix&& blasop, 
    std::vector<Matrixlevel1datas>& Matrixdatasinput_vec, 
    std::vector<Matrixlevel1datas>& Matrixdatasoutput_vec) noexcept
    {
        blasop.run(this->cubl_hand_, Matrixdatasinput_vec, Matrixdatasoutput_vec);
        blasop.destroy();
    }

    template<typename BlasMatrix>
    void executeV2
    (BlasMatrix&& blasop, 
    std::vector<Matrixlevel2datas>& Matrixdatasinput_vec, 
    std::vector<Matrixlevel2datas>& Matrixdatasoutput_vec) noexcept
    {
        blasop.run(this->cubl_hand_, Matrixdatasinput_vec, Matrixdatasoutput_vec);
        blasop.destroy();
    }
};

namespace level1
{
    /*scal向量缩放*/
class blasDscal
{
private:
        //缩放因子
    const float *alpha;

    void *dev_src=nullptr;
    cudaStream_t str;
    cudaEvent_t start;
    cudaEvent_t end;
    std::atomic<bool> isdestroy{false};
public:
    blasDscal() = delete;
    explicit blasDscal(float *alpha);
    ~blasDscal();

    cublasStatus_t run
    (cublasHandle_t& cubl_hand_, 
    std::vector<Matrixlevel1datas>& Matrixdatasinput_vec, 
    std::vector<Matrixlevel1datas>& Matrixdatasoutput_vec);

    void destroy() noexcept;
};

    /*axpy双向量 A * alpha + B*/
class blasAxpy
{
private:
        //A缩放因子
    const float *alpha;

    void *dev_src0=nullptr;
    void *dev_src1=nullptr;
    cudaStream_t str;
    cudaEvent_t start;
    cudaEvent_t end;
    std::atomic<bool> isdestroy{false};
public:
    blasAxpy() = delete;
    explicit blasAxpy(float *alpha);
    ~blasAxpy();

    cublasStatus_t run
    (cublasHandle_t& cubl_hand_, 
    std::vector<Matrixlevel1datas>& Matrixdatasinput_vec, 
    std::vector<Matrixlevel1datas>& Matrixdatasoutput_vec);

    void destroy() noexcept;
};

} // namespace level1

namespace level2
{
    /*gemm双矩阵 C=α⋅(A×B)+β⋅C混合精度*/
class blasGemmEx
{
private:
        //相乘缩放因子
    float *alpha;
        //原始缩放因子
    float *beta;

    void *dev_src0=nullptr;
    void *dev_src1=nullptr;
    void *dev_dst=nullptr;
    cudaStream_t str;
    cudaEvent_t start;
    cudaEvent_t end;
    std::atomic<bool> isdestroy{false};
public:
    blasGemmEx() = delete;
    explicit blasGemmEx(float *alpha, float *beta);
    ~blasGemmEx();

    cublasStatus_t run
    (cublasHandle_t& cubl_hand_, 
    std::vector<Matrixlevel2datas>& Matrixdatasinput_vec, 
    std::vector<Matrixlevel2datas>& Matrixdatasoutput_vec);

    static void CUDART_CB timingcallback
    (cudaStream_t str, cudaError_t status, void *usrdatas) noexcept;

    void destroy() noexcept;
};
    
} // namespace level2

#endif