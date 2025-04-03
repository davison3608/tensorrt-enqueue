#include "/codes/blas/include/cumatrix.h"
using namespace std;
using namespace nvinfer1;
/*一维矩阵数据*/
Matrixlevel1datas::Matrixlevel1datas
(void *datas, DataType type, std::size_t size, std::size_t shape): hos_type(type), memorysize(size)
{
    this->level1shape = shape;

    switch (this->hos_type)
    {
    case DataType::kFLOAT:
    this->memoryelements = this->memorysize / sizeof(float);
    assert(this->memoryelements == shape);
    cudaMallocHost(&this->hos_datas, this->memorysize);
    //复制
    if (datas != nullptr)
        std::memcpy(this->hos_datas, datas, this->memorysize);
    break;

    case DataType::kHALF:
    //处理 half
    this->memoryelements = this->memorysize / sizeof(half);
    assert(this->memoryelements == shape);
    cudaMallocHost(&this->hos_datas, this->memorysize);
    if (datas != nullptr)
        std::memcpy(this->hos_datas, datas, this->memorysize);
    break;

    default:
        throw std::runtime_error("Unsupported data type");
    }
}
Matrixlevel1datas::~Matrixlevel1datas()
{
    cudaFreeHost(this->hos_datas);
}
/*二维矩阵数据*/
Matrixlevel2datas::Matrixlevel2datas
(void *datas, nvinfer1::DataType type, std::size_t size, nvinfer1::Dims2 shape, bool T): 
hos_type(type), memorysize(size), T(T)
{
    this->level2rows=shape.d[0];
    this->level2cols=shape.d[1];

    switch (type)
    {
    case DataType::kFLOAT:
    this->memoryelements=this->memorysize / sizeof(float);
    assert(this->level2rows*this->level2cols == this->memoryelements);
    cudaMallocHost(&this->hos_datas, this->memorysize);
    if (datas != nullptr)
        std::memcpy(this->hos_datas, datas, this->memorysize);
    break;

    case DataType::kHALF:
    this->memoryelements=this->memorysize / sizeof(half);
    assert(this->level2rows*this->level2cols == this->memoryelements);
    cudaMallocHost(&this->hos_datas, this->memorysize);
    if (datas != nullptr)
        std::memcpy(this->hos_datas, datas, this->memorysize);
    break;

    default:
        throw std::runtime_error("Unsupported data type");
    }
}
Matrixlevel2datas::~Matrixlevel2datas()
{
    cudaFreeHost(this->hos_datas);
}

//执行对象
BlasOperator::BlasOperator()
{
    cublasCreate_v2(&this->cubl_hand_);
}
BlasOperator::~BlasOperator()
{
    cublasDestroy_v2(this->cubl_hand_);
}

// level1
/*scal向量缩放*/  
level1::blasDscal::blasDscal(float *alpha) : alpha(alpha)
{}
level1::blasDscal::~blasDscal()
{
    if (!this->isdestroy.load()) {
    cudaFree(this->dev_src);
    cudaEventDestroy(this->start);
    cudaEventDestroy(this->end);
    cudaStreamDestroy(this->str);
    this->isdestroy.store(true);
    }
}
cublasStatus_t level1::blasDscal::run(cublasHandle_t &cubl_hand_,
    std::vector<Matrixlevel1datas> &Matrixdatasinput_vec,
    std::vector<Matrixlevel1datas> &Matrixdatasoutput_vec)
{
    cudaStreamCreateWithFlags(&this->str, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&this->start, cudaEventDefault);
    cudaEventCreateWithFlags(&this->end, cudaEventDefault);

    assert(Matrixdatasinput_vec.size() == 1);
    auto devtype = Matrixdatasinput_vec[0].hos_type;
    auto devsize = Matrixdatasinput_vec[0].memorysize;
    auto develements = Matrixdatasinput_vec[0].level1shape;

    //检查类型与形状
    assert(Matrixdatasoutput_vec.size() == 1);
    assert(devtype == Matrixdatasoutput_vec[0].hos_type);
    assert(devsize == Matrixdatasoutput_vec[0].memorysize);
    assert(develements == Matrixdatasoutput_vec[0].level1shape);

    cublasSetStream(cubl_hand_, this->str);

    switch (devtype)
    {
    case DataType::kFLOAT:
        //形状不变
    cudaMallocAsync(&this->dev_src, devsize, this->str);
    CUDA_CHECK(cudaMemcpyAsync
    (this->dev_src, Matrixdatasinput_vec[0].hos_datas, devsize, cudaMemcpyHostToDevice, this->str)); 
        break;

    default:
    std::cerr<<"dscal不支持的精度\n";
    exit(false);
    }

    cudaStreamSynchronize(this->str);
    cudaEventRecord(start);
    //执行
    cublasStatus_t status=cublasSscal
    (cubl_hand_, develements, this->alpha, static_cast<float*>(this->dev_src), 1);
    assert(status == CUBLAS_STATUS_SUCCESS);
    
    cudaStreamSynchronize(this->str);
    cudaEventRecord(this->end);

    CUDA_CHECK(cudaMemcpyAsync
    (Matrixdatasoutput_vec[0].hos_datas, this->dev_src, devsize, cudaMemcpyDeviceToHost, str));
    cudaStreamSynchronize(this->str);

    float time;
    cudaEventElapsedTime(&time, this->start, this->end);

    std::cout<<"dscal计时 "<<time<<" ms\n";

    return status;
}
void level1::blasDscal::destroy() noexcept
{
    cudaFree(this->dev_src);
    cudaEventDestroy(this->start);
    cudaEventDestroy(this->end);
    cudaStreamDestroy(this->str);
    this->isdestroy.store(true);
}

/*axpy双向量 A * alpha + B*/
level1::blasAxpy::blasAxpy(float *alpha): alpha(alpha)
{}
level1::blasAxpy::~blasAxpy()
{
    if (!this->isdestroy.load()) {
    cudaFreeAsync(this->dev_src0, this->str);
    cudaFreeAsync(this->dev_src1, this->str);
    cudaEventDestroy(this->start);
    cudaEventDestroy(this->end);
    cudaStreamDestroy(this->str);
    this->isdestroy.store(true);
    }
}
cublasStatus_t level1::blasAxpy::run(cublasHandle_t& cubl_hand_,
    std::vector<Matrixlevel1datas>& Matrixdatasinput_vec,
    std::vector<Matrixlevel1datas>& Matrixdatasoutput_vec)
{
    cudaStreamCreateWithFlags(&this->str, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&this->start, cudaEventDefault);
    cudaEventCreateWithFlags(&this->end, cudaEventDefault);

    //双向量信息一致
    assert(Matrixdatasinput_vec.size() == 2);
    auto devtype=Matrixdatasinput_vec[0].hos_type;
    assert(devtype == Matrixdatasinput_vec[1].hos_type);
    auto devsize=Matrixdatasinput_vec[0].memorysize;
    assert(devsize == Matrixdatasinput_vec[1].memorysize);
    auto develements=Matrixdatasinput_vec[0].memoryelements;
    assert(develements == Matrixdatasinput_vec[1].memoryelements);

    //输入输出是否匹配
    assert(Matrixdatasoutput_vec.size() == 1);
    assert(devtype == Matrixdatasoutput_vec[0].hos_type);
    assert(devsize == Matrixdatasoutput_vec[0].memorysize);
    assert(develements == Matrixdatasoutput_vec[0].memoryelements);

    cublasSetStream(cubl_hand_, this->str);

    switch (devtype)
    {
    case DataType::kFLOAT:
    cudaMallocAsync(&this->dev_src0, devsize, this->str);
    cudaMallocAsync(&this->dev_src1, devsize, this->str);
    CUDA_CHECK(cudaMemcpyAsync
    (this->dev_src0, Matrixdatasinput_vec[0].hos_datas, devsize, cudaMemcpyHostToDevice, str));
    CUDA_CHECK(cudaMemcpyAsync
    (this->dev_src1, Matrixdatasinput_vec[1].hos_datas, devsize, cudaMemcpyHostToDevice, str));
        break;

    default:
    std::cerr<<"axpy不支持的精度\n";
    exit(false);
    }

    cudaStreamSynchronize(this->str);
    cudaEventRecord(this->start);
    //执行
    //cublasAxpyEx混合精度
     cublasStatus_t status=cublasSaxpy
     (cubl_hand_, develements, this->alpha, 
     static_cast<float*>(this->dev_src0), 1, static_cast<float*>(this->dev_src1), 1);
    assert(status == CUBLAS_STATUS_SUCCESS);

    cudaStreamSynchronize(this->str);
    cudaEventRecord(this->end);

    CUDA_CHECK(cudaMemcpyAsync
    (Matrixdatasoutput_vec[0].hos_datas, this->dev_src1, devsize, cudaMemcpyDeviceToHost, str));
    cudaStreamSynchronize(this->str);

    float time;
    cudaEventElapsedTime(&time, this->start, this->end);

    std::cout<<"axpy计时 "<<time<<" ms\n";

    return status;
}
void level1::blasAxpy::destroy() noexcept
{
    cudaFree(this->dev_src0);
    cudaFree(this->dev_src1);
    cudaEventDestroy(this->start);
    cudaEventDestroy(this->end);
    cudaStreamDestroy(this->str);
    this->isdestroy.store(true);
}

level2::blasGemmEx::blasGemmEx(float *alpha, float *beta): alpha(alpha), beta(beta)
{}
level2::blasGemmEx::~blasGemmEx()
{
    if (!this->isdestroy.load()) {
    cudaFreeAsync(this->dev_src0, this->str);
    cudaFreeAsync(this->dev_src1, this->str);
    cudaFreeAsync(this->dev_dst, this->str);
    cudaEventDestroy(this->start);
    cudaEventDestroy(this->end);
    cudaStreamDestroy(this->str);
    this->isdestroy.store(true);
    }
}
cublasStatus_t level2::blasGemmEx::run(cublasHandle_t& cubl_hand_,
    std::vector<Matrixlevel2datas>& Matrixdatasinput_vec,
    std::vector<Matrixlevel2datas>& Matrixdatasoutput_vec)
{
    cudaStreamCreateWithFlags(&this->str, cudaStreamNonBlocking);
    cudaEventCreate(&this->start);
    cudaEventCreate(&this->end);

    //双矩阵输入检查
    assert(Matrixdatasinput_vec.size() == 2);
    auto row0=Matrixdatasinput_vec[0].level2rows;
    auto row1=Matrixdatasinput_vec[1].level2rows;
    auto col0=Matrixdatasinput_vec[0].level2cols;
    auto col1=Matrixdatasinput_vec[1].level2cols;
    auto devsize0=Matrixdatasinput_vec[0].memorysize;
    auto devsize1=Matrixdatasinput_vec[1].memorysize;
    DataType devtype=Matrixdatasinput_vec[0].hos_type; //输入双矩阵精度要求一致

    if (col0 != row1) {
    std::cerr<<"gemm输入矩阵行列不匹配\n";
    exit(false);
    } else if (devtype != Matrixdatasinput_vec[1].hos_type) {
    std::cerr<<"gemm输入矩阵精度不匹配\n";
    exit(false);    
    }

    cublasOperation_t Matrix_OP0;
    cublasOperation_t Matrix_OP1;
    std::size_t Matrix_MDim0;
    std::size_t Matrix_MDim1;
    if (Matrixdatasinput_vec[0].T) {
        Matrix_OP0=CUBLAS_OP_T; //行优先必须转置
        Matrix_MDim0=col0; //行优先主维度为行长度
    }
    else {
        Matrix_OP0=CUBLAS_OP_N; //列优先无需转置
        Matrix_MDim0=row0; //列优先主维度为列长度
    }
    if (Matrixdatasinput_vec[1].T) {
        Matrix_OP1=CUBLAS_OP_T; //行优先必须转置
        Matrix_MDim1=col1; //行优先主维度为行长度
    }
    else {
        Matrix_OP1=CUBLAS_OP_N; //列优先无需转置
        Matrix_MDim1=row1; //行优先主维度为行长度
    }
    
    //输出矩阵检查
    assert(Matrixdatasoutput_vec.size() == 1);
    auto out_row=Matrixdatasoutput_vec[0].level2rows;
    auto out_col=Matrixdatasoutput_vec[0].level2cols;
    auto out_devtype=Matrixdatasoutput_vec[0].hos_type;
    auto out_devsize=Matrixdatasoutput_vec[0].memorysize;

    if (out_row != row0) {
    std::cerr<<"gemm输出矩阵行列不匹配\n";
    exit(false);
    } else if (out_col != col1) {
    std::cerr<<"gemm输出矩阵行列不匹配\n";
    exit(false);
    }

    assert(Matrixdatasoutput_vec[0].T == false); //输出总以列优先展平
    cublasOperation_t out_Matrix_OP=CUBLAS_OP_N;

    cublasSetStream(cubl_hand_, this->str);

    //输入分配
    CUDA_CHECK(cudaMallocAsync(&this->dev_src0, devsize0, this->str));
    CUDA_CHECK(cudaMallocAsync(&this->dev_src1, devsize1, this->str));
    
    array<float, 4> datas_arr0{1.2f, 2.2f, 3.2f, 4.2f};
    CUDA_CHECK(cudaMemcpyAsync
    (this->dev_src0, datas_arr0.data(), datas_arr0.size() * sizeof(float), cudaMemcpyHostToDevice, str));
    
    CUDA_CHECK(cudaMemcpyAsync
    (this->dev_src0, Matrixdatasinput_vec[0].hos_datas, devsize0, cudaMemcpyHostToDevice, str));
    CUDA_CHECK(cudaMemcpyAsync
    (this->dev_src1, Matrixdatasinput_vec[1].hos_datas, devsize1, cudaMemcpyHostToDevice, str));

    //输出分配
    switch (out_devtype)
    {
    case DataType::kFLOAT:
    cudaMallocAsync(&this->dev_dst, out_devsize, this->str);
    CUDA_CHECK(cudaMemsetAsync
    (this->dev_dst, 0, out_devsize, this->str));
        break;
    default:
    std::cerr<<"gemm不支持的精度\n";
    exit(false);  
    }

    //尝试tensorcore
    bool istensorcore;
    cublasStatus_t status;
        //半精度输入 行列对齐要求
    if ((devtype == DataType::kHALF) && 
        (row0%8 == 0 && col0%8 == 0) &&
        (row1%8 == 0 && col1%8 == 0)) {
        status=cublasSetMathMode(cubl_hand_, CUBLAS_TENSOR_OP_MATH);
        assert(status == CUBLAS_STATUS_SUCCESS);
        istensorcore=true;
    } else {
        istensorcore=false;
    }

    cudaStreamSynchronize(this->str);
    cudaEventRecord(this->start);
    //执行
    if (istensorcore) {
        //tensorcore执行
    status=cublasGemmEx(cubl_hand_, Matrix_OP0, Matrix_OP1, //双矩阵转置
    row0, col1, row1, //A矩阵行 B矩阵列 双矩阵匹配长度
    this->alpha, 
    this->dev_src0, CUDA_R_16F, Matrix_MDim0, //A矩阵主维度长度
    this->dev_src1, CUDA_R_16F, Matrix_MDim1, //B矩阵主维度长度
    this->beta,
    this->dev_dst, CUDA_R_32F, out_row,
    CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP); //内部计算精度与模式
    assert(status == CUBLAS_STATUS_SUCCESS);
    } else {
        //回退执行
    cudaDataType_t Gemmtype;
    switch (devtype)
    {
    case DataType::kFLOAT:
        Gemmtype=CUDA_R_32F;
        break;
    case DataType::kHALF:
        Gemmtype=CUDA_R_16F;
        break;
    }
    status=cublasGemmEx(cubl_hand_, Matrix_OP0, Matrix_OP1, //双矩阵转置
    row0, col1, row1, //A矩阵行 B矩阵列 双矩阵匹配长度
    this->alpha, 
    this->dev_src0, Gemmtype, Matrix_MDim0, //A矩阵主维度长度
    this->dev_src1, Gemmtype, Matrix_MDim1, //B矩阵主维度长度
    this->beta,
    this->dev_dst, CUDA_R_32F, out_row,
    CUDA_R_32F, CUBLAS_GEMM_DEFAULT); //内部计算精度与模式
    assert(status == CUBLAS_STATUS_SUCCESS);    
    }

    cudaStreamSynchronize(this->str);
    cudaEventRecord(this->end);

    CUDA_CHECK(cudaMemcpyAsync
    (Matrixdatasoutput_vec[0].hos_datas, this->dev_dst, out_devsize, cudaMemcpyDeviceToHost, str));
    cudaStreamSynchronize(this->str);

    float time;
    cudaEventElapsedTime(&time, this->start, this->end);

    cudaStreamAddCallback(this->str, timingcallback, &time, 0);

    //回退默认模式
    cublasSetMathMode(cubl_hand_, CUBLAS_DEFAULT_MATH);
    return status;
}
void CUDART_CB level2::blasGemmEx::timingcallback
(cudaStream_t str, cudaError_t status, void *usrdatas) noexcept
{
    std::cout<<"gemm计时 "<<*static_cast<float*>(usrdatas)<<" ms\n";
}
void level2::blasGemmEx::destroy() noexcept
{
    cudaFreeAsync(this->dev_src0, this->str);
    cudaFreeAsync(this->dev_src1, this->str);
    cudaFreeAsync(this->dev_dst, this->str);
    cudaEventDestroy(this->start);
    cudaEventDestroy(this->end);
    cudaStreamDestroy(this->str);
    this->isdestroy.store(true);
}


