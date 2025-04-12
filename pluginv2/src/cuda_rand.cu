#include "../include/cuda_rand.h"
using namespace nvinfer1;

BaseRand::BaseRand()
{
    curandCreateGenerator(&this->generator, CURAND_RNG_PSEUDO_DEFAULT);
}
BaseRand::~BaseRand()
{
    curandDestroyGenerator(this->generator);
}

//
UniformRand::UniformRand(Dims MatrixShape)
{
    this->MatrixShape=MatrixShape;
    this->nbdims=MatrixShape.nbDims;
    assert(this->nbdims != 0);
    
    this->singledim_vec.resize(this->nbdims);
        for (int i=0; i<this->singledim_vec.size(); i++)
    {
        assert(MatrixShape.d[i] != 0);
        this->singledim_vec[i]=MatrixShape.d[i];
    }
}
UniformRand::~UniformRand()
{}

void UniformRand::GenerateMatrixRand(std::vector<float>& outputmatrix)
{
    cudaStream_t rand_str;
    cudaStreamCreateWithFlags(&rand_str, cudaStreamNonBlocking);
    //绑定流
    curandSetStream(this->generator, rand_str);
    //随机种子设定
    curandSetPseudoRandomGeneratorSeed(this->generator, 1234ULL);
    
    //infos
    float *matrix_ptr;
    std::size_t matrix_size=1;
    std::int32_t matrix_elements=1;
        for (int i=0; i<this->nbdims; i++)
    {
        matrix_elements=matrix_elements * this->singledim_vec[i];
    }
    matrix_size=matrix_elements * sizeof(float);
    //
    CUDA_CHECK(cudaMallocAsync(&matrix_ptr, matrix_size, rand_str));
    CUDA_CHECK(cudaMemset(matrix_ptr, 0, matrix_size));
    //
    cudaStreamSynchronize(rand_str);
    //
    //随机生成0-1分布数
    auto status=curandGenerateUniform(this->generator, matrix_ptr, matrix_elements);
    assert(status == CURAND_STATUS_SUCCESS);
    //
    float *host_matrix_ptr;
    host_matrix_ptr=new float[matrix_elements];

    CUDA_CHECK(cudaMemcpyAsync(host_matrix_ptr, matrix_ptr, matrix_size, cudaMemcpyDeviceToHost, rand_str));
    cudaStreamSynchronize(rand_str);

    if (outputmatrix.size() < matrix_elements)
        outputmatrix.resize(matrix_elements);
    //复制到容器
    std::copy(host_matrix_ptr, host_matrix_ptr + matrix_elements, outputmatrix.begin());
    assert(outputmatrix.size() == matrix_elements);
    //
    delete[] host_matrix_ptr;
    cudaFree(matrix_ptr);
    return ;
}

//
NormalRand::NormalRand(Dims MatrixShape)
{
    this->MatrixShape=MatrixShape;
    this->nbdims=MatrixShape.nbDims;
    assert(this->nbdims != 0);
    
    this->singledim_vec.resize(this->nbdims);
        for (int i=0; i<this->singledim_vec.size(); i++)
    {
        this->singledim_vec[i]=MatrixShape.d[i];
    }
}
NormalRand::~NormalRand()
{}

void NormalRand::GenerateMatrixRand(std::vector<float>& outputmatrix)
{
    cudaStream_t rand_str;
    cudaStreamCreateWithFlags(&rand_str, cudaStreamNonBlocking);
    //绑定流
    curandSetStream(this->generator, rand_str);
    //随机种子设定
    curandSetPseudoRandomGeneratorSeed(this->generator, 1234ULL);
    
    //infos
    float *matrix_ptr;
    std::size_t matrix_size=1;
    std::int32_t matrix_elements=1;
        for (int i=0; i<this->nbdims; i++)
    {
        matrix_elements=matrix_elements * this->singledim_vec[i];
    }
    matrix_size=matrix_elements * sizeof(float);
    //
    CUDA_CHECK(cudaMallocAsync(&matrix_ptr, matrix_size, rand_str));
    CUDA_CHECK(cudaMemset(matrix_ptr, 0, matrix_size));
    //
    cudaStreamSynchronize(rand_str);
    //
    //随机正态度分布生成
    int n=matrix_elements;
    float mean=0.0f;
    float stddev=1.0f;
    auto status=curandGenerateNormal(this->generator, matrix_ptr, n, mean, stddev);
    assert(status == CURAND_STATUS_SUCCESS);
    //
    float *host_matrix_ptr;
    host_matrix_ptr=new float[matrix_elements];

    CUDA_CHECK(cudaMemcpyAsync(host_matrix_ptr, matrix_ptr, matrix_size, cudaMemcpyDeviceToHost, rand_str));
    cudaStreamSynchronize(rand_str);

    if (outputmatrix.size() < matrix_elements)
        outputmatrix.resize(matrix_elements);
    //复制到容器
    std::copy(host_matrix_ptr, host_matrix_ptr + matrix_elements, outputmatrix.begin());
    assert(outputmatrix.size() == matrix_elements);
    //
    delete[] host_matrix_ptr;
    cudaFree(matrix_ptr);
    return ;
}

//
PoissonRand::PoissonRand(Dims MatrixShape)
{
    this->MatrixShape=MatrixShape;
    this->nbdims=MatrixShape.nbDims;
    assert(this->nbdims != 0);
    
    this->singledim_vec.resize(this->nbdims);
        for (int i=0; i<this->singledim_vec.size(); i++)
    {
        this->singledim_vec[i]=MatrixShape.d[i];
    }
}
PoissonRand::~PoissonRand()
{}

void PoissonRand::GenerateMatrixRand(std::vector<float>& outputmatrix) 
{}    
void PoissonRand::GenerateMatrixRand(std::vector<uint>& outputmatrix) 
{
    cudaStream_t rand_str;
    cudaStreamCreateWithFlags(&rand_str, cudaStreamNonBlocking);
    //绑定流
    curandSetStream(this->generator, rand_str);
    //随机种子设定
    curandSetPseudoRandomGeneratorSeed(this->generator, 1234ULL);
    
    //infos
    uint *matrix_ptr;
    std::size_t matrix_size=1;
    std::int32_t matrix_elements=1;
        for (int i=0; i<this->nbdims; i++)
    {
        matrix_elements=matrix_elements * this->singledim_vec[i];
    }
    matrix_size=matrix_elements * sizeof(uint);
    //
    CUDA_CHECK(cudaMallocAsync(&matrix_ptr, matrix_size, rand_str));
    CUDA_CHECK(cudaMemset(matrix_ptr, 0, matrix_size));
    //
    cudaStreamSynchronize(rand_str);
    
    //柏松期望值
    double lambda=5.0;
    auto status=curandGeneratePoisson(this->generator, matrix_ptr, matrix_elements, lambda);
    assert(status);
    //
    uint *host_matrix_ptr;
    host_matrix_ptr=new uint[matrix_elements];

    CUDA_CHECK(cudaMemcpyAsync(host_matrix_ptr, matrix_ptr, matrix_size, cudaMemcpyDeviceToHost, rand_str));
    cudaStreamSynchronize(rand_str);

    if (outputmatrix.size() < matrix_elements)
        outputmatrix.resize(matrix_elements);
    //复制到容器
    std::copy(host_matrix_ptr, host_matrix_ptr + matrix_elements, outputmatrix.begin());
    assert(outputmatrix.size() == matrix_elements);
    //
    delete[] host_matrix_ptr;
    cudaFree(matrix_ptr);
    return ;
}

//
IntRand::IntRand(Dims MatrixShape)
{
    this->MatrixShape=MatrixShape;
    this->nbdims=MatrixShape.nbDims;
    assert(this->nbdims != 0);
    
    this->singledim_vec.resize(this->nbdims);
        for (int i=0; i<this->singledim_vec.size(); i++)
    {
        this->singledim_vec[i]=MatrixShape.d[i];
    }
}
IntRand::~IntRand()
{}

void IntRand::GenerateMatrixRand(std::vector<float>& outputmatrix) 
{} 
void IntRand::GenerateMatrixRand(std::vector<uint>& outputmatrix)
{
    cudaStream_t rand_str;
    cudaStreamCreateWithFlags(&rand_str, cudaStreamNonBlocking);
    //绑定流
    curandSetStream(this->generator, rand_str);
    //随机种子设定
    curandSetPseudoRandomGeneratorSeed(this->generator, 1234ULL);
    
    //infos
    uint *matrix_ptr;
    std::size_t matrix_size=1;
    std::int32_t matrix_elements=1;
        for (int i=0; i<this->nbdims; i++)
    {
        matrix_elements=matrix_elements * this->singledim_vec[i];
    }
    matrix_size=matrix_elements * sizeof(uint);
    //
    CUDA_CHECK(cudaMallocAsync(&matrix_ptr, matrix_size, rand_str));
    CUDA_CHECK(cudaMemset(matrix_ptr, 0, matrix_size));
    //
    cudaStreamSynchronize(rand_str);
    //
    auto status=curandGenerate(this->generator, matrix_ptr, matrix_elements);
        //
    uint *host_matrix_ptr;
    host_matrix_ptr=new uint[matrix_elements];

    CUDA_CHECK(cudaMemcpyAsync(host_matrix_ptr, matrix_ptr, matrix_size, cudaMemcpyDeviceToHost, rand_str));
    cudaStreamSynchronize(rand_str);

    if (outputmatrix.size() < matrix_elements)
        outputmatrix.resize(matrix_elements);
    //复制到容器
    std::copy(host_matrix_ptr, host_matrix_ptr + matrix_elements, outputmatrix.begin());
    assert(outputmatrix.size() == matrix_elements);
    //
    delete[] host_matrix_ptr;
    cudaFree(matrix_ptr);
    return ;
}
////
//LogNormalRand::LogNormalRand(Dims MatrixShape)
//{
//    this->MatrixShape=MatrixShape;
//    this->nbdims=MatrixShape.nbDims;
//    assert(this->nbdims != 0);
//    
//    this->singledim_vec.resize(this->nbdims);
//        for (int i=0; i<this->singledim_vec.size(); i++)
//    {
//        this->singledim_vec[i]=MatrixShape.d[i];
//    }
//}
//LogNormalRand::~LogNormalRand()
//{}







































