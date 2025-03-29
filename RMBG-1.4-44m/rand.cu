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
