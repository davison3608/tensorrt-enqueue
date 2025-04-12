#ifndef CUDA_RAND
#define CUDA_RAND
#include "/codes/cv/include/core.hpp"

class BaseRand
{
private:
protected:
    /*随机生成器*/
    curandGenerator_t generator;
    /*矩阵形状描述*/
    nvinfer1::Dims MatrixShape;
    /*维度数量*/
    int nbdims;
    /*各维度值*/
    std::vector<int> singledim_vec;    
public:
    explicit BaseRand();
    ~BaseRand();

    virtual void GenerateMatrixRand(std::vector<float>& outputmatrix) {
        //默认实现 抛出异常或未实现
        throw std::runtime_error("GenerateMatrixRand not implemented");
    }
};

/*0-1随机分布*/
class UniformRand: public BaseRand
{
private:
public:
    explicit UniformRand(nvinfer1::Dims MatrixShape);
    ~UniformRand();

    void GenerateMatrixRand(std::vector<float>& outputmatrix) 
    override;
};

/*正态分布*/
class NormalRand: public BaseRand
{
private:
public:
    explicit NormalRand(nvinfer1::Dims MatrixShape);
    ~NormalRand();

    void GenerateMatrixRand(std::vector<float>& outputmatrix) 
    override;
};

/*泊松分布*/
class PoissonRand: public BaseRand
{
private:
public:
    explicit PoissonRand(nvinfer1::Dims MatrixShape);
    ~PoissonRand();

    void GenerateMatrixRand(std::vector<float>& outputmatrix) 
    override;

    void GenerateMatrixRand(std::vector<uint>& outputmatrix);
};

/*随机整数分布*/
class IntRand: public BaseRand
{
private:
public:
    explicit IntRand(nvinfer1::Dims MatrixShape);
    ~IntRand();

    void GenerateMatrixRand(std::vector<float>& outputmatrix) 
    override;

    void GenerateMatrixRand(std::vector<uint>& outputmatrix);
};

///*对数分布*/
//class LogNormalRand: public BaseRand
//{
//private:
//public:
//    explicit LogNormalRand(nvinfer1::Dims MatrixShape);
//    ~LogNormalRand();
//
//    void GenerateMatrixRand(std::vector<float>& outputmatrix) 
//    override;
//};
//
#endif