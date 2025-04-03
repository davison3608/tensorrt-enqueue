#include "/codes/blas/include/cumatrix.h"
#include "/codes/blas/include/oclmatrix.h"
#include "/codes/nvnn/include/cuda_rand.h"
using namespace nvinfer1;
using namespace std;

int main(int argc, char const *argv[])
{

{
array<float, 4> datas_arr{2.2f, 2.2f, 2.2f, 2.2f};

std::vector<Matrixlevel1datas> mat_vec;
mat_vec.emplace_back(datas_arr.data(), nvinfer1::DataType::kFLOAT, datas_arr.size() * sizeof(float), 4);

std::vector<Matrixlevel1datas> mat_vec_out;
mat_vec_out.emplace_back(nullptr, nvinfer1::DataType::kFLOAT, datas_arr.size() * sizeof(float), 4);

BlasOperator BlasExec;

float alpha=0.5f;
level1::blasDscal blasop(&alpha);

BlasExec.executeV1(blasop, mat_vec, mat_vec_out);
}

{
array<float, 4> datas_arr0{2.2f, 2.2f, 2.2f, 2.2f};
array<float, 4> datas_arr1{2.2f, 2.2f, 2.2f, 2.2f};

std::vector<Matrixlevel1datas> mat_vec;
mat_vec.emplace_back(datas_arr0.data(), nvinfer1::DataType::kFLOAT, datas_arr0.size() * sizeof(float), 4);
mat_vec.emplace_back(datas_arr1.data(), nvinfer1::DataType::kFLOAT, datas_arr1.size() * sizeof(float), 4);

std::vector<Matrixlevel1datas> mat_vec_out;
mat_vec_out.emplace_back(nullptr, nvinfer1::DataType::kFLOAT, datas_arr0.size() * sizeof(float), 4);

BlasOperator BlasExec;

float alpha=0.5f;
level1::blasAxpy blasop(&alpha);

BlasExec.executeV1(blasop, mat_vec, mat_vec_out);
}

{
array<float, 4> datas_arr0{1.2f, 2.2f, 3.2f, 4.2f};
array<float, 2> datas_arr1{7.0f, 8.0f};

std::vector<Matrixlevel2datas> mat_vec;
mat_vec.emplace_back
(datas_arr0.data(), nvinfer1::DataType::kFLOAT, datas_arr0.size() * sizeof(float), Dims2{2, 2}, true);
mat_vec.emplace_back
(datas_arr1.data(), nvinfer1::DataType::kFLOAT, datas_arr1.size() * sizeof(float), Dims2{2, 1}, false);

std::vector<Matrixlevel2datas> mat_vec_out;
mat_vec_out.emplace_back
(nullptr, nvinfer1::DataType::kFLOAT, 2 * 1 * sizeof(float), Dims2{2, 1}, false);

BlasOperator BlasExec;

float alpha=1.0f;
float beta=1.0f;
level2::blasGemmEx blasop(&alpha, &beta);

BlasExec.executeV2(blasop, mat_vec, mat_vec_out);
}


 return 0;
}
