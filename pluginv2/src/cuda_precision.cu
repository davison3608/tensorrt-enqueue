#include "../include/cuda_precison.h"
using namespace nvinfer1;

void AccHalf::Accuracyhalf(float *inputs, void **outputs, int32_t nbelements)
{
    float *datas=nullptr;
    if (inputs == nullptr)
        return ;
    else
        datas=inputs;

    dim3 block(8, 8);
    int nbgrid=nbelements / block.x * block.y;
    nbgrid++;
    //
    float nbsqrt=std::sqrt(static_cast<float>(nbgrid));
    int gridx=static_cast<int>(std::ceil(nbsqrt));
    int gridy=1;
    while ((gridx * gridy) < nbgrid)
        gridy++;    
    //最佳网格
    dim3 grid(gridx, gridy);  

    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);
    
    cudaStream_t str;
    cudaStreamCreate(&str);
    cudaMallocAsync(outputs, nbelements*sizeof(half), str);

    cudaGraphNode_t node0;
    cudaMemsetParams memsetnode;
    memsetnode.dst=*outputs;
    memsetnode.elementSize=sizeof(half);
    memsetnode.value=0;
    memsetnode.height=1; //一维无需行对齐pitch
    memsetnode.width=nbelements;
    cudaGraphAddMemsetNode(&node0, graph, nullptr, 0, &memsetnode);;

    cudaGraphNode_t node1;
    cudaKernelNodeParams kernelnode;
    kernelnode.func=(void*)kernel_kflo2khal;
    kernelnode.blockDim=block;
    kernelnode.gridDim=grid;
    kernelnode.sharedMemBytes=0;
    kernelnode.extra=nullptr;
    void *params[]={datas, *outputs, &nbelements};
    kernelnode.kernelParams=params;
    cudaGraphAddKernelNode(&node1, graph, &node0, 1, &kernelnode);
    //外部释放原有的资源
    //cudaGraphAddMemFreeNode(&node2, graph, &node1, 1, inputs);
    
    //cudaGraphAddDependencies();
    cudaGraphExec_t graphex;
    cudaGraphInstantiate(&graphex, graph, nullptr, nullptr, 0);

    //启动
    CUDA_CHECK(cudaGraphLaunch(graphex, str));
    cudaStreamSynchronize(str);

    cudaStreamDestroy(str);
    cudaGraphDestroyNode(node0);
    cudaGraphDestroyNode(node1);
    cudaGraphExecDestroy(graphex);
    cudaGraphDestroy(graph);
}

void AccHalf::Accuracyhalfdeser(half *inputs, void **outputs, int32_t nbelements)
{

}

__global__ void kernel_kflo2khal(void *inputs, void *outputs, int32_t nbelements)
{
    int x=blockIdx.x * blockDim.x + threadIdx.x;
    int y=blockIdx.y * blockDim.y + threadIdx.y;
    int idx=y * (gridDim.x * blockDim.x) + x;
    if (idx >= nbelements)
        return ;

    __syncthreads();
    //压缩精度
    auto src=static_cast<float*>(inputs);
    auto value=static_cast<half*>(outputs);
    
    value[idx]=__float2half_rn(src[idx]);

    __threadfence();
    return ;
}

__global__ void kernel_kflo2khaldeser(void *inputs, void *outputs, int32_t nbelements)
{
    int x=blockIdx.x * blockDim.x + threadIdx.x;
    int y=blockIdx.y * blockDim.y + threadIdx.y;
    int idx=y * (gridDim.x * blockDim.x) + x;
    if (idx >= nbelements)
        return ;

    __syncthreads();
    //恢复精度
    auto src=static_cast<half*>(inputs);
    auto value=static_cast<float*>(outputs);
    
    value[idx]=__float2half_rn(src[idx]);

    __threadfence();
    return ;
}

__global__ void kernel_dataprint(void *ptr, int size)
{
    int x=blockIdx.x * blockDim.x + threadIdx.x;
    int y=blockIdx.y * blockDim.y + threadIdx.y;
    int idx=y * (gridDim.x * blockDim.x) + x;
    float *f_ptr=(float*)ptr;
    //
    __syncthreads();
    //
    if (idx >= size/2)
        return ;
    else {
        printf("%f ", f_ptr[idx]);
        return ;
    }
}

template void cudaMemcpydataShow<int>(std::vector<int> vec_datas);
template void cudaMemcpydataShow<float>(std::vector<float> vec_datas);
template void cudaMemcpydataShow<double>(std::vector<double> vec_datas);
template <typename type>
void cudaMemcpydataShow(std::vector<type> vec_datas)
{
    assert(!vec_datas.empty());
    auto host_ptr=vec_datas.data();
    std::size_t mem_size=vec_datas.size();
    //
    cudaStream_t mem_str;
    cudaEvent_t start;
    cudaEvent_t end;
    cudaStreamCreateWithFlags(&mem_str, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&start, cudaEventDefault);
    cudaEventCreateWithFlags(&end, cudaEventDefault);
    //
    type *buffer;
    cudaMallocAsync(&buffer, mem_size * sizeof(type), mem_str);
    cudaMemsetAsync(buffer, 0, mem_size * sizeof(type), mem_str);
    CUDA_CHECK(
    cudaMemcpyAsync(buffer, host_ptr, mem_size * sizeof(type), cudaMemcpyKind::cudaMemcpyHostToDevice, mem_str));
    //
    dim3 grid(4, 4);
    dim3 block(16, 16);
    //
    cudaStreamSynchronize(mem_str);
    cudaEventRecord(start, mem_str);
    kernel_dataprint<<<grid, block, 0, mem_str>>>(buffer, mem_size);
    //
    cudaStreamSynchronize(mem_str);
    cudaEventRecord(end, mem_str);

    auto time=std::make_unique<float>(0);
    cudaEventElapsedTime(time.get(), start, end);
    std::cerr<<"\n核计时---"<<*time<<"---ms\n";

    cudaStreamDestroy(mem_str);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(buffer);
}

