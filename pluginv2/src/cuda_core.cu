#include "../include/cuda_core.h"
using namespace nvinfer1;

__global__ static void kernel_leakrelu_kf32
(const void* inputs, void *outputs, int32_t nbelements, void *workspace, float *alpha_kfloat)
{
    int x=blockIdx.x * blockDim.x + threadIdx.x;
    int y=blockIdx.y * blockDim.y + threadIdx.y;
    int idx=y * (gridDim.x * blockDim.x) + x;
    if (idx >= nbelements)
        return ;
    
    __syncthreads();
    const float *src=static_cast<const float*>(inputs);
    float *value=static_cast<float*>(outputs);

    auto input=src[idx];
    auto output=(input > 0)? input:(*alpha_kfloat * input);
    value[idx]=output;
    
    __threadfence();
    return ;
}

__global__ static void kernel_leakrelu_khal
(const void* inputs, void *outputs, int32_t nbelements, void *workspace, half *alpha_khalf)
{
    int x=blockIdx.x * blockDim.x * threadIdx.x;
    int y=blockIdx.y * blockDim.y + threadIdx.y;
    int idx=y * (gridDim.x * blockDim.x) + x;
    if (idx > nbelements)
        return ;

    __syncthreads();
    const half *src=static_cast<const half*>(inputs);
    float *value=static_cast<float*>(outputs);

    half input=src[idx];
    half output;
    if (__half2float(input) > 0)
        output=input;
    else
        output=__hmul_rn(*alpha_khalf, input);
        
    value[idx]=__half2float(output);
    __threadfence();
    return ;
}

const std::vector<float> WeightswithBiasloading(const char *bin_path)
{
    std::ifstream infile;
    infile.open(bin_path, std::ios::binary);
    assert(infile.is_open());
    assert(infile.good());
    //字节大小
    infile.seekg(0, std::ios::end);
    std::size_t filesize=infile.tellg();
    assert(filesize > 0);
    infile.seekg(0, std::ios::beg);
    //根据字节大小预留容量
    std::vector<float> data_vec;
    data_vec.reserve(filesize / sizeof(float));

    infile.read(reinterpret_cast<char*>(data_vec.data()), filesize);
    if (data_vec.size() == (filesize/sizeof(float)))
        return data_vec;
    else {
        std::cerr<<"读取长度不一致\n";
        exit(false);
    }
}

PluginField CreatePluginField
(const std::vector<float> &data_vec, const char *name, PluginFieldType type)
{
    PluginField pf;
    pf.name=name;
    pf.data=data_vec.data();
    pf.length=data_vec.size();
    //浅拷贝 注意参数生命周期
    assert(pf.name);
    assert(pf.length > 0);
    assert(pf.data);
    //
    switch (type)
    {
    case (PluginFieldType::kFLOAT32):
        pf.type=type;
        break;
    case (PluginFieldType::kFLOAT16):
        pf.type=type;
        break;
    case (PluginFieldType::kINT8):
        pf.type=type;
        break;
    default:
        std::cerr<<"不支持的权重精度\n";  
        break;
    }
    return pf;
}
//
/***********************静态基算子创建器***********************/
//
BaseCreator::BaseCreator()
{
    //不显式命名空间 则静态字符串地址getPluginNamespace传递到creatplugin
    //传递到BasePluign的IPlunamespace
    //在clone创建layer时被克隆对象指向BasePluign的IPlunamespace
    this->IPluCrnamespace="";
}
BaseCreator::~BaseCreator()
{}
//
int32_t BaseCreator::getTensorRTVersion() const noexcept
{
    return (NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + NV_TENSORRT_PATCH;
}

void BaseCreator::setPluginNamespace(AsciiChar const* pluginNamespace) noexcept
{
    std::cerr<<"不要设置命名空间\n";
    assert(pluginNamespace == "");    
}

AsciiChar const* BaseCreator::getPluginNamespace() const noexcept
{
    return this->IPluCrnamespace;
}
//
/***********************静态基算子***********************/
//
BasePluign::BasePluign()
{
    this->IPlunamespace="";
}
BasePluign::~BasePluign()
{}
//
int32_t BasePluign::getTensorRTVersion() const noexcept
{
    return (NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + NV_TENSORRT_PATCH;
}

void BasePluign::setPluginNamespace(AsciiChar const* pluginNamespace) noexcept
{
    std::cerr<<"不要设置命名空间\n";
    assert(pluginNamespace == "");  
} 

AsciiChar const* BasePluign::getPluginNamespace() const noexcept 
{
    return this->IPlunamespace;
}
//
bool BasePluign::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    if ((type == DataType::kFLOAT ||
        type == DataType::kHALF ||
        type == DataType::kINT8) && 
        (format == TensorFormat::kLINEAR ||
        format == TensorFormat::kDLA_LINEAR))
        return true;
    else 
        return false;
}

//
/***********************算子管理***********************/
//
OperManger::OperManger()
{}
OperManger::~OperManger()
{
    for (auto ptr: OpName_lt)
        delete[] ptr;
    for (auto ptr: OpVersion_lt)
        delete[] ptr;    
}

void OperManger::OperatorregistredShow() const noexcept
{
    assert(OpName_lt.size() == OpVersion_lt.size());
    auto iter_0=OpName_lt.begin();
    auto iter_1=OpVersion_lt.begin();
        while((iter_0 != OpName_lt.end()) && (iter_1 != OpVersion_lt.end()))
    {
        std::cout<<*iter_0<<"  ";
        std::cout<<*iter_1<<"\n";
        iter_0++;
        iter_1++;
    }
}

void OperManger::isOperatorregistred(const char* opname, const char* opversion, const char* opnamespace)
{
    //获取注册表
    IPluginRegistry* iregistr=getPluginRegistry();
    
    IPluginCreator* opcr=iregistr->getPluginCreator(opname, opversion, opnamespace);
    assert(opcr != nullptr);
    
    const char* name=new char[24];
    const char* version=new char[4];
    std::memcpy(const_cast<char*>(name), opname, ssd::strlen(opname));
    std::memcpy(const_cast<char*>(version), opversion, ssd::strlen(opversion));
    
    //纳入list
    OpName_lt.push_back(name);
    OpVersion_lt.push_back(version);
    std::cout<<OpName_lt.back()<<" "<<OpVersion_lt.back()<<"\n";
}
//
/***********************自定义算子集合***********************/
//
/************leakrelu激活函数************/
leakrelu::leakrelu(float &alpha)
{
    //标识符构造
    onlyname="ReluLeak";
    onlyversion="1.0";

    //参数加载
    alpha=alpha;
}
leakrelu::leakrelu(const char* name, const void* serialsizedata, std::size_t serialsizelength)
{
    //字节单位
    const char* load_ptr=static_cast<const char*>(serialsizedata);
    const char* start=static_cast<const char*>(serialsizedata);

    //命名空间在父类指向

    //反序列化插件标识
    this->onlyname="LeakRelu";
    this->onlyversion="1.0";

    //最大批量
    std::memcpy(&mmaxbatchsize, load_ptr, sizeof(int32_t));
    load_ptr+=sizeof(int32_t);

    //序列化输入维度结构
    std::memcpy(&mnbInputs, load_ptr, sizeof(int32_t));
    load_ptr+=sizeof(int32_t);
    std::memcpy(&minputDims.nbDims, load_ptr, sizeof(int32_t));
    load_ptr+=sizeof(int32_t);
        for (int idx=0; idx<minputDims.nbDims; idx++)
    {
        std::memcpy(&minputDims.d[idx], load_ptr, sizeof(int32_t));
        load_ptr+=sizeof(int32_t);
    }

    //序列化输出维度结构
    std::memcpy(&mnbOutputs, load_ptr, sizeof(int32_t));
    load_ptr+=sizeof(int32_t);
    std::memcpy(&moutputDims.nbDims, load_ptr, sizeof(int32_t));
    load_ptr+=sizeof(int32_t);
        for (int idx=0; idx<moutputDims.nbDims; idx++)
    {
        std::memcpy(&moutputDims.d[idx], load_ptr, sizeof(int32_t));
        load_ptr+=sizeof(int32_t);
    }

    //精度与数据格式参数
    std::memcpy(&mtype, load_ptr, sizeof(DataType));
    load_ptr+=sizeof(DataType);
    std::memcpy(&mprecision, load_ptr, sizeof(DataType));
    load_ptr+=sizeof(DataType);
    std::memcpy(&mFormat, load_ptr, sizeof(TensorFormat));
    load_ptr+=sizeof(TensorFormat);

    //函数参数
    std::memcpy(&alpha, load_ptr, sizeof(float));
    load_ptr+=sizeof(float);

    assert((load_ptr - start) == serialsizelength);
    //
    //无须序列化变量重新计算
    this->nbelements=1;
    for (int idx=0; idx<this->minputDims.nbDims; idx++)
        this->nbelements=this->nbelements * this->minputDims.d[idx];
}
leakrelu::leakrelu(const IPluginV2 &IPlu)
{
    //检查为leak算子
    const leakrelu* leakplugin=dynamic_cast<const leakrelu*>(&IPlu);
    assert(leakplugin);

    //命名空间在父类指向
  
    //标识符构造
    this->onlyname="LeakRelu";
    this->onlyversion="1.0";

    //最大批量
    this->mmaxbatchsize=leakplugin->mmaxbatchsize;

    //已序列化输入维度结构
    this->mnbInputs=leakplugin->mnbInputs;
    this->minputDims=leakplugin->minputDims;

    //已序列化输出维度结构
    this->mnbOutputs=leakplugin->mnbOutputs;
    this->moutputDims=leakplugin->moutputDims;

    //精度与数据格式参数
    this->mtype=leakplugin->mtype;
    this->mprecision=leakplugin->mprecision;
    this->mFormat=leakplugin->mFormat;

    //函数参数
    this->alpha=leakplugin->alpha;
    //
    //无须序列化变量
    this->nbelements=leakplugin->nbelements;

    //推理期便量无需拷贝
}
leakrelu::~leakrelu()
{}
//
//实现
AsciiChar const* leakrelu::getPluginType() const noexcept
{
    return onlyname;
}
//
AsciiChar const* leakrelu::getPluginVersion() const noexcept
{
    return onlyversion;
}
//
bool leakrelu::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    bool inputSupported=(type == NVfloat || type == NVhalf);
    
    bool formatSupported=(format == PluginFormat::kLINEAR);
    
    return inputSupported && formatSupported;
}
//
int32_t leakrelu::getNbOutputs() const noexcept
{
    return int32_t(1);   
}
//
Dims leakrelu::getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept 
{
    if (nbInputDims>1) {
        std::cerr<<"不支持多输入\n";
        exit(false);
    }
    if (nbInputDims <= 0 || index >= nbInputDims)
        return Dims{};
    else if (nbInputDims != 1)
        return Dims{};
    else if (index != 0)
        return Dims{};

    //leakrelu不改变张量形状
    Dims outputDims;
    outputDims.nbDims=inputs[index].nbDims;
    //
        for (int idx=0; idx<outputDims.nbDims; idx++)
    {
        outputDims.d[idx]=inputs[index].d[idx];
    }
    return outputDims;
}
//
void leakrelu::configureWithFormat
(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs, DataType type, PluginFormat format, int32_t maxBatchSize) noexcept
{
    if (nbInputs != nbOutputs) {
        std::cerr<<"输入输出张量数目不匹配\n";
        exit(false);
    } else if (!supportsFormat(type, format)) {
        std::cerr<<"设置精度或数据类型不支持\n";
        exit(false);
    } else if (nbInputs > 1) {
        std::cerr<<"leakrelu必须为单输入单输出层\n";
        exit(false);
    } else if (inputDims[0].nbDims < 1) {
        std::cerr<<"输入张量维度必须存在\n";
        exit(false);
    } else if (maxBatchSize <= 0) {
        std::cerr<<"张量批量不为0\n";
        exit(false);
    }
    //
    this->nbelements=1;
        for (int idx=0; idx<inputDims[0].nbDims; idx++)
    {
        assert(inputDims[0].d[idx] > 0);
        if (inputDims[0].d[idx] != outputDims[0].d[idx])
        {
            std::cerr<<"输入输出张量形状不匹配\n";
            exit(false);
        }
        nbelements=nbelements * inputDims[0].d[idx];
    }
    this->mmaxbatchsize=maxBatchSize;
    //张量形状信息
    this->minputDims=inputDims[0];
    this->moutputDims=outputDims[0];
    //输入输出Dims数量
    this->mnbInputs=nbInputs;
    this->mnbOutputs=nbOutputs;
    //精度与数据格式信息
    this->mtype=type;
    this->mFormat=format;
        //计算精度与输入保持一致
    switch (this->mtype)
    {
    case NVfloat:
    this->mprecision=NVfloat;
        break;
    case NVhalf:
    this->mprecision=NVfloat;
        break;
    default:
        exit(false);
    }
}
//
std::size_t leakrelu::getWorkspaceSize(int32_t maxBatchSize) const noexcept
{
    //无需额外workspace
    return 0;
}

int32_t leakrelu::initialize() noexcept 
{
    CUDA_CHECK(cudaMalloc(&buffer, sizeof(float)));

    CUDA_CHECK(cudaMemset(this->buffer, 0, sizeof(float)));
    // 将 alpha 参数拷贝到显存
    cudaError_t status=cudaMemcpy(this->buffer, &alpha, sizeof(float), cudaMemcpyHostToDevice);
    assert(status == cudaSuccess);
    return status;
}

int32_t leakrelu::enqueue
(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    auto channels=this->minputDims.d[0];
    auto rows=this->minputDims.d[1];
    auto cols=this->minputDims.d[2];
    int32_t enq_nbelements=batchSize * channels * rows * cols;
    
    dim3 block(8, 8);
    int nbgrid=enq_nbelements / block.x * block.y;
    nbgrid++;
    //
    float nbsqrt=std::sqrt(static_cast<float>(nbgrid));
    int gridx=static_cast<int>(std::ceil(nbsqrt));
    int gridy=1;
    while ((gridx * gridy) < nbgrid)
        gridy++;    
    //最佳网格
    dim3 grid(gridx, gridy);
    assert((grid.x*grid.y)*(block.x*block.y) > nbelements);    

    //单输入输出层
    const float* kernel_input=static_cast<const float*>(inputs[0]);
    float* kernel_output=static_cast<float*>(outputs[0]);
    //计算精度与张量精度一致
    assert(this->mtype == this->mprecision);

    if (this->mtype == NVfloat) 
    {
    //启动
    kernel_leakrelu_kf32<<<grid, block, 0, stream>>>
    (kernel_input, kernel_output, enq_nbelements, workspace, this->buffer);

    cudaError_t status=cudaStreamSynchronize(stream);
    assert(status == cudaSuccess);
    return status;    
    } 
    else if (this->mtype == NVhalf) 
    {
    float *kernel_input_kflo=const_cast<float*>(kernel_input);
    void *kernel_input_khal;
    void *buffer_khal;

    //压缩权重
    AccHalf::Accuracyhalf(this->buffer, &buffer_khal, 1);
    //压缩张量
    AccHalf::Accuracyhalf(kernel_input_kflo, &kernel_input_khal, enq_nbelements);

    //启动
    kernel_leakrelu_khal<<<grid, block, 0, stream>>>
    (kernel_input_khal, kernel_output, enq_nbelements, workspace, static_cast<half*>(buffer_khal));
    
    cudaError_t status=cudaStreamSynchronize(stream);
    assert(status == cudaSuccess);

    //释放临时转换资源
    cudaFreeAsync(kernel_input_khal, stream);
    cudaFreeAsync(buffer_khal, stream);
    return status;  
    } 
    else 
    {
    std::cerr<<"无有效精度信息\n";
    exit(false);    
    }
}

void leakrelu::terminate() noexcept
{
    CUDA_CHECK(cudaFree(buffer));
}

std::size_t leakrelu::getSerializationSize() const noexcept 
{
    std::size_t serializesize=0;
    
    // 命名空间
    // 序列化插件标识信息
    // 指向静态空间无需序列化

    // 最大批量
    serializesize+=sizeof(int32_t);

    // dims数量 输入维度结构 
    serializesize+=sizeof(int32_t); 
    serializesize+=sizeof(int32_t); //nbdims
    serializesize+=minputDims.nbDims * sizeof(int32_t); // 各维度值

    // dims数量 输出维度结构
    serializesize+=sizeof(int32_t); 
    serializesize+=sizeof(int32_t); //nbdims
    serializesize+=moutputDims.nbDims * sizeof(int32_t); // 各维度值

    // 基础类型参数
    serializesize+=sizeof(DataType); 
    serializesize+=sizeof(DataType); 
    serializesize+=sizeof(TensorFormat);
    
    // 自定义参数
    serializesize+=sizeof(float);

    return serializesize;
}

void leakrelu::serialize(void* buffer) const noexcept 
{
    //字节单位
    char* raw_ptr=static_cast<char*>(buffer);
    char* start=static_cast<char*>(buffer);

    //命名空间
    //序列化插件标识信息
    //指向静态空间无需序列化

    //最大批量
    std::memcpy(raw_ptr, &mmaxbatchsize, sizeof(int32_t));
    raw_ptr+=sizeof(int32_t);

    //序列化输入维度结构
    std::memcpy(raw_ptr, &mnbInputs, sizeof(int32_t));
    raw_ptr+=sizeof(int32_t);
    std::memcpy(raw_ptr, &minputDims.nbDims, sizeof(int32_t));
    raw_ptr+=sizeof(int32_t);
        for (int idx=0; idx<minputDims.nbDims; idx++)
    {
        int32_t d=minputDims.d[idx];
        std::memcpy(raw_ptr, &d, sizeof(int32_t));
        raw_ptr+=sizeof(int32_t);
    }

    //序列化输出维度结构
    std::memcpy(raw_ptr, &mnbOutputs, sizeof(int32_t));
    raw_ptr+=sizeof(int32_t);    
    std::memcpy(raw_ptr, &moutputDims.nbDims, sizeof(int32_t));
    raw_ptr+=sizeof(int32_t);    
        for (int idx=0; idx<moutputDims.nbDims; idx++)
    {
        int32_t d=moutputDims.d[idx];
        std::memcpy(raw_ptr, &d, sizeof(int32_t));
        raw_ptr+=sizeof(int32_t);
    }

    //精度与数据格式参数
    std::memcpy(raw_ptr, &mtype, sizeof(DataType));
    raw_ptr+=sizeof(DataType);
    std::memcpy(raw_ptr, &mprecision, sizeof(DataType));
    raw_ptr+=sizeof(DataType);
    std::memcpy(raw_ptr, &mFormat, sizeof(TensorFormat));
    raw_ptr+=sizeof(TensorFormat);
    
    //自定义参数
    std::memcpy(raw_ptr, &alpha, sizeof(float));
    raw_ptr+=sizeof(float);

    assert((raw_ptr - start) == getSerializationSize());
}

IPluginV2* leakrelu::clone() const noexcept 
{
    return new leakrelu(*this);
}

void leakrelu::destroy() noexcept
{
    delete this;
}
/************leakrelu创建器************/
AsciiChar const* leakrelucreator::getPluginName() const noexcept
{
    const char* name="LeakRelu";
    return name;
}
AsciiChar const* leakrelucreator::getPluginVersion() const noexcept
{
    const char* version="1.0";
    return version;
}

PluginFieldCollection const* leakrelucreator::getFieldNames() noexcept
{
    //静态属性避免被析构
    static std::vector<PluginField> field_vec;
    static PluginFieldCollection fieldcoll{int32_t(0), nullptr};

    if (field_vec.empty()) {
        //alpha参数
        field_vec.emplace_back("alpha", nullptr, PluginFieldType::kFLOAT32, int32_t(1));
    }

    //汇总
    fieldcoll.fields=field_vec.data();
    fieldcoll.nbFields=field_vec.size();

    return &fieldcoll;
}

IPluginV2* leakrelucreator::createPlugin(AsciiChar const* name, PluginFieldCollection const* fc) noexcept 
{
    //超参数局部设置
    float alpha_kflo=0.01f;

    if (fc->nbFields != 1) {
        std::cerr<<"pf对象仅需要1\n";
        exit(false);
    }

    //读取
    if (fc != nullptr) {
            for (int i = 0; i<fc->nbFields; ++i) 
        {
            auto &field=fc->fields[i];
            //验证参数名称 类型 元素个数
            assert(!ssd::strcmp(field.name, "alpha"));
            assert(field.type == PluginFieldType::kFLOAT32);
            assert(field.length == int32_t(1));

            alpha_kflo=*(static_cast<const float*>(field.data));
        }
    }
    //
    leakrelu* plugin=new leakrelu(alpha_kflo);
    //命名空间在父类指向
    
    return plugin;
}

IPluginV2* leakrelucreator::deserializePlugin(AsciiChar const* name, void const* serialData, std::size_t serialLength) noexcept 
{
    assert(serialData != nullptr);
    assert(serialLength > 0);

    leakrelu* plugin=new leakrelu(name, serialData, serialLength);
    return plugin;
}

/************一维度卷积全连接************/
Conv1d::Conv1d(nvinfer1::PluginField &weights, nvinfer1::PluginField &bias)
{
    this->onlyname="Conv1d";
    this->onlyversion="1.0";
        //核权重
    if (weights.type != PluginFieldType::kFLOAT32) 
    {
    std::cerr<<"不支持的精度\n";
    exit(false);    
    } 
    else 
    {
    this->weights_type=weights.type;
    this->weights_length=weights.length;
    //
    this->weights_ptr=new float[weights.length];
    std::memcpy(this->weights_ptr, const_cast<const void*>(weights.data), sizeof(float) * weights.length);
    } 
        //偏置权重
    if (bias.type != PluginFieldType::kFLOAT32) 
    {
    std::cerr<<"不支持的精度\n";
    exit(false);    
    } 
    else
    {
    this->bias_type=bias.type;
    this->bias_length=bias.length;
    //
    this->bias_ptr=new float[bias.length];
    std::memcpy(this->bias_ptr, const_cast<const void*>(bias.data), sizeof(float) * bias.length);
    }
}
Conv1d::Conv1d(const char *name, const void* serialsizedata, std::size_t serialsizelength)
{
    this->onlyname="Conv1d";
    this->onlyversion="1.0";
    const char* raw_ptr=static_cast<const char*>(serialsizedata); // char单位
    const char* start=static_cast<const char*>(serialsizedata); 
        // 张量精度与格式
    std::memcpy(&this->mtype, raw_ptr, sizeof(DataType));
    raw_ptr += sizeof(DataType);
    std::memcpy(&this->mprecision, raw_ptr, sizeof(DataType));
    raw_ptr += sizeof(DataType);
    std::memcpy(&this->mformat, raw_ptr, sizeof(TensorFormat));
    raw_ptr += sizeof(TensorFormat);
        // 最大批量
    std::memcpy(&this->maxbatch, raw_ptr, sizeof(uint));
    raw_ptr += sizeof(uint);
        // 输入数量及维度
    std::memcpy(&this->mnbinputs, raw_ptr, sizeof(uint));
    raw_ptr += sizeof(uint);
    std::memcpy(&this->minputdims.nbDims, raw_ptr, sizeof(int32_t));
    raw_ptr += sizeof(int32_t);
    for (int i=0; i<this->minputdims.nbDims; ++i) {
        std::memcpy(&this->minputdims.d[i], raw_ptr, sizeof(int32_t));
        raw_ptr += sizeof(int32_t);
    }
        // 输出数量及维度
    std::memcpy(&this->mnboutputs, raw_ptr, sizeof(uint));
    raw_ptr += sizeof(uint);
    std::memcpy(&this->moutputdims.nbDims, raw_ptr, sizeof(int32_t));
    raw_ptr += sizeof(int32_t);
    for (int i=0; i<this->moutputdims.nbDims; ++i) {
        std::memcpy(&this->moutputdims.d[i], raw_ptr, sizeof(int32_t));
        raw_ptr += sizeof(int32_t);
    }

        // 核权重反序列化
    std::memcpy(&this->weights_type, raw_ptr, sizeof(PluginFieldType));
    raw_ptr += sizeof(PluginFieldType);
    std::memcpy(&this->weights_length, raw_ptr, sizeof(int32_t));
    raw_ptr += sizeof(int32_t);
        // 
        switch (this->mprecision)
    {
    case NVfloat:
    this->weights_ptr = new float[this->weights_length];
    std::memcpy(this->weights_ptr, raw_ptr, this->weights_length * sizeof(float));
    raw_ptr += this->weights_length * sizeof(float);
        break;
    case NVhalf:
    this->weights_ptr = new half[this->weights_length];
    std::memcpy(this->weights_ptr, raw_ptr, this->weights_length * sizeof(half));
    raw_ptr += this->weights_length * sizeof(half);
        break;
    default:
        exit(false);
    } 
        // 偏置权重反序列化
    std::memcpy(&this->bias_type, raw_ptr, sizeof(PluginFieldType));
    raw_ptr += sizeof(PluginFieldType);
    std::memcpy(&this->bias_length, raw_ptr, sizeof(int32_t));
    raw_ptr += sizeof(int32_t);
        //
        switch (this->mprecision)
    {
    case NVfloat:
    this->bias_ptr = new float[this->bias_length];
    std::memcpy(this->bias_ptr, raw_ptr, this->bias_length * sizeof(float));
    raw_ptr += this->bias_length * sizeof(float);
        break;
    case NVhalf:
    this->bias_ptr = new half[this->bias_length];
    std::memcpy(this->bias_ptr, raw_ptr, this->bias_length * sizeof(half));
    raw_ptr += this->bias_length * sizeof(half);
        break;
    default:
        exit(false);
    } 

    assert((raw_ptr-start) == serialsizelength);
}
Conv1d::Conv1d(const nvinfer1::IPluginV2 &IPlu)
{
    this->onlyname="Conv1d";
    this->onlyversion="1.0";
    const Conv1d *conv=dynamic_cast<const Conv1d*>(&IPlu);
    assert(conv);
        // 张量精度与格式
    this->mtype=conv->mtype;
    this->mprecision=conv->mprecision;
    this->mformat=conv->mformat;
        // 最大批量
    this->maxbatch=conv->maxbatch;
        // 输入数量及维度
    this->mnbinputs=conv->mnbinputs;
    this->minputdims=conv->minputdims;
        // 输出数量及维度
    this->mnboutputs=conv->mnboutputs;
    this->moutputdims=conv->moutputdims;

        // 核权重
    this->weights_type=conv->weights_type;
    this->weights_length=conv->weights_length;
        switch (this->mprecision)
    {
    case NVfloat:
    this->weights_ptr = new float[this->weights_length];
    std::memcpy(this->weights_ptr, conv->weights_ptr, this->weights_length * sizeof(float));
        break;
    case NVhalf:
    this->weights_ptr = new half[this->weights_length];
    std::memcpy(this->weights_ptr, conv->weights_ptr, this->weights_length * sizeof(half));
        break;
    default:
        exit(false);
    } 
        // 偏置权重
    this->bias_type=conv->bias_type;
    this->bias_length=conv->bias_length;
        switch (this->mprecision)
    {
    case NVfloat:
    this->bias_ptr = new float[this->bias_length];
    std::memcpy(this->bias_ptr, conv->bias_ptr, this->bias_length * sizeof(float));
        break;
    case NVhalf:
    this->bias_ptr = new half[this->bias_length];
    std::memcpy(this->bias_ptr, conv->bias_ptr, this->bias_length * sizeof(half));
        break;
    default:
        exit(false);
    }  
}
Conv1d::~Conv1d()
{
    if (this->weights_ptr != nullptr) {
        delete[] this->weights_ptr;
        this->weights_ptr=nullptr;
    } else if (this->bias_ptr != nullptr) {
        delete[] this->bias_ptr;
        this->bias_ptr=nullptr;
    }
}
//
AsciiChar const* Conv1d::getPluginType() const noexcept 
{
    return this->onlyname;
}
AsciiChar const* Conv1d::getPluginVersion() const noexcept 
{
    return this->onlyversion;
}

bool Conv1d::supportsFormat(DataType type, PluginFormat format) const noexcept 
{
    bool inputSupported=(type == NVfloat || type == NVhalf);
    
    bool formatSupported=(format == PluginFormat::kLINEAR);
    
    return inputSupported && formatSupported;
}

int32_t Conv1d::getNbOutputs() const noexcept
{
    return int32_t(1);
}
Dims Conv1d::getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept 
{
    if (nbInputDims>1) {
        std::cerr<<"不支持多输入\n";
        exit(false);
    } else if (inputs[index].nbDims != 3) {
        std::cerr<<"一维卷积仅支持dims3输入\n";
        exit(false);
    } else if (inputs[index].d[1] != 1) {
        std::cerr<<"输入张量第1维度必须为1\n";
        exit(false);
    } else if(nbInputDims <= 0) {
        return Dims{};
    } else if (index >= nbInputDims) {
        return Dims{};
    }
    // 
    int32_t sequence=inputs[index].d[2];
    int32_t in_channels=inputs[index].d[0];
    assert((sequence>0) && (in_channels>0));

    //计算padding保持步长不变
    int32_t padding=(this->kernelsize - 1) / 2;

    assert(((sequence-this->kernelsize) + (2*padding)) % this->stride == 0);
    int32_t out_sequence=((sequence-this->kernelsize) + (2*padding)) / this->stride;
    out_sequence++;

    //输入输出通道保持一致 CHW格式
    Dims3 out_dims3{int32_t(in_channels), 1, int32_t(out_sequence)};
    return out_dims3;
}

void Conv1d::configureWithFormat(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs, DataType type, PluginFormat format, int32_t maxBatchSize) noexcept 
{
    if (nbInputs != nbOutputs) {
        std::cerr<<"输入输出张量数目不匹配\n";
        exit(false);
    } else if (!supportsFormat(type, format)) {
        std::cerr<<"设置精度或数据类型不支持\n";
        exit(false);
    } else if (nbInputs > 1) {
        std::cerr<<"conv1d必须为单输入单输出层\n";
        exit(false);
    } else if (inputDims[0].nbDims != 3) {
        std::cerr<<"输入张量维度必须为3\n";
        exit(false);
    } else if (inputDims[0].d[1] != 1) {
        std::cerr<<"输入张量第1维度必须为1\n";
        exit(false);
    } else if (inputDims[0].d[0] != outputDims[0].d[0]) {
        std::cerr<<"输入输出通道保持一致\n";
        exit(false);
    } else if (maxBatchSize <= 0) {
        std::cerr<<"张量批量不为0或必须显式指定\n";
        exit(false);
    }
    //精度与数据格式信息
    this->mtype=type;
    //计算精度与输入张量保持一致
    this->mprecision=this->mtype;
    this->mformat=format;
    //静态批量
    this->maxbatch=maxBatchSize;
    //输入
    this->mnbinputs=nbInputs;
    this->minputdims.nbDims=3;
        for (int idx=0; idx<this->minputdims.nbDims; idx++)
    {
        minputdims.d[idx]=inputDims->d[idx];
    }
    //输出
    this->mnboutputs=nbOutputs;
    this->moutputdims.nbDims=3;
        for (int idx=0; idx<this->moutputdims.nbDims; idx++)
    {
        moutputdims.d[idx]=outputDims->d[idx];
    }

    //压缩核精度
    if (this->mprecision == NVfloat)
    {} 
    else if (this->mprecision == NVhalf) 
    {
        float *dev_p;
        void *dev_hp;
        cudaMalloc(&dev_p, this->weights_length * sizeof(float));
        CUDA_CHECK(cudaMemcpy
        (dev_p, this->weights_ptr, this->weights_length * sizeof(float), cudaMemcpyHostToDevice));
            //压缩
        AccHalf::Accuracyhalf(dev_p, &dev_hp, this->weights_length);
            //重新赋值
        this->weights_type=PluginFieldType::kFLOAT16;
        delete[] this->weights_ptr;
        this->weights_ptr=new half[this->weights_length];
        CUDA_CHECK(cudaMemcpy
        (this->weights_ptr, dev_hp, this->weights_length * sizeof(half), cudaMemcpyDeviceToHost));
            //释放临时资源
        cudaFree(dev_p);
        cudaFree(dev_hp);
    } 
    else 
    {
        std::cerr<<"不支持的Conv1d计算精度\n";
        exit(false);
    }
    //压缩偏置精度
    if (this->mprecision == NVfloat)
    {} 
    else if (this->mprecision == NVhalf) 
    {
        float *dev_p;
        void *dev_hp;
        cudaMalloc(&dev_p, this->bias_length * sizeof(float));
        CUDA_CHECK(cudaMemcpy
        (dev_p, this->bias_ptr, this->bias_length * sizeof(float), cudaMemcpyHostToDevice));
            //压缩
        AccHalf::Accuracyhalf(dev_p, &dev_hp, this->bias_length);
            //重新赋值
        this->bias_type=PluginFieldType::kFLOAT16;
        delete[] this->bias_ptr;
        this->bias_ptr=new half[this->bias_length];
        CUDA_CHECK(cudaMemcpy
        (this->bias_ptr, dev_hp, this->bias_length * sizeof(half), cudaMemcpyDeviceToHost));
            //释放临时资源
        cudaFree(dev_p);
        cudaFree(dev_hp);
    } 
    else 
    {
        std::cerr<<"不支持的Conv1d计算精度\n";
        exit(false);
    }
}

std::size_t Conv1d::getWorkspaceSize(int32_t maxBatchSize) const noexcept 
{
    //无需额外显存
    return 0;
}

int32_t Conv1d::initialize() noexcept 
{
    cudnnCreate(&nn_handle_descr);
        //冗余判断
    assert(this->mtype == this->mprecision);
    assert(this->mnbinputs == 1);
    assert(this->mnboutputs == 1);
        //设定推理期间张量精度与格式
        //config中设定张量精度与计算精度一致
    switch (this->mtype)
    {
    case DataType::kFLOAT: //张量全精度
        nn_tensorprecision=cudnnDataType_t::CUDNN_DATA_FLOAT;
        break;
    case DataType::kHALF:
        nn_tensorprecision=cudnnDataType_t::CUDNN_DATA_HALF;
        break;
    default:
        std::cerr<<"不支持张量的精度\n"; //已在config检查
        exit(false);
    }
        switch (this->mformat)
    {
    case TensorFormat::kLINEAR:
        nn_tensorformat=cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
        break;
    default:
        std::cerr<<"不支持的数据格式\n";
        break;
    }
        //设定推理期间权重精度（计算精度）
    switch (this->mprecision)
    {
    case NVfloat:
        nn_weightsprecision=cudnnDataType_t::CUDNN_DATA_FLOAT;
        break;
    case NVhalf:
        nn_weightsprecision=cudnnDataType_t::CUDNN_DATA_HALF;
        break;
    default:
        std::cerr<<"不支持权重的精度\n"; //已在config检查
        exit(false);
    }

        //输入tensor设定
    cudnnCreateTensorDescriptor(&nn_inputtensor_descr);    
    cudnnSetTensor4dDescriptor(nn_inputtensor_descr,
        nn_tensorformat, 
        nn_tensorprecision,
        this->maxbatch, //最大批量预留足够的worksize
        this->minputdims.d[0], //输入通道
        1, //伪三维张量
        this->minputdims.d[1] //序列长度
    );
        //卷积滤波句柄设定
    cudnnCreateFilterDescriptor(&nn_conv1d_filter_descr);
    cudnnSetFilter4dDescriptor(nn_conv1d_filter_descr, 
        nn_weightsprecision, 
        nn_tensorformat,
        this->minputdims.d[0], //输入通道数
        this->moutputdims.d[0], //输出通道数
        1, //高度设置必须为1
        this->kernelsize //卷积核水平尺寸 
    );
        //卷积操作句柄设定
    cudnnCreateConvolutionDescriptor(&nn_conv1d_descr);
    int padding=(this->kernelsize - 1) / 2; //序列长填充1
    cudnnSetConvolution2dDescriptor(nn_conv1d_descr,
        0, padding, //只序列长度填充
        1, 1, //水平步长为1
        1, 1, //卷积无间隔无空洞
        cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION, //卷积模式
        nn_weightsprecision //计算时精度  
    ); 
        //输出tensor设定 得到向前输出形状
    int n, c, h, w=0;
    cudnnGetConvolution2dForwardOutputDim(nn_conv1d_descr,
        nn_inputtensor_descr,
        nn_conv1d_filter_descr, 
        &n, &c, &h, &w //形状信息
    );
    if ((n==this->maxbatch) && (c==this->minputdims.d[0]) && (h==1) && (w==this->minputdims.d[1]))
    {} //通道与序列长度不变
    else {
        std::cerr<<"向前传播形状不正确\n";
        std::cerr<<"batch:"<<n<<"channels:"<<c<<"height:"<<h<<"width:"<<w<<"\n";
        exit(false);
    }
        //设定
    cudnnCreateTensorDescriptor(&nn_outputtensor_descr);
    cudnnSetTensor4dDescriptor(nn_outputtensor_descr,
        cudnnDataType_t::CUDNN_DATA_FLOAT,
        nn_tensorprecision,
        n, c, h, w
    );
        //偏置设定
    cudnnCreateTensorDescriptor(&nn_bias_descr);
    cudnnSetTensor4dDescriptor(nn_bias_descr,
        nn_tensorformat,
        nn_tensorprecision,
        1, c, 1, 1 //批量为1广播到所有样本的所有元素
    );

        //算法描述设定
    int peralgo_count=0;
    auto status=cudnnFindConvolutionForwardAlgorithm(nn_handle_descr,
    nn_inputtensor_descr, 
    nn_conv1d_filter_descr, 
    nn_conv1d_descr, 
    nn_outputtensor_descr, 
    5, &peralgo_count, //申请最优解 实际返回的解法数目
    nn_conv_allalgo_descr //储存算法结构体        
    );
    assert(peralgo_count > 0);
    assert(status == CUDNN_STATUS_SUCCESS);
        //尝试tensorcore
    if (nn_tensorprecision == cudnnDataType_t::CUDNN_DATA_HALF)
    {
        for (int i=0; i<peralgo_count; i++)
    {
        if (nn_conv_allalgo_descr[i].status == CUDNN_STATUS_SUCCESS)
        {
            if (nn_conv_allalgo_descr[i].mathType==CUDNN_TENSOR_OP_MATH ||
                nn_conv_allalgo_descr[i].mathType==CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)
            {
                status=cudnnSetConvolutionMathType
                (nn_conv1d_descr, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
                assert(status == CUDNN_STATUS_SUCCESS); //设置fp32与half混合

                nn_conv_bestalgo_descr=nn_conv_allalgo_descr[i].algo;
                break;
            }
        }        
        else
            continue;
    }
    }
        //回退
    else 
    {
        for (int i=0; i<peralgo_count; i++)
    {
        if (nn_conv_allalgo_descr[i].status == CUDNN_STATUS_SUCCESS)
        {
            if (nn_conv_allalgo_descr[i].mathType==CUDNN_DEFAULT_MATH ||
                nn_conv_allalgo_descr[i].mathType==CUDNN_FMA_MATH)
            {
                status=cudnnSetConvolutionMathType
                (nn_conv1d_descr, CUDNN_DEFAULT_MATH);
                assert(status == CUDNN_STATUS_SUCCESS); //设置默认fp32

                nn_conv_bestalgo_descr=nn_conv_allalgo_descr[i].algo;
                break;
            }
        }        
        else
            continue;
    }
    }
        //计算向前传播空间
    cudnnGetConvolutionForwardWorkspaceSize(nn_handle_descr,
        nn_inputtensor_descr,
        nn_conv1d_filter_descr,
        nn_conv1d_descr,
        nn_outputtensor_descr,
        nn_conv_bestalgo_descr,
        &cu_workspacesize
    );

        //cuda
    assert(cu_workspacesize > 0);
    CUDA_CHECK(cudaMalloc(&cu_workptr, cu_workspacesize));
        //权重数据
    if (nn_weightsprecision == cudnnDataType_t::CUDNN_DATA_FLOAT) 
    {
    CUDA_CHECK(cudaMalloc(&cu_weightsptr, this->weights_length * sizeof(float)));
    CUDA_CHECK(cudaMemset(cu_weightsptr, 0, this->weights_length * sizeof(float)));
    CUDA_CHECK(cudaMemcpy
    (cu_weightsptr, this->weights_ptr, this->weights_length * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
    } 
    else if (nn_weightsprecision == cudnnDataType_t::CUDNN_DATA_HALF) 
    {
    CUDA_CHECK(cudaMalloc(&cu_weightsptr, this->weights_length * sizeof(half)));
    CUDA_CHECK(cudaMemset(cu_weightsptr, 0, this->weights_length * sizeof(half)));
    CUDA_CHECK(cudaMemcpy
    (cu_weightsptr, this->weights_ptr, this->weights_length * sizeof(half), cudaMemcpyKind::cudaMemcpyHostToDevice));
    } else {
        std::cerr<<"不支持权重的精度无法分配\n";
        exit(false);
    }
        //偏置数据
    if (nn_tensorprecision == cudnnDataType_t::CUDNN_DATA_FLOAT) 
    {
    CUDA_CHECK(cudaMalloc(&cu_biasptr, this->bias_length * sizeof(float)));
    CUDA_CHECK(cudaMemset(cu_biasptr, 0, this->bias_length * sizeof(float)));
    CUDA_CHECK(cudaMemcpy
    (cu_biasptr, this->bias_ptr, this->bias_length * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
    } 
    else if (nn_tensorprecision == cudnnDataType_t::CUDNN_DATA_HALF) 
    {
    CUDA_CHECK(cudaMalloc(&cu_biasptr, this->bias_length * sizeof(half)));
    CUDA_CHECK(cudaMemset(cu_biasptr, 0, this->bias_length * sizeof(half)));
    CUDA_CHECK(cudaMemcpy
    (cu_biasptr, this->bias_ptr, this->bias_length * sizeof(half), cudaMemcpyKind::cudaMemcpyHostToDevice));
    } else {
        std::cerr<<"不支持偏置的精度无法分配\n";
        exit(false);
    }

    return status;
}

int32_t Conv1d::enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    cudnnSetStream(nn_handle_descr, stream);
    //更新输入
    int channels, height, width=0;
    cudnnGetTensor4dDescriptor(nn_inputtensor_descr,
        nullptr, 
        nullptr, &channels, &height, &width,
        nullptr, nullptr, nullptr, nullptr
    );
    cudnnSetTensor4dDescriptor(nn_inputtensor_descr,
        nn_tensorformat,
        nn_tensorprecision,
        batchSize, channels, height, width    
    );
    //推理
    //output = alpha * Op(inputs) + beta * output
    float alpha=1.0f; //输入缩放因子
    float beta=0.0f; //输出缩放因子

    float *inputs_flo=static_cast<const float*>(inputs[0]);
    void *inputs_hal;

    if (this->mtype == NVhalf) 
    {
        //压缩
    int32_t inputs_elements=batchSize * channels * height * width;
    AccHalf::Accuracyhalf(inputs_flo, &inputs_hal, inputs_elements);

    cudnnStatus_t status=cudnnConvolutionForward(nn_handle_descr,
    &alpha, nn_inputtensor_descr, inputs_hal, //输入tensor
    nn_conv1d_filter_descr, cu_weightsptr, //滤波句柄 权重指针 
    nn_conv1d_descr, //操作句柄
    nn_conv_bestalgo_descr, //算法选择
    cu_workptr, cu_workspacesize, //推理空间
    &beta, nn_outputtensor_descr, outputs[0] //输出tensor
    );

    cudaFreeAsync(inputs_hal, stream);
    } 
    else 
    cudnnStatus_t status=cudnnConvolutionForward(nn_handle_descr,
    &alpha, nn_inputtensor_descr, inputs[0], //输入tensor
    nn_conv1d_filter_descr, cu_weightsptr, //滤波句柄 权重指针 
    nn_conv1d_descr, //操作句柄
    nn_conv_bestalgo_descr, //算法选择
    cu_workptr, cu_workspacesize, //推理空间
    &beta, nn_outputtensor_descr, outputs[0] //输出tensor
    );

    assert(status == CUDNN_STATUS_SUCCESS);
    cudaStreamSynchronize(stream);
    //添加偏置
    //C = alpha * A + beta * C
    beta=1.0f;
    status=cudnnAddTensor(nn_handle_descr,
        &alpha, nn_bias_descr, cu_biasptr,
        &beta, nn_outputtensor_descr, outputs[0]
    );
    assert(status == CUDNN_STATUS_SUCCESS);

    return status;
}

void Conv1d::terminate() noexcept
{
    cudnnDestroyTensorDescriptor(nn_inputtensor_descr);
    cudnnDestroyFilterDescriptor(nn_conv1d_filter_descr);
    cudnnDestroyConvolutionDescriptor(nn_conv1d_descr);
    cudnnDestroyTensorDescriptor(nn_outputtensor_descr);
    cudnnDestroyTensorDescriptor(nn_bias_descr);
    cudnnDestroy(nn_handle_descr);
    //
    CUDA_CHECK(cudaFree(cu_workptr));
    CUDA_CHECK(cudaFree(cu_weightsptr));
    CUDA_CHECK(cudaFree(cu_biasptr));
}

std::size_t Conv1d::getSerializationSize() const noexcept
{
    std::size_t serializesize=0;
        //核权重序列化
    serializesize+=sizeof(PluginFieldType);
    serializesize+=sizeof(int32_t);
        //根据计算精度保存权重 
    switch (this->mprecision)
    {
    case NVfloat:
        serializesize+=this->weights_length * sizeof(float);
        break;
    case NVhalf:
        serializesize+=this->weights_length * sizeof(half);
        break;
    default:
        exit(false);
    }
        //偏置权重序列化
    serializesize+=sizeof(PluginFieldType);
    serializesize+=sizeof(int32_t); 
        //根据计算精度保存偏置 
    switch (this->mprecision)
    {
    case NVfloat:
        serializesize+=this->bias_length * sizeof(float);
        break;
    case NVhalf:
        serializesize+=this->bias_length * sizeof(half);
        break;
    default:
        exit(false);
    }
        //张量精度与格式
    serializesize+=sizeof(DataType);
    serializesize+=sizeof(DataType);
    serializesize+=sizeof(TensorFormat);
        //最大批量
    serializesize+=sizeof(uint);
        //输入数量 维度
    serializesize+=sizeof(uint);
    serializesize+=sizeof(int32_t); //nbdims
    serializesize+=sizeof(int32_t) * this->minputdims.nbDims; //各维度值
        //输出数量 维度
    serializesize+=sizeof(uint);
    serializesize+=sizeof(int32_t); //nbdims
    serializesize+=sizeof(int32_t) * this->moutputdims.nbDims; //各维度值
    //
    return serializesize;
}

void Conv1d::serialize(void* buffer) const noexcept 
{
    char* raw_ptr=static_cast<char*>(buffer); // char单位
    char* start=static_cast<char*>(buffer);
  
        //张量精度与格式
    std::memcpy(raw_ptr, &this->mtype, sizeof(DataType));
    raw_ptr+=sizeof(DataType);
    std::memcpy(raw_ptr, &this->mprecision, sizeof(DataType));
    raw_ptr+=sizeof(DataType);
    std::memcpy(raw_ptr, &this->mformat, sizeof(TensorFormat));
    raw_ptr+=sizeof(TensorFormat);
        //最大批量
    std::memcpy(raw_ptr, &this->maxbatch, sizeof(uint));
    raw_ptr+=sizeof(uint);
        //输入数量及维度
    std::memcpy(raw_ptr, &this->mnbinputs, sizeof(uint));
    raw_ptr+=sizeof(uint);
    std::memcpy(raw_ptr, &this->minputdims.nbDims, sizeof(int32_t));
    raw_ptr+=sizeof(int32_t);
    for (int i=0; i<this->minputdims.nbDims; ++i) {
        std::memcpy(raw_ptr, &this->minputdims.d[i], sizeof(int32_t));
        raw_ptr+=sizeof(int32_t);
    }
        // 输出数量及维度
    std::memcpy(raw_ptr, &this->mnboutputs, sizeof(uint));
    raw_ptr+=sizeof(uint);
    std::memcpy(raw_ptr, &this->moutputdims.nbDims, sizeof(int32_t));
    raw_ptr+=sizeof(int32_t);
    for (int i=0; i<this->moutputdims.nbDims; ++i) {
        std::memcpy(raw_ptr, &this->moutputdims.d[i], sizeof(int32_t));
        raw_ptr+=sizeof(int32_t);
    }

        //核权重序列化
    std::memcpy(raw_ptr, &this->weights_type, sizeof(PluginFieldType));
    raw_ptr+=sizeof(PluginFieldType);
    std::memcpy(raw_ptr, &this->weights_length, sizeof(int32_t));
    raw_ptr+=sizeof(int32_t);
    //
    if (this->mprecision == NVfloat)
    {
    std::memcpy(raw_ptr, this->weights_ptr, this->weights_length * sizeof(float));
    raw_ptr+=this->weights_length * sizeof(float);
    } 
    else if (this->mprecision == NVhalf) 
    {
    std::memcpy(raw_ptr, this->weights_ptr, this->weights_length * sizeof(half));
    raw_ptr+=this->weights_length * sizeof(half);
    } 
    else 
        exit(false);
        //偏置权重序列化
    std::memcpy(raw_ptr, &this->bias_type, sizeof(PluginFieldType));
    raw_ptr+=sizeof(PluginFieldType);
    std::memcpy(raw_ptr, &this->bias_length, sizeof(int32_t));
    raw_ptr+=sizeof(int32_t);
    //
    if (this->mprecision == NVfloat)
    {
    std::memcpy(raw_ptr, this->bias_ptr, this->bias_length * sizeof(float));
    raw_ptr+=this->bias_length * sizeof(float);
    } 
    else if (this->mprecision == NVhalf) 
    {
    std::memcpy(raw_ptr, this->bias_ptr, this->bias_length * sizeof(half));
    raw_ptr+=this->bias_length * sizeof(half);
    } 
    else 
        exit(false);

    assert((raw_ptr-start) == getSerializationSize());
    //
    delete[] this->weights_ptr;
    delete[] this->bias_ptr;
    this->weights_ptr=nullptr;
    this->bias_ptr=nullptr;
}

IPluginV2 *Conv1d::clone() const noexcept
{
    return new Conv1d(*this);
}

void Conv1d::destroy() noexcept
{
    delete this;
} 
/************一维度卷积创建器************/
// 定义静态成员变量
std::vector<nvinfer1::PluginField> Conv1dCreator::field_vec;
nvinfer1::PluginFieldCollection Conv1dCreator::fieldcoll={0, nullptr};
AsciiChar const* Conv1dCreator::getPluginName() const noexcept
{
    const char *name="Conv1d";
    return name;
}
AsciiChar const* Conv1dCreator::getPluginVersion() const noexcept
{
    const char *version="1.0";
    return version;
}
//
PluginFieldCollection const* Conv1dCreator::getFieldNames() noexcept
{
    fieldcoll.nbFields=0;
    fieldcoll.fields=nullptr;
    if (field_vec.empty()) {
        //out_channels参数
        field_vec.emplace_back("out_channels", nullptr, PluginFieldType::kUNKNOWN, int32_t(1));
        //weights
        field_vec.emplace_back("weights", nullptr, PluginFieldType::kUNKNOWN, 0);
        //bias
        field_vec.emplace_back("bias", nullptr, PluginFieldType::kUNKNOWN, 0);
    }
    //汇总
    fieldcoll.fields=field_vec.data();
    fieldcoll.nbFields=field_vec.size();

    return &fieldcoll;
}

IPluginV2* Conv1dCreator::createPlugin(AsciiChar const* name, PluginFieldCollection const* fc) noexcept 
{
    assert(fc != nullptr);
    assert(fc->nbFields == 3);
    if (ssd::strcmp(fc->fields[0].name, Conv1dCreator::field_vec[0].name) != 0) {
        std::cerr<<"权重传递名称不符合"<<Conv1dCreator::field_vec[0].name<<"\n";
        exit(false);
    } else if (ssd::strcmp(fc->fields[1].name, Conv1dCreator::field_vec[1].name) != 0) {
        std::cerr<<"权重传递名称不符合"<<Conv1dCreator::field_vec[1].name<<"\n";
        exit(false);
    } else if (ssd::strcmp(fc->fields[2].name, Conv1dCreator::field_vec[2].name) != 0) {
        std::cerr<<"权重传递名称不符合"<<Conv1dCreator::field_vec[2].name<<"\n";
        exit(false);
    }
    //指针浅拷贝
    PluginField pf_weights;
    pf_weights.name=fc->fields[1].name;
    pf_weights.data=fc->fields[1].data;
    pf_weights.type=fc->fields[1].type;
    pf_weights.length=fc->fields[1].length;
    
    PluginField pf_bias;
    pf_bias.name=fc->fields[2].name;
    pf_bias.data=fc->fields[2].data;
    pf_bias.type=fc->fields[2].type;
    pf_bias.length=fc->fields[2].length;
    //
    Conv1d* plugin=new Conv1d(pf_weights, pf_bias);
    //PluginField构造后析构
    //命名空间在父类指向
    //
    return plugin;
}

IPluginV2* Conv1dCreator::deserializePlugin(AsciiChar const* name, void const* serialData, std::size_t serialLength) noexcept 
{
    assert(serialData != nullptr);
    assert(serialLength > 0);
    Conv1d* plugin=new Conv1d(name, serialData, serialLength);
    return plugin;
}



