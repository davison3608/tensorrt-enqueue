#ifndef CUDA_CORE
#define CUDA_CORE
#include "/codes/cv/include/core.hpp"
#include "/codes/nvnn/include/cuda_rand.h"
#include "/codes/nvnn/include/cuda_precison.h"

    //传递已分配输入矩阵数据 输出矩阵输出 矩阵元素数目 额外缓存 alpha参数
__global__ static void kernel_leakrelu_kf32
(const void* inputs, void *outputs, int32_t nbelements, void *workspace, float *alpha_kfloat);

__global__ static void kernel_leakrelu_khal
(const void* inputs, void *outputs, int32_t nbelements, void *workspace, half *alpha_khalf);

    //权重拷贝到容器
const std::vector<float> WeightswithBiasloading(const char *bin_path); 

    //数据包装成pf类 数据最终 指向datas指针
nvinfer1::PluginField CreatePluginField(const std::vector<float> &data_vec, const char *name, nvinfer1::PluginFieldType type);

/* 静态算子创建器 */
class BaseCreator: public nvinfer1::IPluginCreator
{
private:
    //插件注册必须先于网络构建
    //创建器类的生命周期独立于插件实例，且可能被多个插件共享
    //创建器仅负责插件实例的创建和反序列化，不持有运行时数据
protected:
    const char* IPluCrnamespace;

    explicit BaseCreator();
    ~BaseCreator();
public:
    /* 返回插件所依赖的 TensorRT 的版本号 */
    int32_t getTensorRTVersion() const noexcept override;
    /* 设置命名空间 默认""
       通过setPluginNamespace显式设置，同时调用到插件setPluginNamespace */
    void setPluginNamespace(nvinfer1::AsciiChar const* pluginNamespace) noexcept override;

    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override;

    /* 特定算子实现 */    
    /*            */
    /* 返回插件类型的名称（唯一） */
    virtual nvinfer1::AsciiChar const* getPluginName() const noexcept override=0;

    /* 返回插件版本（唯一） */
    virtual nvinfer1::AsciiChar const* getPluginVersion() const noexcept override=0;

    /* 核心，返回一个PluginFieldCollection对象，声明包含插件的所有可配置超参数，注意data指针的数据与length元素数量匹配
       涉及算子本身的公式参数与类型 与算子类的configureWithFormat方法不同，后者配置实际的张量数据信息
       在网络构建阶段前通过createPlugin方法传递给插件的构造函数 声明的结构体应该是静态属性避免被析构*/
    virtual nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override=0;

    /* IPluginV2 的构造函数中，解析从IPluginCreator::createPlugin传递的静态参数（如卷积核大小、固定权重等
       getFieldNames返回引用结构体，只起到声明作用，fc是用户或框架传递的参数值集合，而非 getFieldNames() 返回的元数据集合*/
    virtual nvinfer1::IPluginV2* createPlugin(nvinfer1::AsciiChar const* name, nvinfer1::PluginFieldCollection const* fc) 
    noexcept override=0;

    /* 反序列化构造 */
    virtual nvinfer1::IPluginV2* deserializePlugin(nvinfer1::AsciiChar const* name, void const* serialData, std::size_t serialLength) 
    noexcept override=0;
};

/*静态张量算子*/
class BasePluign: public nvinfer1::IPluginV2 
{ 
private: 
    /* supportsFormat配置期计算变量 无需序列化 克隆期计算
       initialize推理期 无须序列化变量 无须克隆 */
protected:
    const char* IPlunamespace;

    explicit BasePluign();     
    /* 构造函数 在构建引擎时期传递必要的配置信息 */
    /* 反序列化构造函数 必须与 serialize 方法配合使用 反序列化调用 */
    /* 复制构造函数 引擎构建过程中可能需要克隆插件对象以支持多设备或优化 */
    /* 避免隐式传递，而是通过调用clone函数实现传递副本 */
    
    /* 析构函数 */
    /* 释放插件持有的主机资源 */
    ~BasePluign();      
public:     
    /* 返回插件所依赖的 TensorRT 的版本号 */
    int32_t getTensorRTVersion() 
    const noexcept override;     
    /* 设置命名空间 默认"" */
    void setPluginNamespace(nvinfer1::AsciiChar const* pluginNamespace) 
    noexcept override;
    /* 获取当前插件的命名空间，唯一性与多线程安全
       支持插件反序列化，当反序列化引擎时，Tensorrt通过命名空间+插件类型的组合查找对应的IPluginCreator */
    nvinfer1::AsciiChar const* getPluginNamespace() 
    const noexcept override;
    /* 是否支持特定精度 特定数据格式 确保输入输出数据类型组合正确 */
    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) 
    const noexcept override;     

    /* 特定算子实现 */    
    /*            */
    /* 返回插件类型的名称（唯一） */
    virtual nvinfer1::AsciiChar const* getPluginType() 
    const noexcept override=0;     
    
    /* 返回插件版本（唯一） */
    virtual nvinfer1::AsciiChar const* getPluginVersion() 
    const noexcept override=0;     
    
    /* 返回输出数量 */
    virtual int32_t getNbOutputs() 
    const noexcept override=0;     
    
    /* 返回计算的输出维度 输入索引 dims数组 张量数量 
       每个profile的min、opt、max形状均会触发验证 */
    virtual nvinfer1::Dims getOutputDimensions(int32_t index, nvinfer1::Dims const* inputs, int32_t nbInputDims) 
    noexcept override=0;     
    
    /* 构建引擎配置插件的输入输出维度、数据类型、张量格式以及最大批处理大小 在内进行输入出张量形状检查 精度与数据格式检查 
       保存私有参数 builder->buildSerializedNetwork()，TensorRT 会遍历所有层，并为每个插件调用configureWithFormat
       此方法中调整内部逻辑（例如分配内存、校验输入合法性、选择算法实现等）
       隐式批次模式，dims中不包含批量大小 */
    virtual void configureWithFormat(nvinfer1::Dims const* inputDims, int32_t nbInputs, nvinfer1::Dims const* outputDims, int32_t nbOutputs,
    nvinfer1::DataType type, nvinfer1::PluginFormat format, int32_t maxBatchSize) 
    noexcept override=0;     
    
    /* 插件分配必要的资源，比如gpu内存作为临时缓冲区 cuda内核 常量参数如权重，偏置等 在run之前调用一次 
       区别于getWorkspaceSize 前者为长期资源，在引擎加载到销毁长期有效 */
    virtual int32_t initialize() 
    noexcept override=0;     
    
    /* 推理 依赖initialize分配的资源 enqueue内不可使用cudamalloc申请内存，以免造成内存冲突，由tensorrt统一管理 
       推理期间传递的workspace指针依赖于getWorkspaceSize返回，由tensorrt统一管理 
       batchSize由推理期间固定的批量大小 其中inputs与outputs为指针数组 */
    virtual int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) 
    noexcept override=0;     
    
    /* 在即将销毁前调用一次 释放资源 
       与算子析构函数分离 前者由tensorrt管理后者由c++上下文管理否则cuda环境失效使用cudafree */
    virtual void terminate() 
    noexcept override=0;     
    
    /* 构建引擎时期用于确定插件执行过程中需要的额外临时工作显存空间大小 */
    virtual std::size_t getWorkspaceSize(int32_t maxBatchSize) 
    const noexcept override=0;     
    
    /* 返回序列化的内存字节大小 算子参数 识别与版本号 输入输出维度信息 */
    virtual std::size_t getSerializationSize() 
    const noexcept override=0;     
    
    /* 序列化算子 储存信息 配合反序列化构造函数 禁止使用c++流，并且无需析构中释放由tensorrt统一管理 
       engine->serialize调用各算子getSerializationSize划分连续总内存 每个算子buffer指针指向该插件的专属存储区域
       buffer已开辟空间，使用深拷贝，避免内存冲突与多重释放 */
    virtual void serialize(void* buffer) 
    const noexcept override=0;  

    /* 算子融合与多设备推理复制算子本身，必须实现所有资源与成员的深拷贝，以及推理时依赖句柄
       多线程构建网络，并行构建网络的不同部分时需要线程隔离的插件副本
       return new 子类（*this）创建一份新的副本，同时调用副本拷贝构造函数，并传递当前的类
       在nerwork配置期最后调用 无需重复计算私有算子变量 */
    virtual IPluginV2* clone() 
    const noexcept override=0;

    /* 最后一步销毁插件本身，区别于terminate与析构函数的生命周期 */
    virtual void destroy() 
    noexcept override=0; 
};
//
/* 算子管理 */
class OperManger
{
private:
    //算子集标识与版本
    std::list<const char*> OpName_lt;
    std::list<const char*> OpVersion_lt;
public:
    explicit OperManger();
    ~OperManger();

    /* 链表中哪些算子可被注册 */
    void OperatorregistredShow() const noexcept;

    /* 将算子列入链表 */
    void isOperatorregistred(const char* opname, const char* opversion, const char* opnamespace);
};

/* 自定义算子集合 */
    //leakrelu激活函数
class leakrelu: public BasePluign
{
private:
    const char* onlyname;
    const char* onlyversion;

    int32_t mmaxbatchsize;

    int32_t mnbInputs;
    nvinfer1::Dims minputDims;

    int32_t mnbOutputs;
    nvinfer1::Dims moutputDims;     

    nvinfer1::DataType mtype;
    nvinfer1::DataType mprecision;
    nvinfer1::TensorFormat mFormat;

    //配置期计算 无须序列化变量
    int32_t nbelements;
    
    //initialize推理期 无须序列化变量 无须克隆
    float alpha;
    float *buffer=nullptr;
public:
    explicit leakrelu(float &alpha);
    explicit leakrelu(const char* name, const void* serialsizedata, std::size_t serialsizelength);
    explicit leakrelu(const nvinfer1::IPluginV2 &IPlu);
    explicit leakrelu() = delete;
    ~leakrelu();

    nvinfer1::AsciiChar const* getPluginType() 
    const noexcept override;

    nvinfer1::AsciiChar const* getPluginVersion() 
    const noexcept override;

    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) 
    const noexcept override;

    int32_t getNbOutputs() 
    const noexcept override;

    nvinfer1::Dims getOutputDimensions(int32_t index, nvinfer1::Dims const* inputs, int32_t nbInputDims) 
    noexcept override;
    //
    void configureWithFormat(nvinfer1::Dims const* inputDims, int32_t nbInputs, nvinfer1::Dims const* outputDims, int32_t nbOutputs,
    nvinfer1::DataType type, nvinfer1::PluginFormat format, int32_t maxBatchSize) 
    noexcept override;

    std::size_t getWorkspaceSize(int32_t maxBatchSize) 
    const noexcept override;

    int32_t initialize() 
    noexcept override;

    void terminate() 
    noexcept override;
    //
    int32_t enqueue
    (int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) 
    noexcept override;

    std::size_t getSerializationSize() 
    const noexcept override;

    void serialize(void* buffer) 
    const noexcept override;

    nvinfer1::IPluginV2* clone() 
    const noexcept override;

    void destroy() 
    noexcept override;
};
class leakrelucreator: public BaseCreator
{
private:
public:
    explicit leakrelucreator() = default;
    ~leakrelucreator() = default;

    nvinfer1::AsciiChar const* getPluginName() 
    const noexcept override;

    nvinfer1::AsciiChar const* getPluginVersion() 
    const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() 
    noexcept override;

    nvinfer1::IPluginV2* createPlugin(nvinfer1::AsciiChar const* name, nvinfer1::PluginFieldCollection const* fc) 
    noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(nvinfer1::AsciiChar const* name, void const* serialData, std::size_t serialLength) 
    noexcept override;
};
    //一维卷积全连接
class Conv1d: public BasePluign
{
private:
    const char* onlyname;
    const char* onlyversion;
    /*初始构造*/ 
        //卷积核尺寸 步幅 输出通道
    uint kernelsize=3;
    uint stride=1;
        //核权重与偏置权重
    nvinfer1::PluginFieldType weights_type;
    int32_t weights_length;
    mutable void *weights_ptr=nullptr;
        
    nvinfer1::PluginFieldType bias_type;
    int32_t bias_length;
    mutable void *bias_ptr=nullptr;

    /*config需要序列化*/
        //张量精度与格式
    nvinfer1::DataType mtype;
    nvinfer1::DataType mprecision;
    nvinfer1::TensorFormat mformat;
        //批量大小 张量形状信息
    uint maxbatch;
    uint mnbinputs;
    nvinfer1::Dims3 minputdims;
    uint mnboutputs;
    nvinfer1::Dims3 moutputdims; 
    /*config计算变量无需序列化*/
    
    /*equeue计算变量无需序列化*/
    cudnnHandle_t nn_handle_descr;
        //推理时张量精度与格式
    cudnnDataType_t nn_tensorprecision;
    cudnnTensorFormat_t nn_tensorformat;
        //权重精度（计算精度）
    cudnnDataType_t nn_weightsprecision;
        //输入 滤波 操作 输出 偏置句柄
        //确保输入输出张量数量为1
    cudnnTensorDescriptor_t nn_inputtensor_descr;
    cudnnFilterDescriptor_t nn_conv1d_filter_descr;
    cudnnConvolutionDescriptor_t nn_conv1d_descr;
    cudnnTensorDescriptor_t nn_outputtensor_descr;
    cudnnTensorDescriptor_t nn_bias_descr;
        //算法对象句柄
    cudnnConvolutionFwdAlgoPerf_t nn_conv_allalgo_descr[5];
    cudnnConvolutionFwdAlgo_t nn_conv_bestalgo_descr;
        //cudnn空间
    std::size_t cu_workspacesize=0;
    void* cu_workptr=nullptr;
    void* cu_weightsptr=nullptr;
    void* cu_biasptr=nullptr;
public:
    explicit Conv1d(nvinfer1::PluginField &weights, nvinfer1::PluginField &bias);
    explicit Conv1d(const char *name, const void* serialsizedata, std::size_t serialsizelength);
    explicit Conv1d(const nvinfer1::IPluginV2 &IPlu);
    explicit Conv1d() = delete;
    ~Conv1d();

    nvinfer1::AsciiChar const* getPluginType() 
    const noexcept override;

    nvinfer1::AsciiChar const* getPluginVersion() 
    const noexcept override;

    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) 
    const noexcept override;

    int32_t getNbOutputs() 
    const noexcept override;
    //参考卷积核 步幅 输入输出通道 
    nvinfer1::Dims getOutputDimensions(int32_t index, nvinfer1::Dims const* inputs, int32_t nbInputDims) 
    noexcept override;
    //
    void configureWithFormat(nvinfer1::Dims const* inputDims, int32_t nbInputs, 
    nvinfer1::Dims const* outputDims, int32_t nbOutputs,
    nvinfer1::DataType type, nvinfer1::PluginFormat format,
    int32_t maxBatchSize) 
    noexcept override;

    std::size_t getWorkspaceSize(int32_t maxBatchSize) 
    const noexcept override;

    int32_t initialize() 
    noexcept override;

    void terminate() 
    noexcept override;
    //
    int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, 
    void* workspace, cudaStream_t stream) 
    noexcept override;

    std::size_t getSerializationSize() 
    const noexcept override;
    //序列化后释放权重内存 buffer已开辟getSerializationSize空间 序列化完毕释放void*
    void serialize(void* buffer) 
    const noexcept override;

    nvinfer1::IPluginV2* clone() 
    const noexcept override;

    void destroy() 
    noexcept override;
};
class Conv1dCreator: public BaseCreator
{
private:
        //静态属性避免被析构
    static std::vector<nvinfer1::PluginField> field_vec;
    static nvinfer1::PluginFieldCollection fieldcoll;
public:
    explicit Conv1dCreator() = default;
    ~Conv1dCreator() = default;

    nvinfer1::AsciiChar const* getPluginName() 
    const noexcept override;

    nvinfer1::AsciiChar const* getPluginVersion() 
    const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() 
    noexcept override;

    nvinfer1::IPluginV2* createPlugin(nvinfer1::AsciiChar const* name, nvinfer1::PluginFieldCollection const* fc) 
    noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(nvinfer1::AsciiChar const* name, void const* serialData, std::size_t serialLength) 
    noexcept override;
};

#ifndef PLUGINV2_REGISTER
#define PLUGINV2_REGISTER
//宏注册
//leakrelu1.0激活
    REGISTER_TENSORRT_PLUGIN(leakrelucreator);
//conv1d1.0序列卷积
    REGISTER_TENSORRT_PLUGIN(Conv1dCreator);

#endif

#endif
