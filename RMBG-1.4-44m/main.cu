    //curand
    //std::vector<float> matrix_vec;
    //UniformRand rand(Dims4{4, 3, 1024, 1024});
    //rand.GenerateMatrixRand(matrix_vec);

    //4张图片 预处理 归一
    CVpre pre0("/codes/vision_tensorrt/3rth670.jpg");
    CVpre pre1("/codes/vision_tensorrt/ferri2fsd.jpg");
    CVpre pre2("/codes/vision_tensorrt/sunmwd.jpg");
    CVpre pre3("/codes/vision_tensorrt/kmr45ghn.jpg");
    pre0.ResizePicturesShape(std::vector<uint>{1024, 1024});
    pre1.ResizePicturesShape(std::vector<uint>{1024, 1024});
    pre2.ResizePicturesShape(std::vector<uint>{1024, 1024});
    pre3.ResizePicturesShape(std::vector<uint>{1024, 1024});
    //连续展平
    std::array<std::vector<float>, 4> img_data_vecs;
    pre0.GetCHWPictureNormalizedData(img_data_vecs[0]);
    pre1.GetCHWPictureNormalizedData(img_data_vecs[1]);
    pre2.GetCHWPictureNormalizedData(img_data_vecs[2]);
    pre3.GetCHWPictureNormalizedData(img_data_vecs[3]);
    //连续插入到vec
    std::size_t vec_counts=img_data_vecs[0].size() * 4;
    std::vector<float> src_vec;
    src_vec.reserve(vec_counts);
        for(auto& vec: img_data_vecs)
    {
        src_vec.insert(src_vec.end(), vec.begin(), vec.end());
    }  
    assert(src_vec.size() == vec_counts);

    //cv chw保存 test
    //auto pic=GetCHWflatten2Picture(img_data_vecs[0], vector<uint>{1024, 1024, 3});
    //assert(SavePictures(pic, "/codes/vision_tensorrt/src_img.jpg"));

    //rand test
    //std::vector<float> matrix_test;
    //UniformRand rand_test(Dims3{3, 1024, 1024});
    //rand_test.GenerateMatrixRand(matrix_test);
    //auto img=GetCHWflatten2Picture(matrix_test, vector<uint>{1024, 1024, 3});
    //assert(SavePictures(img, "/codes/vision_tensorrt/rand_img.jpg"));

    CUDA_CHECK(cudaSetDevice(0));

ifstream file("/codes/vision_tensorrt/py/loading_all/trt_export/RMBG-1.4-44m/engine.trt", std::ios::binary);
assert(file.is_open());

//读取
file.seekg(0, file.end);
std::size_t size=file.tellg();
file.seekg(0, file.beg);

std::vector<char> engine_vec;
engine_vec.resize(size);
file.read(engine_vec.data(), size);

    auto runtime=createInferRuntime(self_log);
    //反序列化
    auto engine=runtime->deserializeCudaEngine(engine_vec.data(), size);

    //上下文
    auto context=engine->createExecutionContext();
    //上下文缓存
    int nbiotensors=engine->getNbIOTensors();
    void *buffers[nbiotensors];

    //流
    cudaStream_t con0_enq;
    cudaStreamCreateWithFlags(&con0_enq, cudaStreamNonBlocking);

    cudaEvent_t proc_evt;
    cudaEventCreate(&proc_evt);

    //声明固定后的输出形状
    static std::size_t out_row;
    static std::size_t out_col;
    static std::size_t out_channels;
    //static std::size_t out_pitch;
        for (int idx=0; idx<nbiotensors; idx++)
    {
    //infos
    const char *name=engine->getIOTensorName(idx);
    //name type 动态形状
    TensorIOMode mode=engine->getTensorIOMode(name);

    DataType type=engine->getTensorDataType(name);
    
    Dims min_dims=engine->getProfileShape(name, 0, OptProfileSelector::kMIN);
    Dims opt_dims=engine->getProfileShape(name, 0, OptProfileSelector::kOPT);
    Dims max_dims=engine->getProfileShape(name, 0, OptProfileSelector::kMAX);
    //
    //输入输出固定
    if (mode == TensorIOMode::kINPUT) 
    {   
    assert(type == DataType::kFLOAT);
    assert(min_dims.d[0] != -1);
    assert(opt_dims.d[0] != -1);
    assert(max_dims.d[0] != -1);
    //设置最大批量
    context->setInputShape(name, max_dims);

    //开辟空间
    Dims input_shape=max_dims;
    std::size_t input_size;
    std::size_t input_elements;
    std::size_t row=input_shape.d[input_shape.nbDims - 2];
    std::size_t col=input_shape.d[input_shape.nbDims - 1];
    std::size_t pitch;
    input_elements=input_shape.d[0] * input_shape.d[1] * row * col;
    input_size=input_elements * sizeof(float);

    assert(input_size/sizeof(float) == input_shape.d[0] * input_shape.d[1] * input_shape.d[2] * input_shape.d[3]);
    
    CUDA_CHECK(cudaMallocAsync(&buffers[idx], input_size, con0_enq));
    CUDA_CHECK(cudaMemset(buffers[idx], 0, input_size));
    CUDA_CHECK
    (cudaMemcpyAsync(buffers[idx], src_vec.data(), input_size, cudaMemcpyHostToDevice, con0_enq));
    assert(src_vec.size() == input_shape.d[0] * input_shape.d[1] * row * col);

    //rand memcpy
    //CUDA_CHECK(cudaMemcpyAsync(buffers[idx], matrix_vec.data(), input_size, cudaMemcpyHostToDevice, con0_enq));
    //assert(matrix_vec.size() == input_shape.d[0] * input_shape.d[1] * row * col);
    
    //CUDA_CHECK(cudaMallocPitch(&buffers[idx], &pitch, col * sizeof(float), row));
    //CUDA_CHECK(cudaMemcpy2DAsync
    //(buffers[idx], pitch, matrix_vec.data(), col * sizeof(float), col * sizeof(float), row, cudaMemcpyHostToDevice, con0_enq));
    //
    } else
    {
    assert(type == DataType::kFLOAT);
    //固定后张量
    Dims output_shape=context->getTensorShape(name);
    assert(output_shape.nbDims == 4);

    std::size_t output_size;
    std::size_t output_elements;
    out_row=output_shape.d[output_shape.nbDims - 2];
    out_col=output_shape.d[output_shape.nbDims - 1];
    out_channels=output_shape.d[output_shape.nbDims - 3];
    //开辟空间
    output_elements=output_shape.d[0] * output_shape.d[1] * out_row * out_col;
    output_size=output_elements * sizeof(float);

    assert(output_size/sizeof(float) == output_shape.d[0] * output_shape.d[1] * output_shape.d[2] * output_shape.d[3]);

    CUDA_CHECK(cudaMallocAsync(&buffers[idx], output_size, con0_enq));
    //CUDA_CHECK(cudaMallocPitch(&buffers[idx], &out_pitch, out_col * sizeof(float), out_row));
    CUDA_CHECK(cudaMemset(buffers[idx], 0, output_size));
    }
    }
    //
    cudaStreamSynchronize(con0_enq);
    CUDA_CHECK(cudaEventRecord(proc_evt));

    //记录
    cudaStreamWaitEvent(con0_enq, proc_evt);
    std::array<cudaEvent_t, 1> enq_event_arr;
    for (auto& event: enq_event_arr)
        cudaEventCreate(&event);

    //推理
    auto status0=context->enqueueV2(buffers, con0_enq, &enq_event_arr[0]);
    assert(status0);
    cudaStreamSynchronize(con0_enq);

    //记录
    for (auto& event: enq_event_arr)
        cudaEventRecord(event);

    float time;
    for (auto& event: enq_event_arr)
    {
        cudaEventElapsedTime(&time, proc_evt, event);
        cout<<"\n推理用时--"<<time<<"\n\n";
    }

    //回传
    std::vector<float> dst_vec;
    std::array<std::vector<float>, 4> out_img_data_vecs;
    //
    for (int idx=0; idx<nbiotensors; idx++)
    {
    //infos
    const char *name=engine->getIOTensorName(idx);
    //name type 形状
    TensorIOMode mode=engine->getTensorIOMode(name);
    DataType type=engine->getTensorDataType(name);
    //
    if (mode == TensorIOMode::kINPUT) 
        continue;
    else {
        Dims output_dims=context->getTensorShape(name);
        std::size_t out_counts=output_dims.d[0] * output_dims.d[1] * output_dims.d[2] * output_dims.d[3];
        std::size_t single_out_counts=out_counts / output_dims.d[0];

    assert(output_dims.d[2] * output_dims.d[3] == out_row * out_col);

    //out_counts全批量输出元素数目
    dst_vec.resize(out_counts);
    CUDA_CHECK
    (cudaMemcpy(dst_vec.data(), buffers[idx], out_counts * sizeof(float), cudaMemcpyDeviceToHost));
    assert(dst_vec.size() == out_counts);
    
    //single_out_counts为单样本输出元素数目
        for (int idx=0; idx<out_img_data_vecs.size(); idx++)
    {
    out_img_data_vecs[idx].reserve(single_out_counts);
    //
    auto begin=dst_vec.data() + idx*single_out_counts;
    auto end=begin + single_out_counts;
    out_img_data_vecs[idx].insert(out_img_data_vecs[idx].end(), begin, end);
    //
    assert(out_img_data_vecs[idx].size() == single_out_counts);
    }

    //rand test
    //CUDA_CHECK(cudaMemcpyAsync(matrix_vec.data(), buffers[idx], out_counts * sizeof(float), cudaMemcpyDeviceToHost, con0_enq));

    //CUDA_CHECK(cudaMemcpy2D
    //(hos_vec.data(), out_col * sizeof(float), buffers[idx], out_pitch, out_col * sizeof(float), out_row, cudaMemcpyDeviceToHost));
    }
    }

    //cv保存每个容器chw格式数据
    auto seg_img0=GetCHWflatten2Picture
    (out_img_data_vecs[0], std::vector<uint>{static_cast<uint>(out_row), static_cast<uint>(out_col), static_cast<uint>(out_channels)});
    auto seg_img1=GetCHWflatten2Picture
    (out_img_data_vecs[1], std::vector<uint>{static_cast<uint>(out_row), static_cast<uint>(out_col), static_cast<uint>(out_channels)});
    auto seg_img2=GetCHWflatten2Picture
    (out_img_data_vecs[2], std::vector<uint>{static_cast<uint>(out_row), static_cast<uint>(out_col), static_cast<uint>(out_channels)});
    auto seg_img3=GetCHWflatten2Picture
    (out_img_data_vecs[3], std::vector<uint>{static_cast<uint>(out_row), static_cast<uint>(out_col), static_cast<uint>(out_channels)});
    assert(SavePictures(seg_img0, "/codes/vision_tensorrt/seg_img0.png"));
    assert(SavePictures(seg_img1, "/codes/vision_tensorrt/seg_img1.png"));
    assert(SavePictures(seg_img2, "/codes/vision_tensorrt/seg_img2.png"));
    assert(SavePictures(seg_img3, "/codes/vision_tensorrt/seg_img3.png"));

    //清理
    context->destroy();
    engine->destroy();

    for (auto& event: enq_event_arr)
        cudaEventDestroy(event);

    cudaStreamDestroy(con0_enq);

    for (int i=0; i<nbiotensors; i++)
        cudaFree(buffers[i]);
