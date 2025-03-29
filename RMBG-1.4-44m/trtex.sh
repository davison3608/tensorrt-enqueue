ONNX_MODEL_PATH="/codes/vision_tensorrt/py/loading_all/huggingface_downloads/RMBG-1.4-44m/onnxruntime/model.onnx"
TRT_ENGINE_PATH="/codes/vision_tensorrt/py/loading_all/trt_export/RMBG-1.4-44m/engine.trt"

# 设置工作空间为 4 GB（单位为 MB）
MEM_POOL_SIZE=4096  

MIN_SHAPES="input:1x3x1024x1024"  
OPT_SHAPES="input:2x3x1024x1024"  
MAX_SHAPES="input:4x3x1024x1024"

# 使用 trtexec 转换 ONNX 模型为 TensorRT 引擎
trtexec --onnx=$ONNX_MODEL_PATH \
        --saveEngine=$TRT_ENGINE_PATH \
        --workspace=$MEM_POOL_SIZE \
        --minShapes=$MIN_SHAPES \
        --optShapes=$OPT_SHAPES \
        --maxShapes=$MAX_SHAPES
