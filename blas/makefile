CUDA_ROOT       = /usr/local/cuda
TENSORRT_ROOT   = /usr/local/TensorRT-8.6.1.6
CV2             = /usr/local/include/opencv4
NVNN            = /codes/nvnn
CV              = /codes/cv
BLAS            = /codes/blas
# -02 不优化
all:
	nvcc -g -G -std=c++14 \
	-Xcompiler -fopenmp \
	-gencode arch=compute_61,code=sm_61 \
        -gencode arch=compute_80,code=sm_80 \
        -I $(CUDA_ROOT)/include \
        -I $(TENSORRT_ROOT)/include \
        -L $(CUDA_ROOT)/lib64 \
        -L $(TENSORRT_ROOT)/lib \
        -I $(CV2) \
        -I $(NVNN)/include \
        -I $(CV)/include \
        -I $(BLAS)/include \
        -o ./cumain \
        blas.cpp \
        $(BLAS)/src/cumatrix.cu \
        $(BLAS)/src/oclmatrix.cpp \
        $(CV)/src/vision_cv.cpp \
        $(CV)/src/core.cpp \
        $(NVNN)/src/cuda_core.cu \
        $(NVNN)/src/cuda_rand.cu \
        -lnvinfer -lnvinfer_plugin \
        -lcudart -ldl -lrt -lnvonnxparser \
        -lcudnn -lcublas -lcurand -lOpenCL -lclBLAS \
        -Xlinker -rpath -Xlinker \
        $(shell pkg-config --libs opencv4)

ocl:
	g++ -g -Wall -std=c++14 \
        -fopenmp \
        -I $(CV2) \
        -I $(CV)/include \
        -I $(BLAS)/include \
        -o ./clmain \
        blas.cpp \
        $(BLAS)/src/oclmatrix.cpp \
        $(BLAS)/src/kernel.cl \
        $(CV)/src/vision_cv.cpp \
        $(CV)/src/core.cpp \
        -lOpenCL -lclBLAS \
        $(shell pkg-config --libs opencv4)

clean:
	rm -f ./clmain ./cumain
