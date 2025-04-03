#include "include/test.h"
using namespace std;
int main(int argc, char const *argv[])
{
cl_int errNum;
cl_uint num_platforms=0;
cl_platform_id* platforms=nullptr;

//第一次调用获取平台数量
errNum = clGetPlatformIDs(0, nullptr, &num_platforms);
if (errNum != CL_SUCCESS) {
    std::cout << "查询平台数量失败 错误码 "<<errNum<<std::endl;
}

if (num_platforms == 0) {
    std::cout << "无可用平台"<<std::endl;
}

//分配内存
platforms=new cl_platform_id[num_platforms];

//第二次调用获取平台id
errNum = clGetPlatformIDs(num_platforms, platforms, nullptr);
if (errNum != CL_SUCCESS) {
    std::cout << "获取平台 ID 失败 错误码 "<<errNum<<std::endl;
    delete[] platforms;
}

std::cout<<"查询到可用平台数目 "<<num_platforms<<std::endl;

//当前平台
cl_platform_id current_plat=platforms[0];

//获取平台信息
//与前面一致 分两次传入 第一次申请获取内存字节大小 分配后传入
std::size_t size;

//版本
errNum=clGetPlatformInfo(current_plat, CL_PLATFORM_VERSION, 0, nullptr, &size);
char *version=new char[size];
errNum=clGetPlatformInfo(current_plat, CL_PLATFORM_VERSION, size, version, 0);
assert(errNum == CL_SUCCESS);
std::cout<<"OpenCL版本 "<<version<<"\n";

delete[] version;
//平台名称
errNum=clGetPlatformInfo(current_plat, CL_PLATFORM_NAME, 0, nullptr, &size);
char *name=new char[size];
errNum=clGetPlatformInfo(current_plat, CL_PLATFORM_NAME, size, name, 0);
assert(errNum == CL_SUCCESS);
std::cout<<"平台信息 "<<name<<"\n";

delete[] name;
//平台开发商
errNum=clGetPlatformInfo(current_plat, CL_PLATFORM_VENDOR, 0, nullptr, &size);
char *vendor=new char[size];
errNum=clGetPlatformInfo(current_plat, CL_PLATFORM_VENDOR, size, vendor, 0);
assert(errNum == CL_SUCCESS);
std::cout<<"平台开发商 "<<vendor<<"\n";

delete[] vendor;
//平台扩展信息
errNum=clGetPlatformInfo(current_plat, CL_PLATFORM_EXTENSIONS, 0, nullptr, &size);
char *extension=new char[size];
errNum=clGetPlatformInfo(current_plat, CL_PLATFORM_EXTENSIONS, size, extension, 0);
assert(errNum == CL_SUCCESS);
std::cout<<"平台扩展信息 "<<extension<<"\n\n";

delete[] extension;

//选定平台的独一设备
cl_uint num_devices;
cl_device_id *devices=nullptr;

errNum=clGetDeviceIDs(current_plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
if (num_devices < 1) 
    std::cout<<"平台无可用设备\n";
else {
    std::cout<<"查询到可用设备数目 "<<num_devices<<"\n";
    devices=new cl_device_id[num_devices];
    errNum=clGetDeviceIDs(current_plat, CL_DEVICE_TYPE_GPU, num_devices, devices, 0);
}
assert(errNum == CL_SUCCESS);

//当前设备
cl_device_id current_dev=devices[0];
delete[] platforms;
delete[] devices;

//查询设备信息
//风格一致 先申请 再分配内存后传递
cl_device_type *type;
errNum=clGetDeviceInfo(current_dev, CL_DEVICE_TYPE, 0, nullptr, &size);
type=new cl_device_type[size];
errNum=clGetDeviceInfo(current_dev, CL_DEVICE_TYPE, size, type, 0);
assert(errNum == CL_SUCCESS);
std::cout<<"设备类型 "<<*type<<"\n";

delete[] type;
char *dev_vendor;
errNum=clGetDeviceInfo(current_dev, CL_DEVICE_VENDOR, 0, nullptr, &size);
dev_vendor=new char[size];
errNum=clGetDeviceInfo(current_dev, CL_DEVICE_VENDOR, size, dev_vendor, 0);
assert(errNum == CL_SUCCESS);
std::cout<<"设备开发商 "<<dev_vendor<<"\n";

delete[] dev_vendor;
cl_uint *blocks;
errNum=clGetDeviceInfo(current_dev, CL_DEVICE_MAX_COMPUTE_UNITS, 0, nullptr, &size);
blocks=new cl_uint[size];
errNum=clGetDeviceInfo(current_dev, CL_DEVICE_MAX_COMPUTE_UNITS, size, blocks, 0);
assert(errNum == CL_SUCCESS);
std::cout<<"最大并行线程块 "<<*blocks<<"\n";

delete[] blocks;
cl_uint *cores;
errNum=clGetDeviceInfo(current_dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, 0, nullptr, &size);
cores=new cl_uint[size];
errNum=clGetDeviceInfo(current_dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, size, cores, 0);
assert(errNum == CL_SUCCESS);
std::cout<<"线程块最大并行核心 "<<*cores<<"\n\n";

delete[] cores;
//省略其他信息

//
//根据现有的平台与设备创建上下文 设备维护独有的context对象 后者可并发执行队列
//上下文属性设定
const cl_context_properties context_proper[]={CL_CONTEXT_PLATFORM, (cl_context_properties)current_plat, 0}; //定义上下文属性

cl_context context=clCreateContext(context_proper, 1, &current_dev, nullptr, nullptr, &errNum); //context类型与设备一致 
assert(errNum == CL_SUCCESS);

//查看上下文信息
//clGetContextInfo

//
//异步内核函数执行流程
//队列属性 启用性能分析
cl_queue_properties queue_proper[]={CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0}; 

//创建命令队列
cl_command_queue queue=clCreateCommandQueueWithProperties(context, current_dev, queue_proper, &errNum); //启用性能分析
assert(errNum == CL_SUCCESS);

//hos内存
std::array<float, 1024> hos_data_arr;
std::fill(hos_data_arr.begin(), hos_data_arr.end(), 3.14f);
//创建device buffer
//原地计算只创建单个缓存区
cl_mem dev_buffers;
std::size_t dev_buffers_size=hos_data_arr.size() * sizeof(float);

dev_buffers=clCreateBuffer
//缓存可读可写类型 
(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dev_buffers_size, hos_data_arr.data(), &errNum);
assert(errNum == CL_SUCCESS);

//异步复制并记录事件
cl_event start, end;

//0偏移量 字节大小 主机数据指针 等待事件数量 等待事件数组 异步完成记录事件
errNum=clEnqueueWriteBuffer
(queue, dev_buffers, CL_NON_BLOCKING, 0, dev_buffers_size, hos_data_arr.data(), 0, nullptr, &start);
assert(errNum == CL_SUCCESS);
//队列同步 事件数量为1
//clEnqueueWaitForEvents(queue, 1, &start);
//事件同步 阻塞式
clWaitForEvents(1, &start);
//获取内存对象信息
//clGetMemObjectInfo()

//加载cl文件
std::ifstream file("/codes/openclsamples/kernel.cl", std::ios::binary);
assert(file.is_open());
std::cout<<"打开cl文件 /codes/openclsamples/kernel.cl"<<"\n";

file.seekg(0, std::ios::end);
std::size_t filesize=file.tellg();
file.seekg(0, std::ios::beg);

std::vector<char> file_vec(filesize);

file.read(file_vec.data(), filesize);
assert(file_vec.size() == filesize);
file.close();

//根据二进制文件注册程序对象
//作为容器储存核函数代码
cl_program program;
const char *program_src=file_vec.data();
program=clCreateProgramWithSource(context, 1, &program_src, &filesize, &errNum);
assert(errNum == CL_SUCCESS);

//构建program
//程序对象可关联到多个设备
errNum=clBuildProgram(program, 1, &current_dev, nullptr, nullptr, nullptr);
assert(errNum == CL_SUCCESS);
//是否成功构建
errNum=clGetProgramBuildInfo(program, current_dev, CL_PROGRAM_BUILD_STATUS, 0, nullptr, &size);
char *build_status=new char[size];
errNum=clGetProgramBuildInfo(program, current_dev, CL_PROGRAM_BUILD_STATUS, size, build_status, nullptr);

assert(errNum == CL_SUCCESS);
std::cout<<"program对象已建立 "<<build_status<<"\n";

//查看program信息
//clGetProgramInfo()

//检查program内核函数数量
cl_uint num_kernels;
clCreateKernelsInProgram(program, 0, nullptr, &num_kernels);
assert(num_kernels > 0);

//根据program注册内核对象
cl_kernel kernel_square=clCreateKernel(program, "square",  &errNum);
assert(errNum == CL_SUCCESS);

//查看kernel对象信息
//clGetKernelInfo()

//设置核函数对象参数 
//注意！内核对象参数设置是永久性的 因此不保证线程安全 多个线程不能对同一个cl_kernel设置参数
//索引 内存参数对象大小 内存地址
errNum=clSetKernelArg(kernel_square, 0, sizeof(cl_mem), &dev_buffers);
assert(errNum == CL_SUCCESS);

errNum=clSetKernelArg(kernel_square, 1, sizeof(float), nullptr);
assert(errNum == CL_SUCCESS);

//设置工作组 项 
std::size_t work_cores=hos_data_arr.size();
std::size_t work_item=32;
//向上取整
std::size_t work_group=(work_cores+work_item-1)/work_item * work_item;
//保证group x iteam > 元素数量
//group对应cuda cores all
//item对应cuda blocks

//启动
//队列 核对象 工作维度 偏移量 线程工作组 项 等待事件  启动完成标志事件
errNum=clEnqueueNDRangeKernel
(queue, kernel_square, 1, 0, &work_group, &work_item, 0, nullptr, &end);
if (errNum == CL_SUCCESS)
    std::cout<<"\nkernel square 核函数已执行\n";
else {
    std::cerr<<"kernel square 核函数失败\n";
    std::cerr<<"errNum="<<errNum<<"\n";
    exit(EXIT_FAILURE);
}

clWaitForEvents(1, &end);
//clFinish(queue);

//计算时间
cl_ulong time_s, time_e;
clGetEventProfilingInfo(start, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_s, nullptr);
clGetEventProfilingInfo(end, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_e, nullptr);
std::cout<<"\nkernel square用时 "<<time_s - time_e<<" ns\n";

clEnqueueWaitForEvents(queue, 1, &end);
errNum=clEnqueueReadBuffer
(queue, dev_buffers, CL_NON_BLOCKING, 0, dev_buffers_size, hos_data_arr.data(), 0, nullptr, nullptr);

clFinish(queue);
assert(hos_data_arr.size() == 1024);

clReleaseEvent(start);
clReleaseEvent(end);
//注意顺序
clReleaseMemObject(dev_buffers);
clReleaseKernel(kernel_square);
clReleaseProgram(program);
clReleaseCommandQueue(queue);

clReleaseContext(context);

return 0;

}