__kernel void square
//全局内存 本地内存
//1024flo  1flo
(global float *glo_datas, local float *loc_datas) 
{
    //一维工作全局索引
    size_t loc_size=get_local_size(0);
    size_t loc_idx=get_local_id(0);
    size_t gro_idx=get_group_id(0);
    size_t glo_idx=loc_size * gro_idx + loc_idx;
    if (glo_idx >= 1024)
        return ;
    barrier(CLK_LOCAL_MEM_FENCE);

    float square=glo_datas[glo_idx] * glo_datas[glo_idx]; 
    //barrier(CLK_GLOBAL_MEM_FENCE);
    
    //原子交换
    atomic_xchg(loc_datas, square);
    printf("%f ", loc_datas[0]);
}

//引用机制 保证多线程访问资源有效性
//引用自增 
//封装后线程内自增自减 管理对象生命周期
//clRetainCommandQueue 
//clRetainMemObject

//clRetainProgram
//clRetainKernel