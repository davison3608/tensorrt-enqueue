//将chw格式图像展平后的数据重塑为Mat对象
cv::Mat GetCHWflatten2Picture(std::vector<float>& flatten, std::vector<uint> h_w_c)
{
    uint rows=h_w_c[0];
    uint cols=h_w_c[1];
    uint channels=h_w_c[2];
    int elements=flatten.size();

    assert(h_w_c.size() == 3);
    assert(elements == channels*rows*cols);

    int type;
    switch (channels)
    {
    case 1:
        type=CV_8UC1;
        break;
    case 3:
        type=CV_8UC3;
        break;
    default:
        std::cerr<<"不支持的通道数\n";
        exit(false);
    }

    //待合并对象
    cv::Mat hwc_img(rows, cols, type);
    if (hwc_img.type() == CV_8UC3) {
        for (int r=0; r<rows; r++)
    {
            for (int l=0; l<cols; l++)
        {
            uchar red, green, blue;  
            //计算各通道索引 CHW格式
            //CHW索引 = c * H * W + h * W + w
            int ch0_idx=0 * rows * cols + r * cols + l;  
            int ch1_idx=1 * rows * cols + r * cols + l;  
            int ch2_idx=2 * rows * cols + r * cols + l;
            //归一数据到0-255  
            red=static_cast<uchar>(std::min(std::max(flatten[ch0_idx] * 255.0f, 0.0f), 255.0f));
            green=static_cast<uchar>(std::min(std::max(flatten[ch1_idx] * 255.0f, 0.0f), 255.0f));
            blue=static_cast<uchar>(std::min(std::max(flatten[ch2_idx] * 255.0f, 0.0f), 255.0f));
            //单像素像素通道
            cv::Vec3b pxel(blue, green, red);
            hwc_img.at<cv::Vec3b>(r, l)=pxel;
        }
        cv::cvtColor(hwc_img, hwc_img, cv::COLOR_BGR2RGB);
    }
    } else if (hwc_img.type() == CV_8UC1) {
            for (int r=0; r<rows; r++)
    {
            for (int l=0; l<cols; l++)
        {
            uchar x;
            //CHW索引 = c * H * W + h * W + w
            int ch0_idx=0 * rows * cols + r * cols + l;  
            //单像素
            x=static_cast<uchar>(std::min(std::max(flatten[ch0_idx] * 255.0f, 0.0f), 255.0f));
            hwc_img.at<uchar>(r, l)=x;
        }
    }
    }
    return hwc_img;
}

//这段代码进行hwc到chw格式转化 并归一化处理
void CVpre::GetCHWPictureNormalizedData(std::vector<float> &output_vec, bool normal)
{
    auto y=Proc_Img.rows;
    auto x=Proc_Img.cols;
    //创建副本
    cv::Mat Nor_img;
    this->Proc_Img.convertTo(Nor_img, CV_32F);
    //
    if (normal) 
        cv::normalize(Nor_img, Nor_img, 0, 1, cv::NORM_MINMAX, CV_32F);
    else {
        cv::Mat mean;
        cv::Mat stddev;
        cv::meanStdDev(Nor_img, mean, stddev);
        cv::subtract(Nor_img, *mean.ptr<CV_32F>(0), Nor_img);
        cv::divide(Nor_img, *stddev.ptr<CV_32F>(0), Nor_img);
    }
    //chw通道
    cv::Mat chw_nor_img(3, x * y, CV_32F);
    //
    //分离三通道到容器
    std::vector<cv::Mat> single_channel_vec(3);
    cv::split(Nor_img, single_channel_vec);
    //
        for (int idx=0; idx<3; idx++)
    {
        //展平单通道图
        auto flatten=single_channel_vec[idx].reshape(1, 1);
        auto mat=chw_nor_img.row(idx);
        assert(flatten.dims == mat.dims);
        assert(flatten.rows == mat.rows);
        assert(flatten.cols == mat.cols);
        assert(flatten.type() == mat.type());
        flatten.copyTo(chw_nor_img.row(idx));
    }
    //准备容器
    std::vector<float> chw_nor_datas;
        for (int idx=0; idx<3; idx++)
    {
        //指向每一行
        float *raw_ptr=chw_nor_img.ptr<float>(idx);
        //插入
        chw_nor_datas.insert(chw_nor_datas.end(), raw_ptr, raw_ptr + (x * y));
        //std::copy(raw_ptr, raw_ptr + (x * y), chw_nor_datas.begin() + (idx * x * y));
    }
    assert(chw_nor_datas.size() == 3 * y * x);
    //移交数据
    if (output_vec.size() != 3*y*x)
        output_vec.resize(3*y*x);
    output_vec=std::move(chw_nor_datas);
    //
    assert(output_vec.size() == 3*y*x);
    std::cerr<<"元素数目--"<<output_vec.size()<<"--";
    for (int idx=0; idx<20; idx++) 
        std::cerr<<output_vec[idx]<<" ";
    std::cerr<<"省略";
    std::cerr<<"\n";
}
