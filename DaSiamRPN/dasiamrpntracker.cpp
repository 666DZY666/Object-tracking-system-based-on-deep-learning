/* DaSiamRPNCaffe2
# Licensed under The MIT License
# Written by wzq*/
#include "dasiamrpntracker.h"
#include <caffe2/core/tensor.h>
#include <caffe2/utils/math.h>


namespace{
cv::Mat get_subwindow_tracking(cv::Mat im,
                            TrackInfo& info,float s_z,int instance_size)
{
    auto sz = s_z;
    int imw = im.cols;
    int imh = im.rows;
    float c = (sz + 1) / 2;
    float context_xmin = round(info.binfo.xc - c);
    float context_xmax = context_xmin + sz -1;
    float context_ymin = round(info.binfo.yc - c);
    float context_ymax = context_ymin + sz -1;
    float left_pad = int(std::max<float>(0., -context_xmin));
    float top_pad  = int(std::max<float>(0., -context_ymin));
    float right_pad = int(std::max<float>(0., context_xmax-imw + 1));
    float bottom_pad = int(std::max<float>(0., context_ymax - imh + 1));

    context_xmin = context_xmin + left_pad;
    context_xmax = context_xmax + left_pad;
    context_ymin = context_ymin + top_pad;
    context_ymax = context_ymax + top_pad;
    auto r = im.rows;
    c = im.cols;
    auto k = im.channels();
    cv::Mat im_patch_original;
    if(!(top_pad == 0 && bottom_pad ==0 && left_pad == 0 && right_pad == 0))
    {
        cv::Mat te_im = cv::Mat::zeros(cv::Size(c + left_pad + right_pad,r + top_pad + bottom_pad), CV_8UC3);

        im.copyTo(te_im(cv::Rect(cv::Point(left_pad,top_pad),
                       cv::Point(left_pad + c,top_pad + r))));
        if(top_pad){
            te_im(cv::Rect(cv::Point(left_pad,0),
                  cv::Point(left_pad + c, top_pad))).setTo(info.avg_chans);
        }
        if(bottom_pad)
            te_im(cv::Rect(cv::Point(left_pad,r+top_pad),
                           cv::Point(left_pad+c,r+top_pad + bottom_pad))).setTo(info.avg_chans);
        if(left_pad)
            te_im(cv::Rect(cv::Point(0,0),
                           cv::Point(left_pad,r+top_pad + bottom_pad))).setTo(info.avg_chans);
        if(right_pad)
            te_im(cv::Rect(cv::Point(c+left_pad,0),
                           cv::Point(c+left_pad + right_pad, r+top_pad + bottom_pad))).setTo(info.avg_chans);
        im_patch_original = te_im(
                    cv::Rect(cv::Point(context_xmin,context_ymin),
                             cv::Point(context_xmax+1,context_ymax+1)));
#if DEBUG_
        cv::Mat matf;
        te_im.convertTo(matf,CV_32FC3);
        std::vector<cv::Mat> channels(3);
        cv::split(matf,channels);
        std::vector<float> data;
        for(auto& c: channels) {
            data.insert(data.end(),(float*)c.datastart,(float*)c.dataend);
        }
        std::cout << std::endl;
        cv::Mat matff;
        im_patch_original.convertTo(matff,CV_32FC3);
        std::vector<cv::Mat> channelsf(3);
        cv::split(matff,channelsf);
        std::vector<float> dataf;
        for(auto& c: channelsf) {
            dataf.insert(dataf.end(),(float*)c.datastart,(float*)c.dataend);
        }
        std::cout << std::endl;
        //    std::cout << caffe2::EigenMatrixMap<float>(roi.data(),feature_out->size(),1) << std::endl;
#endif
    }else
    {
        im_patch_original =   im(cv::Rect(cv::Point(int(context_xmin),
                                                         int(context_ymin)),
                                               cv::Point(int(context_xmax+1),
                                                         int(context_ymax+1))));
    }
    cv::Mat im_patch;

    if(instance_size != s_z)
        cv::resize(im_patch_original,im_patch,cv::Size(instance_size,instance_size));
    else
        im_patch = im_patch_original;
    cv::imwrite("/opt/im_patch_original.jpg",im_patch_original);
    cv::imwrite("/opt/im_patch.jpg",im_patch);
    return im_patch;
}

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor>
generate_anchor(int total_stride,
                const Eigen::Array<float, 1, 1>&scales,
                const Eigen::Array<float, 5, 1>&ratios,
                int score_size)
{
    int anchor_num = ratios.size()*scales.size();
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> anchor,anchor_wws,anchor_hhs;
    anchor.setZero(anchor_num,4);
    anchor_wws.setZero(anchor_num,1);
    anchor_hhs.setZero(anchor_num,1);
    int size = total_stride * total_stride;
    int count = 0;
    for(int i = 0;i < ratios.size();i ++){
        auto ratio = ratios(i,0);
        int ws = int(sqrt(size/ratio));
        int hs = int(ws*ratio);
        for(int j = 0;j < scales.size();j ++){
            auto scale = scales(j,0);
            auto wws = ws * scale;
            auto hhs = hs * scale;
            anchor(count,0) = 0;
            anchor(count,1) = 0;
            anchor(count,2) = wws;
            anchor(count,3) = hhs;
            anchor_wws(count,0) = wws;
            anchor_hhs(count,0) = hhs;
            count +=1;
        }
    }
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> anchorout_wws = anchor_wws.replicate(1,score_size*score_size);
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> anchorouot_hhs = anchor_hhs.replicate(1,score_size*score_size);

    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> anchorout = anchor.replicate(score_size*score_size,1);
    anchorout.resize(anchorout.size()/4,4);
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> dx,dy;
    dx.resize(1,score_size);
    dy.resize(1,score_size);

    int ori = -(score_size/2) * total_stride;
    for(int i = 0;i < score_size;i ++){
        dx(0,i) = ori + total_stride * i;
        dy(0,i) = ori + total_stride * i;
    }

    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> xx = dx.replicate(1,score_size);
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> yy = dy.replicate(score_size,1);
    xx.resize(1,xx.size());
    yy.resize(1,yy.size());
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> xxout = xx.replicate(1,anchor_num);
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> yyout = yy.replicate(1,anchor_num);
    float * xxdata = xxout.data();
    float * yydata = yyout.data();
    float * wwdata = anchorout_wws.data();
    float * hhdata = anchorouot_hhs.data();
    Eigen::VectorXf xxvec,yyvec,wwvec,hhvec;
    xxvec.resize(xxout.size());
    yyvec.resize(yyout.size());
    wwvec.resize(yyout.size());
    hhvec.resize(yyout.size());
    for(int i = 0;i < xxvec.size();i ++){
        xxvec(i) = xxdata[i];
        yyvec(i) = yydata[i];
        wwvec(i) = wwdata[i];
        hhvec(i) = hhdata[i];
    }

    anchorout.col(0) = xxvec;
    anchorout.col(1) = yyvec;
    anchorout.col(2) = wwvec;
    anchorout.col(3) = hhvec;
//    std::cout << anchorout << std::endl;
    return anchorout;
}

std::vector<float> hanning(int M){
    std::vector<float>w(M);
    for(int i = 0;i < M;i ++){
        w[i] = (0.5-0.5*cos((2*3.1415926*i)/(M-1)));
        if(w[i] < 0.00000001)
            w[i] =0;
    }
    return w;
}

template<typename T>
std::vector<int> argsort(const vector<T>& a){
    int Len = a.size();
    std::vector<int>idx(Len,0);
    for(int i = 0; i < Len; i++){
        idx[i] = i;
    }
    std::sort(idx.begin(), idx.end(), [&a](int i1, int i2){return a[i1]> a[i2];});
    return idx;
}

}


DaSiamRPNTracker::DaSiamRPNTracker()
{
    trackerEngine = std::make_shared<caffe2::Workspace>(new Workspace("DaSiamRPN"));
    this->global_init_net = std::unique_ptr<caffe2::NetDef>(new caffe2::NetDef());
    this->temple_net      = std::unique_ptr<caffe2::NetDef>(new caffe2::NetDef());
    this->track_net       = std::unique_ptr<caffe2::NetDef>(new caffe2::NetDef());

    this->adjsut_init_net = std::unique_ptr<caffe2::NetDef>(new caffe2::NetDef);
    this->adjust_net      = std::unique_ptr<caffe2::NetDef>(new caffe2::NetDef());
}

void DaSiamRPNTracker::SiamRPN_init(const cv::Mat &mat,
                                    BoxInfo &info,
                                    TrackInfo& output,const std::string&global_init_net_file,
                                    const std::string temple_net_file,const std::string& track_net_file,
                                    caffe2::DeviceType d,
                                    int gpuid)
{
    this->mode = d;
    this->gpuid = gpuid;
    //this->devOption.set_device_type(this->mode);
    //this->devOption.set_cuda_gpu_id(this->gpuid);
    CAFFE_ENFORCE(ReadProtoFromFile(global_init_net_file,global_init_net.get()));
    CAFFE_ENFORCE(ReadProtoFromFile(temple_net_file,temple_net.get()));
    CAFFE_ENFORCE(ReadProtoFromFile(track_net_file,track_net.get()));
    CAFFE_ENFORCE(ReadProtoFromFile(adjust_pred_net_file,adjust_net.get()));
    CAFFE_ENFORCE(ReadProtoFromFile(adjust_init_net_file,adjsut_init_net.get()));


    if(this->mode == caffe2::CUDA){
        global_init_net->mutable_device_option()->set_device_type(caffe2::CUDA);
        temple_net->mutable_device_option()->set_device_type(caffe2::CUDA);
        track_net->mutable_device_option()->set_device_type(caffe2::CUDA);
        adjust_net->mutable_device_option()->set_device_type(caffe2::CUDA);
        adjsut_init_net->mutable_device_option()->set_device_type(caffe2::CUDA);

        global_init_net->mutable_device_option()->set_cuda_gpu_id(gpuid);
        temple_net->mutable_device_option()->set_cuda_gpu_id(gpuid);
        track_net->mutable_device_option()->set_cuda_gpu_id(gpuid);
        adjust_net->mutable_device_option()->set_cuda_gpu_id(gpuid);
        adjsut_init_net->mutable_device_option()->set_cuda_gpu_id(gpuid);
    }else if(this->mode == caffe2::CPU){
        global_init_net->mutable_device_option()->set_device_type(caffe2::CPU);
        temple_net->mutable_device_option()->set_device_type(caffe2::CPU);
        track_net->mutable_device_option()->set_device_type(caffe2::CPU);
        adjust_net->mutable_device_option()->set_device_type(caffe2::CPU);
        adjsut_init_net->mutable_device_option()->set_cuda_gpu_id(caffe2::CPU);
    }else{
        LOG(FATAL) << "Not support mode config:" << this->mode
                   << ", support mode:caffe2::CUDA caffe2::CPU.";
    }
    this->trackerEngine->RunNetOnce(*global_init_net.get());
    this->trackerEngine->RunNetOnce(*adjsut_init_net.get());
    this->trackerEngine->CreateNet(*temple_net.get());
    this->trackerEngine->CreateNet(*track_net.get());
    this->trackerEngine->CreateNet(*adjust_net.get());


    output.im_h = mat.rows;
    output.im_w = mat.cols;
    if(((info.h*info.w)/float(output.im_h*output.im_w)) < 0.004)
        output.cfg.instance_size = 287;
    else
        output.cfg.instance_size = 271;

    output.cfg.score_size = ((output.cfg.instance_size - output.cfg.exemplar_size)/output.cfg.total_stride + 1);
    output.cfg.anchor = generate_anchor(output.cfg.total_stride,
                                        output.cfg.scales,output.cfg.ratios,
                                        output.cfg.score_size);
//    std::cout << output.cfg.anchor;
//    std::cout << output.cfg.anchor.col(0);
    output.avg_chans = cv::mean(mat);
    int wc_z = (info.w + output.cfg.context_amount*(info.h+info.w));
    int hc_z = (info.h + output.cfg.context_amount*(info.h+info.w));
    int s_z = round(sqrt(wc_z*hc_z));
    cv::Mat roi = get_subwindow_tracking(mat,output,s_z,output.cfg.exemplar_size);





    //switch z_crop (3,127,127) to z (1,3,127,127)
    this->FeedInputToNet(roi,"data");
    //run temple net(z as input)
    this->trackerEngine->RunNet(this->temple_net->name());
#if DEBUG_
    cv::Mat matf;
    mat.convertTo(matf,CV_32FC3);
    std::vector<cv::Mat> channels(3);
    cv::split(matf,channels);
    std::vector<float> data;
    for(auto& c: channels) {
        data.insert(data.end(),(float*)c.datastart,(float*)c.dataend);
    }
    cv::imwrite("/opt/dcpp.jpg",roi);
    std::cout << this->temple_net->DebugString() << std::endl;
    caffe2::Tensor* feature_out = new caffe2::Tensor();
    this->GetTensorToHost(this->trackerEngine->GetBlob("feature_out")->template GetMutable<caffe2::Tensor>(),feature_out);
    TensorPrinter tpter("data","/opt/data.blob",100000000000);
    tpter.Print<float>(*feature_out);
//    std::cout << caffe2::EigenMatrixMap<float>(roi.data(),feature_out->size(),1) << std::endl;
#endif
    if(this->mode == caffe2::CUDA){
        std::string ss = "conv_r1";
        auto conv_r1 = this->trackerEngine->GetBlob(ss)->template GetMutable<caffe2::Tensor>();
        auto kernel_size = int(sqrt(conv_r1->size()/(4*512*this->trackInfo.cfg.anchor_num)));
        ss = "conv_cls1";
        auto conv_cls1 = this->trackerEngine->GetBlob(ss)->template GetMutable<caffe2::Tensor>();

        auto input = this->trackerEngine->CreateBlob("r1_kernel")->GetMutable<caffe2::TensorCUDA>();
        input->Resize(this->trackInfo.cfg.anchor_num*4,
                                                  512,kernel_size,kernel_size);
        caffe2::CUDAContext ctx;
        ctx.template Copy<float,CPUContext,CUDAContext>(conv_r1->size(),
                                                        conv_r1->template data<float>(),
                                                        input->template mutable_data<float>());

        auto input_cls = this->trackerEngine->CreateBlob("cls1_kernel")->GetMutable<caffe2::TensorCUDA>();
        input_cls->Resize(this->trackInfo.cfg.anchor_num*2,
                                                        512,kernel_size,kernel_size);
        ctx.template Copy<float,CPUContext,CUDAContext>(conv_cls1->size(),
                                                        conv_cls1->template data<float>(),
                                                        input_cls->template mutable_data<float>());
    }else{
        std::string ss = "conv_r1";
        auto conv_r1 = this->trackerEngine->GetBlob(ss)->template GetMutable<caffe2::Tensor>();
        auto kernel_size = int(sqrt(conv_r1->size()/(4*512*this->trackInfo.cfg.anchor_num)));
        ss = "conv_cls1";
        auto conv_cls1 = this->trackerEngine->GetBlob(ss)->template GetMutable<caffe2::Tensor>();

        auto input = this->trackerEngine->CreateBlob("r1_kernel")->GetMutable<caffe2::TensorCPU>();
        input->Resize(this->trackInfo.cfg.anchor_num*4,
                                                  512,kernel_size,kernel_size);
        caffe2::CUDAContext ctx;
        ctx.template Copy<float,CPUContext,CPUContext>(conv_r1->size(),
                                                        conv_r1->template data<float>(),
                                                        input->template mutable_data<float>());

        auto input_cls = this->trackerEngine->CreateBlob("cls1_kernel")->GetMutable<caffe2::TensorCPU>();
        input_cls->Resize(this->trackInfo.cfg.anchor_num*2,
                                                        512,kernel_size,kernel_size);
        ctx.template Copy<float,CPUContext,CPUContext>(conv_cls1->size(),
                                                        conv_cls1->template data<float>(),
                                                        input_cls->template mutable_data<float>());
    }

#if DEBUG_
    TensorPrinter tpter1("data","/opt/r1_kernel.blob",100000000000);
    tpter1.Print<float>(*r1_kernel);

    TensorPrinter tpter2("data","/opt/cls1_kernel.blob",100000000000);
    tpter2.Print<float>(*cls1_kernel);

    caffe2::TensorCPU* r2_out_w = new caffe2::TensorCPU();
    this->GetTensorToHost(this->trackerEngine->GetBlob("r2_out_w")->template GetMutable<caffe2::Tensor>(),r2_out_w);
    caffe2::TensorCPU* r2_out_b = new caffe2::TensorCPU();
    this->GetTensorToHost(this->trackerEngine->GetBlob("r2_out_b")->template GetMutable<caffe2::Tensor>(),r2_out_b);
    TensorPrinter tpter3("data","/opt/r2_out_w.blob",100000000000);
    tpter3.Print<float>(*r2_out_w);

    TensorPrinter tpter4("data","/opt/r2_out_b.blob",100000000000);
    tpter4.Print<float>(*r2_out_b);
#endif
    int score_size = output.cfg.score_size;
    if(output.cfg.windowing == "cosine")
    {

         Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>tmp = (EigenMatrixMap<float>(hanning(score_size).data(),score_size,1)
                             *
                             EigenMatrixMap<float>(hanning(score_size).data(),1,score_size));
         tmp.resize(1,score_size*score_size);
         output.cfg.window = tmp.replicate(1,output.cfg.anchor_num);
    }
    else if(output.cfg.windowing == "uniform"){
         Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>tmp;
         tmp.setOnes(1,score_size*score_size);
         output.cfg.window = tmp.replicate(1,output.cfg.anchor_num);
    }

    output.binfo = info;
    this->trackInfo = output;
}

/*BoxInfo */void DaSiamRPNTracker::tracker_eval(cv::Mat x_crop,
                                                BoxInfo& binfo,
                                    float& scale_z,
                                                TrackerConfig &p)
{
    //run track net (x_crop as input)
//    time_t firstcall;
//    time(&firstcall);
    this->FeedInputToNet(x_crop,"data");
    this->trackerEngine->RunNet(this->track_net->name());
    //get the output delta and score
    this->regress_adjust();
//    time_t lastcall;
//    time(&lastcall);
//    float duration = difftime(lastcall,firstcall);
//    LOG(INFO) << "duration:" << duration << " seconds";
    //softmax score

//#if 0
    caffe2::Argument arg;
    arg.set_name("axes");
    arg.add_ints(1);
    arg.add_ints(2);
    arg.add_ints(3);
    arg.add_ints(0);
    static auto permute_op = caffe2::CreateOperatorDef("Transpose",
                                                  "permute",
                                                  {"r2_out"},{"r2_permute_out"},{arg},this->devOption);
    this->trackerEngine->RunOperatorOnce(permute_op);
    static caffe2::TensorCPU* r2_premute = new caffe2::TensorCPU();
    static caffe2::TensorCPU* score_res =new caffe2::TensorCPU();
    if(this->mode == caffe2::CUDA){
        caffe2::TensorCUDA* r2_premute_CUDA = this->trackerEngine->GetBlob("r2_permute_out")->template GetMutable<caffe2::Tensor>();
        this->GetTensorToHost(r2_premute_CUDA,r2_premute);
        r2_premute->Reshape(std::vector<int>{4,r2_premute->size()/4});
        static auto cls2_op = caffe2::CreateOperatorDef("Transpose",
                                                      "cls2_permute",
                                                      {"cls_out"},{"cls2_permute_out"},{arg},this->devOption);
        this->trackerEngine->RunOperatorOnce(cls2_op);
        caffe2::TensorCUDA* cls2_premute = this->trackerEngine->GetBlob("cls2_permute_out")->template GetMutable<caffe2::Tensor>();
        cls2_premute->Reshape(std::vector<int>{2,cls2_premute->size()/2});
        caffe2::Argument arg_transpose;
        arg_transpose.add_ints(1);
        arg_transpose.add_ints(0);
        static auto cls2_transpose_op = caffe2::CreateOperatorDef("Transpose",
                                                      "cls2_tranpose",
                                                      {"cls2_permute_out"},{"cls2_tranpose_out"},{arg_transpose},this->devOption);
        this->trackerEngine->RunOperatorOnce(cls2_transpose_op);
        caffe2::Argument arg_soft;
        arg_soft.set_name("axis");
        arg_soft.set_i(1);
        static auto softmax_op = caffe2::CreateOperatorDef("Softmax",
                                                           "cls_soft",
                                                            {"cls2_tranpose_out"},
                                                            {"cls2_softmax_out"},{arg_soft},this->devOption);
        this->trackerEngine->RunOperatorOnce(softmax_op);
        static auto softmax_transpose_op = caffe2::CreateOperatorDef("Transpose",
                                                      "softmax_tranpose",
                                                      {"cls2_softmax_out"},{"softmax_tranpose_out"},{arg_transpose},this->devOption);
        this->trackerEngine->RunOperatorOnce(softmax_transpose_op);
        caffe2::TensorCUDA* score_res_CUDA = this->trackerEngine->GetBlob("softmax_tranpose_out")->template GetMutable<caffe2::Tensor>();

        this->GetTensorToHost(score_res_CUDA,score_res);
    }else{
        r2_premute = this->trackerEngine->GetBlob("r2_permute_out")->template GetMutable<caffe2::Tensor>();
        r2_premute->Reshape(std::vector<int>{4,r2_premute->size()/4});
        static auto cls2_op = caffe2::CreateOperatorDef("Transpose",
                                                      "cls2_permute",
                                                      {"cls_out"},{"cls2_permute_out"},{arg},this->devOption);
        this->trackerEngine->RunOperatorOnce(cls2_op);
        caffe2::TensorCPU* cls2_premute = this->trackerEngine->GetBlob("cls2_permute_out")->template GetMutable<caffe2::Tensor>();
        cls2_premute->Reshape(std::vector<int>{2,cls2_premute->size()/2});
        caffe2::Argument arg_transpose;
        arg_transpose.add_ints(1);
        arg_transpose.add_ints(0);
        static auto cls2_transpose_op = caffe2::CreateOperatorDef("Transpose",
                                                      "cls2_tranpose",
                                                      {"cls2_permute_out"},{"cls2_tranpose_out"},{arg_transpose},this->devOption);
        this->trackerEngine->RunOperatorOnce(cls2_transpose_op);
        caffe2::Argument arg_soft;
        arg_soft.set_name("axis");
        arg_soft.set_i(1);
        static auto softmax_op = caffe2::CreateOperatorDef("Softmax",
                                                           "cls_soft",
                                                            {"cls2_tranpose_out"},
                                                            {"cls2_softmax_out"},{arg_soft},this->devOption);
        this->trackerEngine->RunOperatorOnce(softmax_op);
        static auto softmax_transpose_op = caffe2::CreateOperatorDef("Transpose",
                                                      "softmax_tranpose",
                                                      {"cls2_softmax_out"},{"softmax_tranpose_out"},{arg_transpose},this->devOption);
        this->trackerEngine->RunOperatorOnce(softmax_transpose_op);
        score_res = this->trackerEngine->GetBlob("softmax_tranpose_out")->template GetMutable<caffe2::Tensor>();
    }

    caffe2::CPUContext ctx;
    //delta 0
    int N = r2_premute->size()/4;
#if DEBUG_
    caffe2::TensorCPU* cls2_permute_out = new caffe2::TensorCPU();
    this->GetTensorToHost(this->trackerEngine->GetBlob("cls2_permute_out")->template GetMutable<caffe2::Tensor>(),cls2_permute_out);
    caffe2::TensorCPU* cls2_softmax_out = new caffe2::TensorCPU();
    this->GetTensorToHost(this->trackerEngine->GetBlob("softmax_tranpose_out")->template GetMutable<caffe2::Tensor>(),cls2_softmax_out);
    TensorPrinter tpter1("data","/opt/cls2_permute_out.blob",100000000000);
    tpter1.Print<float>(*cls2_permute_out);
    TensorPrinter tpter2("data","/opt/softmax_tranpose_out.blob",100000000000);
    tpter2.Print<float>(*cls2_softmax_out);
#endif
    std::vector<float>score;
    score.resize(N);
    memcpy(score.data(),score_res->mutable_data<float>() + N,sizeof(float)*N);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> col0 = this->trackInfo.cfg.anchor.col(0);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> col1 = this->trackInfo.cfg.anchor.col(1);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> col2 = this->trackInfo.cfg.anchor.col(2);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> col3 = this->trackInfo.cfg.anchor.col(3);
    caffe2::math::Mul<float,caffe2::CPUContext>(N,
                                                r2_premute->mutable_data<float>(),
                                                col2.data(),
                                                r2_premute->mutable_data<float>(),
                                                &ctx);

    caffe2::math::Add<float,caffe2::CPUContext>(N,
                                                r2_premute->mutable_data<float>(),
                                                col0.data(),
                                                r2_premute->mutable_data<float>(),
                                                &ctx);
    //delta 1
    caffe2::math::Mul<float,caffe2::CPUContext>(N,
                                                r2_premute->mutable_data<float>() + N,
                                                col3.data(),
                                                r2_premute->mutable_data<float>() + N,
                                                &ctx);

    caffe2::math::Add<float,caffe2::CPUContext>(N,
                                                r2_premute->mutable_data<float>() + N ,
                                                col1.data(),
                                                r2_premute->mutable_data<float>() + N,
                                                &ctx);
    //delta 2
    caffe2::math::Exp<float,caffe2::CPUContext>(N,
                                                r2_premute->mutable_data<float>() + N*2,
                                                r2_premute->mutable_data<float>() + N*2,
                                                &ctx);
    caffe2::math::Mul<float,caffe2::CPUContext>(N,
                                                r2_premute->mutable_data<float>() + N*2,
                                                col2.data(),
                                                r2_premute->mutable_data<float>() + N*2,
                                                &ctx);

    //delta 3
    caffe2::math::Exp<float,caffe2::CPUContext>(N,
                                                r2_premute->mutable_data<float>() + N*3,
                                                r2_premute->mutable_data<float>() + N*3,
                                                &ctx);
    caffe2::math::Mul<float,caffe2::CPUContext>(N,
                                                r2_premute->mutable_data<float>() + N*3,
                                                col3.data(),
                                                r2_premute->mutable_data<float>() + N*3,
                                                &ctx);


    //
    auto change = [&](float* r, int N,float* o){
      //use eigen map to compute maximum (r,1./r)
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> tmp;
        tmp.setOnes(1,N);
        std::vector<float>tmp1;
        tmp1.resize(N);
        caffe2::math::Div<float,caffe2::CPUContext>(N,tmp.data(),r,tmp1.data(),&ctx);
        caffe2::math::ElemwiseMax<float,caffe2::CPUContext>(N,r,tmp1.data(),o,&ctx);
//        caffe2::ElemwiseMax;
    };
    auto sz = [&](float*w,float*h,int N,float * o){
        caffe2::math::Add<float,caffe2::CPUContext>(N,w,h,o,&ctx);
        caffe2::math::Scale<float,caffe2::CPUContext>(N,0.5,o,o,&ctx);
        std::vector<float>tmp1(N),tmp2(N),tmp3(N);
        caffe2::math::Add<float,caffe2::CPUContext>(N,w,o,tmp1.data(),&ctx);
        caffe2::math::Add<float,caffe2::CPUContext>(N,h,o,tmp2.data(),&ctx);
        caffe2::math::Mul<float,caffe2::CPUContext>(N,tmp1.data(),tmp2.data(),
                                                    tmp3.data(),&ctx);
        caffe2::math::Sqrt<float,caffe2::CPUContext>(N,tmp3.data(),o,&ctx);
    };
    auto sz_wh = [](float w,float h) ->float{
        float pad = (w + h)*0.5;
        float sz2 = (w + pad) * (h + pad);
        float ret = std::sqrt(sz2);
        return ret;
    };
    std::vector<float>sz_o,chang_z_o,s_c;
    sz_o.resize(N);
    chang_z_o.resize(N);
    s_c.resize(N);
    sz(r2_premute->mutable_data<float>() + N*2,
       r2_premute->mutable_data<float>() + N*3,N,
       sz_o.data());
//    BoxInfo tmp;
//    tmp.xc = binfo.xc;
//    tmp.yc = binfo.yc;
//    tmp.h = binfo.h*scale_z;
//    tmp.w = binfo.w*scale_z;
//    change(sz_o.data(),N,chang_z_o.data());
    float sz_hw_o = sz_wh(binfo.w*scale_z,binfo.h*scale_z);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> di;
    di.setConstant(1,N,sz_hw_o);
    caffe2::math::Div<float,caffe2::CPUContext>(N,sz_o.data(),
                                                di.data(),chang_z_o.data(),&ctx);
    change(chang_z_o.data(),N,s_c.data());
    //uper is ok
    float for_rc_div = ((binfo.w*scale_z)/(binfo.h*scale_z));
    di.setConstant(1,N,for_rc_div);
    std::vector<float>raio_penalty,rat_out;
    raio_penalty.resize(N);
    rat_out.resize(N);
    caffe2::math::Div<float,caffe2::CPUContext>(N,r2_premute->mutable_data<float>() + N*2,
                                                r2_premute->mutable_data<float>() + N*3,
                                                raio_penalty.data(),&ctx);
    caffe2::math::Div<float,caffe2::CPUContext>(N,di.data(),
                                                raio_penalty.data(),rat_out.data(),&ctx);
    std::vector<float>r_c;
    r_c.resize(N);
    change(rat_out.data(),N,r_c.data());

    std::vector<float>penalty;
    penalty.resize(N);
    std::vector<float>r_c_mul_s_c,sub_res;
    r_c_mul_s_c.resize(N);
    sub_res.resize(N);
    caffe2::math::Mul<float,caffe2::CPUContext>(N,s_c.data(),r_c.data(),
                                                r_c_mul_s_c.data(),&ctx);
    di.setConstant(1,N,1.0);
    caffe2::math::Sub<float,caffe2::CPUContext>(N,r_c_mul_s_c.data(),di.data(),sub_res.data(),&ctx);
    caffe2::math::Scale<float,caffe2::CPUContext>(N,-1*p.penalty_k,sub_res.data(),sub_res.data(),&ctx);
    caffe2::math::Exp<float,caffe2::CPUContext>(N,sub_res.data(),penalty.data(),&ctx);

    std::vector<float>pscore,window_scale_influence;
    window_scale_influence.resize(N);
    pscore.resize(N);
    caffe2::math::Mul<float,caffe2::CPUContext>(N,penalty.data(),score.data(),
                                                pscore.data(),&ctx);
    caffe2::math::Scale(N,1-p.window_influence,pscore.data(),pscore.data(),&ctx);
    caffe2::math::Scale(N,p.window_influence,p.window.data(),window_scale_influence.data(),&ctx);
    caffe2::math::Add<float,caffe2::CPUContext>(N,pscore.data(),window_scale_influence.data(),
                                                pscore.data(),&ctx);
    auto sort_res = argsort<float>(pscore);
    int best_pscore_id = sort_res[0];
    std::vector<float>target(4);
    for(int i = 0;i < target.size();i ++){
        target[i] = r2_premute->mutable_data<float>()[i*N + best_pscore_id]/scale_z;
    }
//    binfo.w = tmp.w/scale_z;
//    binfo.h = tmp.h/scale_z;
    float lr = penalty[best_pscore_id] * score[best_pscore_id]*p.lr;

    float res_x = target[0] + binfo.xc;
    float res_y = target[1] + binfo.yc;

    float res_w = binfo.w*(1-lr) + target[2]*lr;
    float res_h = binfo.h*(1-lr) + target[3]*lr;

    binfo.xc = res_x;
    binfo.yc = res_y;
    binfo.w = res_w;
    binfo.h = res_h;

    binfo.best_score = score[best_pscore_id];
//#endif
}

void DaSiamRPNTracker::SiamRPN_track(cv::Mat mat, TrackInfo &info)
{
    float hc_z = info.binfo.w + info.cfg.context_amount*(info.binfo.w + info.binfo.h);
    float wc_z = info.binfo.h + info.cfg.context_amount*(info.binfo.w + info.binfo.h);
    float s_z = std::sqrt(wc_z*hc_z);
    float scale_z = info.cfg.exemplar_size/s_z;
    int d_search = (info.cfg.instance_size - info.cfg.exemplar_size)/2;
    float pad = d_search / scale_z;
    float s_x = s_z + 2 * pad;

    auto x_crop = get_subwindow_tracking(mat,
                                         info,
                                         std::round(s_x),
                                         info.cfg.instance_size);

    this->tracker_eval(x_crop,
                       info.binfo,
                       scale_z,
                       info.cfg);
    info.binfo.xc = std::max<float>(0,std::min<float>(info.im_w,info.binfo.xc));
    info.binfo.yc = std::max<float>(0,std::min<float>(info.im_h,info.binfo.yc));
    info.binfo.w = std::max<float>(0,std::min<float>(info.im_w,info.binfo.w));
    info.binfo.h = std::max<float>(0,std::min<float>(info.im_h,info.binfo.h));
    this->trackInfo = info;
}

void DaSiamRPNTracker::FeedInputToNet(cv::Mat &rgb, const std::__cxx11::string &blob_name)
{
    caffe2::CUDAContext ctx;
    rgb.convertTo(rgb,CV_32FC3);
    std::vector<cv::Mat> channels(3);
    cv::split(rgb,channels);
    std::vector<float> data;
    for(auto& c: channels) {
        data.insert(data.end(),(float*)c.datastart,(float*)c.dataend);
    }
    std::vector<TIndex>dims({1,rgb.channels(),rgb.rows,rgb.cols});
    caffe2::TensorCPU tensor(dims,data,NULL);
    this->mode == caffe2::CUDA ?
                this->trackerEngine->CreateBlob(blob_name)->GetMutable<caffe2::TensorCUDA>()->ResizeLike(tensor)
              :
                this->trackerEngine->CreateBlob(blob_name)->GetMutable<caffe2::TensorCPU>()->ResizeLike(tensor);
    this->mode == caffe2::CUDA ?
                ctx.template Copy<float,CPUContext,CUDAContext>(
                tensor.size(),tensor.template data<float>(),this->trackerEngine->CreateBlob(blob_name)->GetMutable<caffe2::TensorCUDA>()->template mutable_data<float>())
        :
          ctx.template Copy<float,CPUContext,CPUContext>(
              tensor.size(),tensor.template data<float>(),this->trackerEngine->CreateBlob(blob_name)->GetMutable<caffe2::TensorCPU>()->template mutable_data<float>());
}

void DaSiamRPNTracker::regress_adjust()
{
    caffe2::OperatorDef conv_def;
    caffe2::OperatorDef conv_def_cls;
    if(this->mode == caffe2::CUDA){
        auto r1_kernel = this->trackerEngine->GetBlob("r1_kernel")->template GetMutable<caffe2::Tensor>();
        auto cls1_kernel = this->trackerEngine->GetBlob("cls1_kernel")->template GetMutable<caffe2::Tensor>();

        caffe2::Argument arg_conv;
        arg_conv.set_name("kernel");
        arg_conv.set_i(r1_kernel->dim32(2));
        conv_def =
        caffe2::CreateOperatorDef("Conv",
                                  "reg",{"conv_r2","r1_kernel"},
                                    {"conv_conv_r2"},{arg_conv},this->devOption);

        arg_conv.set_i(cls1_kernel->dim32(2));
        conv_def_cls =
        caffe2::CreateOperatorDef("Conv",
                                  "cls",{"conv_cls2","cls1_kernel"},
                                    {"cls_out"},{arg_conv},this->devOption);
    }else{
        auto r1_kernel = this->trackerEngine->GetBlob("r1_kernel")->template GetMutable<caffe2::Tensor>();
        auto cls1_kernel = this->trackerEngine->GetBlob("cls1_kernel")->template GetMutable<caffe2::Tensor>();

        caffe2::Argument arg_conv;
        arg_conv.set_name("kernel");
        arg_conv.set_i(r1_kernel->dim32(2));
        conv_def =
        caffe2::CreateOperatorDef("Conv",
                                  "reg",{"conv_r2","r1_kernel"},
                                    {"conv_conv_r2"},{arg_conv},this->devOption);

        arg_conv.set_i(cls1_kernel->dim32(2));
        conv_def_cls =
        caffe2::CreateOperatorDef("Conv",
                                  "cls",{"conv_cls2","cls1_kernel"},
                                    {"cls_out"},{arg_conv},this->devOption);
    }


    this->trackerEngine->RunOperatorOnce(conv_def);
    this->trackerEngine->RunOperatorOnce(conv_def_cls);


    this->trackerEngine->RunNet("adjust");

#if DEBUG_
    caffe2::TensorCPU* r2_out = new caffe2::TensorCPU();
    this->GetTensorToHost(this->trackerEngine->GetBlob("r2_out")->template GetMutable<caffe2::Tensor>(),r2_out);
    caffe2::TensorCPU* cls_out = new caffe2::TensorCPU();
    this->GetTensorToHost(this->trackerEngine->GetBlob("cls_out")->template GetMutable<caffe2::Tensor>(),cls_out);
    TensorPrinter tpter1("data","/opt/r2_out.blob",100000000000);
    tpter1.Print<float>(*r2_out);
    TensorPrinter tpter2("data","/opt/cls_out.blob",100000000000);
    tpter2.Print<float>(*cls_out);
#endif

}
