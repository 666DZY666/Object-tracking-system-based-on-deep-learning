/* DaSiamRPNCaffe2
# Licensed under The MIT License
# Written by wzq*/
#ifndef DASIAMRPNTRACKER_H
#define DASIAMRPNTRACKER_H
#include <array>
#include <vector>
#include <math.h>
#include <iostream>
#include <caffe2/core/init.h>
#include <caffe2/core/context.h>
#include <caffe2/core/context_gpu.h>
#include <caffe2/utils/proto_utils.h>
#include <caffe2/core/predictor.h>
#include <opencv2/opencv.hpp>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#define DEBUG_ 0
static std::string global_init_net_file = "/home/nvidia/Develop/Project/ROS/Tracking/src/tracking/DaSiamRPN/Model/global_init_net.pb";
static std::string temple_net_file = "/home/nvidia/Develop/Project/ROS/Tracking/src/tracking/DaSiamRPN/Model/temple_pred_net.pb";
static std::string track_net_file = "/home/nvidia/Develop/Project/ROS/Tracking/src/tracking/DaSiamRPN/Model/track_pred_net.pb";

static std::string adjust_init_net_file = "/home/nvidia/Develop/Project/ROS/Tracking/src/tracking/DaSiamRPN/Model/adjust_init_net.pb";
static std::string adjust_pred_net_file = "/home/nvidia/Develop/Project/ROS/Tracking/src/tracking/DaSiamRPN/Model/adjust_pred_net.pb";

static std::string Correlation_init_net_file = "/home/nvidia/Develop/Project/ROS/Tracking/src/tracking/DaSiamRPN/Model/Correlation_init_net.pb";
static std::string Correlation_pred_net_file = "/home/nvidia/Develop/Project/ROS/Tracking/src/tracking/DaSiamRPN/Model/Correlation_pred_net.pb";

struct TrackerConfig{
// These are the default hyper-params for DaSiamRPN 0.3827
   std::string windowing = "cosine";// # to penalize large displacements [cosine/uniform]
   Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> window;
   // Params from the network architecture, have to be consistent with the training
   int exemplar_size = 127;  // input z size
   int instance_size = 271;  // input x size (search region)
   int total_stride = 8;
   int score_size = (instance_size-exemplar_size)/total_stride+1;
   float context_amount = 0.5;  // context amount for the exemplar
   Eigen::Array<float, 5, 1> ratios;
   Eigen::Array<float, 1, 1> scales;
   int anchor_num;
   Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> anchor;
   float penalty_k = 0.055;
   float window_influence = 0.42;
   float lr = 0.295;
   TrackerConfig(){
        ratios <<0.33,0.5,1,2,3;
        scales << 8;
        anchor_num = ratios.size()*scales.size();
   }
};
struct BoxInfo{
    float xc;
    float yc;
    float w;
    float h;
    float best_score;
};

struct TrackInfo{
    int im_h;
    int im_w;
    BoxInfo binfo;
    std::string window;
    TrackerConfig cfg;
    cv::Scalar avg_chans;
};


using namespace caffe2;
class DaSiamRPNTracker
{
public:
    DaSiamRPNTracker();
    void SiamRPN_init(const cv::Mat& mat,BoxInfo&info,TrackInfo& output,const std::string&global_init_net_file,
                      const std::string temple_net_file,const std::string& track_net_file,caffe2::DeviceType d,int gpuid);
    void tracker_eval(cv::Mat x_crop,
                      BoxInfo &binfo,
                      float& scale_z,
                      TrackerConfig &p);
    void SiamRPN_track(cv::Mat mat,TrackInfo&info);

private:
    std::unique_ptr<caffe2::NetDef> global_init_net;
    std::unique_ptr<caffe2::NetDef> temple_net;
    std::unique_ptr<caffe2::NetDef> track_net;
    std::unique_ptr<caffe2::NetDef> adjust_net;
    std::unique_ptr<caffe2::NetDef> adjsut_init_net;
    std::shared_ptr<caffe2::Workspace> trackerEngine;
    caffe2::DeviceType mode;
    int gpuid;
    caffe2::DeviceOption devOption;
    TrackInfo trackInfo;
    //for gpu inference
    std::vector<caffe2::TensorCUDA*>cuda_r1_kernels;
    std::vector<caffe2::TensorCUDA*>cuda_cls_kernels;
   //for cpu inference
    std::vector<caffe2::TensorCPU*>cpu_r1_kernels;
    std::vector<caffe2::TensorCPU*>cpu_cls_kernels;
protected:
    //these functions are only called internal, set them protected.
    void GetTensorToHost(const Tensor<CUDAContext>* tensor,std::vector<float>& data ) {
        data.resize(tensor->size());
        CUDAContext context_;
        context_.template Copy<float,CUDAContext,CPUContext>(
                    data.size(),tensor->template data<float>(),data.data());
    }

    void GetTensorToHost( Tensor<CUDAContext>* tensor, Tensor<CPUContext>* ctensor ) {
        ctensor->Resize(tensor->dims());
        CUDAContext context_;
        context_.template Copy<float,CUDAContext,CPUContext>(
                    ctensor->size(),tensor->template data<float>(),ctensor->template mutable_data<float>());
    }

    void FeedInputToNet(cv::Mat& input,const std::string& blob_name);
    void regress_adjust();
};

#endif // DASIAMRPNTRACKER_H
