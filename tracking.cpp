#include <ros/ros.h>
#include <signal.h>
#include <geometry_msgs/Twist.h>

#include "mtcnn/face_detector.hpp"
#include "mtcnn/helpers.hpp"
#include "eco/eco.hpp"
#include "eco/parameters.hpp"
#include "kcf/kcftracker.hpp"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <math.h>

//隔帧处理
#define num 3
//中心跟踪触发框
#define L  295
#define R  355
#define U  50
#define D  400
#define W  60
#define H  350
//x方向阈值
#define t0 260
#define t1 410
//z方向阈值
#define s0      12666
#define s1      21666
#define ST0     1366
#define ST1     466
//速度阈值
#define v0      0.86    //x方向
#define v1      0.23    //Z方向
#define v2      0.01   
#define v       0.002
//加速度
#define s_scale 1.2     //启动
#define e_scale 0.76    //停止
//跟踪框调整
//#define b       0.22
//#define b0      0.4

using namespace cv;
using namespace std;
using namespace mtcnn;
using namespace ros;

Publisher cmdVelPub;

void shutdown(int sig)
{
  cmdVelPub.publish(geometry_msgs::Twist());
  ROS_INFO("\r\n\r\n!Tracking system shut down!");
  ros::shutdown();
}
    
int main(int argc, char **argv)
{
    init(argc, argv, "Tracking");
    NodeHandle node;
    cmdVelPub = node.advertise<geometry_msgs::Twist>("mobile_base/commands/velocity", 1);
    signal(SIGINT, shutdown); 
    ROS_INFO("\r\n!!!Tracking system turn on!!!\r\n");
    geometry_msgs::Twist speed;
    speed.linear.x = 0;
    speed.angular.z = 0;

    int key, flag = 0, flag1 = 0, flag2 = 0, flag3 = 1, flag4 = 0, flag5 = 0, flag6 = 0, count = 0, f = 0, n = 0;
    double timerkcf, timemtcnn, fpskcf, fpsmtcnn, bboxWidth, bboxHeight, s;
    bool okkcf;
    //int maxv = 0, maxi = 0;

    vector<mtcnn::Face> faces;
    //vector<cv::Point> pts;

    Rect2f bboxGroundtruth;
    Mat frame;
    //Mat channels[3], frame1;
    Point pointInterest, pointInterest1;
    VideoCapture capture;
    capture.open(1);
    if (!capture.isOpened())
    {
        std::cout << "Capture device failed to open!" << std::endl;
        return -1;
    }
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, 640);
    capture >> frame;

    string window_name = "Tracking";
    namedWindow(window_name, 0);
    //************************************* 創建 Trackers、Detecter ****************************************
    //MTCNN
    FaceDetector fd("/home/nvidia/Develop/Project/ROS/Tracking/src/tracking/mtcnn/model", 0.9f, 0.9f, 0.9f, false, true, 0);

    // KCF
    bool HOG = true, FIXEDWINDOW = true, MULTISCALE = true, LAB = true, DSST = false; //HOG + LAB(color)
    kcf::KCFTracker kcftracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);
    Rect2d kcfbbox((int)bboxGroundtruth.x, (int)bboxGroundtruth.y, (int)bboxGroundtruth.width, (int)bboxGroundtruth.height);
    //kcftracker.init(frame, kcfbbox);
    
    while (frame.data)
    {
        while(ros::ok()){
            key = cvWaitKey(6);

            //KCF —— HOG + LAB
            if(key == 's'){
                printf("KCF —— HOG + LAB\r\n");
                flag = 2;
                flag2 = 0;
                flag3 = 1;
                flag4 = 1;
            }
            else if(key == 'e'){
                flag = 0;
                flag2 = 0;
                flag3 = 1;
                flag4 = 0;
                flag5 = 0;
            }
            //*************************************** Tracking、Detecting******************************************
            if(f >= num){
                f = 0;
                //均衡化（RGB）
    /*	
                split(frame, channels);
	            for (int i = 0; i < 3; i++)
	            {
	               equalizeHist(channels[i], channels[i]);
	            }
	            merge(channels, 3, frame1);
    */
                //（YCrCb）
    /*      
                cvtColor(frame, frame1, COLOR_BGR2YCrCb);
	            split(frame1, channels);
	            equalizeHist(channels[0], channels[0]);
	            merge(channels, 3, frame1);
                cvtColor(frame1, frame1, COLOR_YCrCb2BGR);    
    */ 
                //**************************MTCNN****************************
                //rectangle(frame, Rect(L, U, W, H), Scalar(0, 0, 255), 3);
                if(flag3){
                    if(flag5){
                        if(n == 1){
                            faces = fd.detect(frame, 86.6f, 0.16f);
                            if(faces.size()){
                                n = faces.size();
                                bboxWidth = faces[0].bbox.x2 - faces[0].bbox.x1;
	                            bboxHeight = faces[0].bbox.y2 - faces[0].bbox.y1;

                                kcfbbox.x = faces[0].bbox.x1;
		                        kcfbbox.y = faces[0].bbox.y1;
		                        kcfbbox.width = bboxWidth;
		                        kcfbbox.height = bboxHeight;
/*
		                        kcfbbox.x = kcfbbox.x - kcfbbox.width * b;
		                        kcfbbox.y = kcfbbox.y - kcfbbox.height * b0;
		                        kcfbbox.width = kcfbbox.width + kcfbbox.width * b0;
		                        kcfbbox.height = kcfbbox.height + kcfbbox.height * b0;
*/
		                        kcftracker.init(frame, kcfbbox);
                                okkcf = kcftracker.update(frame, kcfbbox);

                                if(okkcf){
                                    flag3 = 0;
                                    flag5 = 0;
                                    flag2 = 1;
                                } 
                           }
                        }
                        else{
                            faces = fd.detect(frame, 86.6f, 0.16f);
                            if(faces.size()){
                                n = faces.size();
                                for (size_t i = 0; i < faces.size(); ++i){
                                    bboxWidth = faces[i].bbox.x2 - faces[i].bbox.x1;
	                                bboxHeight = faces[i].bbox.y2 - faces[i].bbox.y1;

                                    kcfbbox.x = faces[i].bbox.x1;
		                            kcfbbox.y = faces[i].bbox.y1;
		                            kcfbbox.width = bboxWidth;
		                            kcfbbox.height = bboxHeight;
/*
		                            kcfbbox.x = kcfbbox.x - kcfbbox.width * b;
		                            kcfbbox.y = kcfbbox.y - kcfbbox.height * b0;
		                            kcfbbox.width = kcfbbox.width + kcfbbox.width * b0;
		                            kcfbbox.height = kcfbbox.height + kcfbbox.height * b0;
*/
                                    if((kcfbbox.width * kcfbbox.height >= (s - ST0)) && (kcfbbox.width * kcfbbox.height <= (s + ST1))){
                                        kcftracker.init(frame, kcfbbox);
                                        break;
				                    }
                                }
                                okkcf = kcftracker.update(frame, kcfbbox);
                                if(okkcf){
                                    flag3 = 0;
                                    flag5 = 0;
                                    flag2 = 1;
                                } 
                            }
                        }
                    }
                    else{
                        //timemtcnn = (double)getTickCount();
	                    faces = fd.detect(frame, 86.6f, 0.16f);
	                    //fpsmtcnn = getTickFrequency() / ((double)getTickCount() - timemtcnn);
                
                        if(faces.size()){
                            n = faces.size();
    /*                
                            maxv = (faces[0].bbox.x2 - faces[0].bbox.x1) * (faces[0].bbox.y2 - faces[0].bbox.y1);
                            for (size_t i = 0; i < faces.size(); ++i){
                                bboxWidth = faces[i].bbox.x2 - faces[i].bbox.x1;
                                bboxHeight = faces[i].bbox.y2 - faces[i].bbox.y1;
                                if(maxv <= bboxWidth * bboxHeight){
                                    maxi = i;
                                }
                            }
                            bboxWidth = faces[maxi].bbox.x2 - faces[maxi].bbox.x1;
                            bboxHeight = faces[maxi].bbox.y2 - faces[maxi].bbox.y1;
                            pointInterest.x = faces[maxi].bbox.x1 + bboxWidth / 2;
                            pointInterest.y = faces[maxi].bbox.y1 + bboxHeight / 2;

                            rectangle(frame, faces[maxi].bbox.getRect(), Scalar(0, 0, 255), 3);
                            circle(frame, pointInterest, 2, Scalar(0, 0, 255), 2);
    */
	                        for (size_t i = 0; i < faces.size(); ++i){
                                bboxWidth = faces[i].bbox.x2 - faces[i].bbox.x1;
                                bboxHeight = faces[i].bbox.y2 - faces[i].bbox.y1;
                                pointInterest.x = faces[i].bbox.x1 + bboxWidth / 2;
                                pointInterest.y = faces[i].bbox.y1 + bboxHeight / 2;
                            
                                if(flag4){
                                    if(pointInterest.x >= L && pointInterest.x <= R && pointInterest.y >= U && pointInterest.y <= D){
                                        flag3 = 0; 
                                   
                                        flag2 = 1;
                                        //KCF初始化
                                        Rect2d kcfbbox(faces[i].bbox.x1, faces[i].bbox.y1, bboxWidth, bboxHeight);
/*
                                        kcfbbox.x = kcfbbox.x - kcfbbox.width * b;
                                        kcfbbox.y = kcfbbox.y - kcfbbox.height * b0;
                                        kcfbbox.width = kcfbbox.width + kcfbbox.width * b0;
                                        kcfbbox.height = kcfbbox.height + kcfbbox.height * b0;
*/
                                        kcftracker.init(frame, kcfbbox);//!!!!!!!!!!!!!!!!!!
                                    }
                                    else{
                                        rectangle(frame, faces[i].bbox.getRect(), Scalar(0, 255, 0), 3);
                                        circle(frame, pointInterest, 2, Scalar(0, 255, 0), 2);
                                        speed.linear.x = speed.linear.x * e_scale;
                                        if(fabs(speed.linear.x) < v2){
                                            speed.linear.x = 0;
                                        }
                                        speed.angular.z = 0;
                                        cmdVelPub.publish(speed);
                                    }
                                }
                                else{
                                    rectangle(frame, faces[i].bbox.getRect(), Scalar(0, 255, 0), 3);
                                    circle(frame, pointInterest, 2, Scalar(0, 255, 0), 2);
                                    speed.linear.x = speed.linear.x * e_scale;
                                    if(fabs(speed.linear.x) < v2){
                                        speed.linear.x = 0;
                                    }
                                    speed.angular.z = 0;
                                    cmdVelPub.publish(speed);
                                }
	                        }
                        }
/*
                        ostringstream os1;
                        os1 << float(fpsmtcnn);
	                    putText(frame, "FPS: " + os1.str(), Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
*/
                    }
                }
                if(flag2){
                    flag6++;
                    if(flag6 >= 10){
                        flag6 = 0;
                        faces = fd.detect(frame, 86.6f, 0.16f);
                        if(faces.size()){
                            n = faces.size();
                        }
                    }
                    //timerkcf = (double)getTickCount();
                    okkcf = kcftracker.update(frame, kcfbbox);//!!!!!!!!!!!!!!!!!!
                    //fpskcf = getTickFrequency() / ((double)getTickCount() - timerkcf);
                    if (okkcf){
                        rectangle(frame, kcfbbox, Scalar(0, 0, 255), 3);

                        pointInterest.x = kcfbbox.x + kcfbbox.width / 2;
                        pointInterest.y = kcfbbox.y + kcfbbox.height / 2;
                        circle(frame, pointInterest, 2, Scalar(0, 0, 255), 2);
                        s = kcfbbox.width * kcfbbox.height;

                        if(pointInterest.x > t1){
                            speed.linear.x = speed.linear.x * e_scale;
                            if(fabs(speed.linear.x) < v2){
                                speed.linear.x = 0;
                            }
                            speed.angular.z = -v0;
                            cmdVelPub.publish(speed);
                        }
                        else if(pointInterest.x < t0){
                            speed.linear.x = speed.linear.x * e_scale;
                            if(fabs(speed.linear.x) < v2){
                                speed.linear.x = 0;
                            }
                            speed.angular.z = v0;
                            cmdVelPub.publish(speed);
                        }
                        else if(s < s0){
                            speed.angular.z = 0;
                            speed.linear.x = (fabs(speed.linear.x) + 0.01) * s_scale;
                            if(speed.linear.x > v1){
                                speed.linear.x = v1;
                            }
                            cmdVelPub.publish(speed);
                        }
                        else if(s > s1){
                            speed.angular.z = 0;
                            if(speed.linear.x > 0){
                                speed.linear.x = -speed.linear.x;
                            }
                            speed.linear.x = (speed.linear.x - 0.01) * s_scale;
                            if(fabs(speed.linear.x) > v1){
                                speed.linear.x = -v1;
                            }
                            cmdVelPub.publish(speed);
                        }
                        else{
                            speed.linear.x = speed.linear.x * e_scale;
                            if(fabs(speed.linear.x) < v2){
                                speed.linear.x = 0;
                            }
                            speed.angular.z = 0;
                            cmdVelPub.publish(speed);
                        }
                    }
                    else{
	                    //putText(frame, "!!!NO Target!!!", cv::Point(130, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
                        speed.linear.x = speed.linear.x * e_scale;
                        if(fabs(speed.linear.x) < v2){
                            speed.linear.x = 0;
                        }
                        speed.angular.z = 0;
                        cmdVelPub.publish(speed);
                        if(n == 1)
                        {
                            faces = fd.detect(frame, 86.6f, 0.16f);
                            if(faces.size()){
                                bboxWidth = faces[0].bbox.x2 - faces[0].bbox.x1;
	                            bboxHeight = faces[0].bbox.y2 - faces[0].bbox.y1;

                                kcfbbox.x = faces[0].bbox.x1;
		                        kcfbbox.y = faces[0].bbox.y1;
		                        kcfbbox.width = bboxWidth;
		                        kcfbbox.height = bboxHeight;
/*
		                        kcfbbox.x = kcfbbox.x - kcfbbox.width * b;
		                        kcfbbox.y = kcfbbox.y - kcfbbox.height * b0;
		                        kcfbbox.width = kcfbbox.width + kcfbbox.width * b0;
		                        kcfbbox.height = kcfbbox.height + kcfbbox.height * b0;
*/
		                        kcftracker.init(frame, kcfbbox);
                            }
                            else{
                                flag2 = 0;
                                flag3 = 1;
                                flag5 = 1;
                            }
                        }
                        else{
                            faces = fd.detect(frame, 86.6f, 0.16f);
                            if(faces.size()){
                                for (size_t i = 0; i < faces.size(); ++i){
                                    bboxWidth = faces[i].bbox.x2 - faces[i].bbox.x1;
	                                bboxHeight = faces[i].bbox.y2 - faces[i].bbox.y1;

                                    kcfbbox.x = faces[i].bbox.x1;
		                            kcfbbox.y = faces[i].bbox.y1;
		                            kcfbbox.width = bboxWidth;
		                            kcfbbox.height = bboxHeight;
/*
		                            kcfbbox.x = kcfbbox.x - kcfbbox.width * b;
		                            kcfbbox.y = kcfbbox.y - kcfbbox.height * b0;
		                            kcfbbox.width = kcfbbox.width + kcfbbox.width * b0;
		                            kcfbbox.height = kcfbbox.height + kcfbbox.height * b0;
*/
                                    if((kcfbbox.width * kcfbbox.height >= (s - ST0)) && (kcfbbox.width * kcfbbox.height <= (s + ST1))){
				                        kcftracker.init(frame, kcfbbox);
                                        break;
				                    }
                                }
                            }
                            else{
                                 flag2 = 0;
                                 flag3 = 1;
                                 flag5 = 1;
                            }
                        }
                    }
	            // Display FPS
/*
                    ostringstream os3;
                    os3 << float(fpskcf);
	                putText(frame, "FPS: " + os3.str(), Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
*/
                }
                if(flag4){
                    putText(frame, "Tracking: ON", cv::Point(480, 20), FONT_HERSHEY_SIMPLEX, 0.66, Scalar(0, 0, 255), 2);
                }
                else{
                    putText(frame, "Tracking: OFF", cv::Point(480, 20), FONT_HERSHEY_SIMPLEX, 0.66, Scalar(0, 255, 0), 2);
                }
                //output
                imshow("Tracking", frame);
                waitKey(1);
            }
            //Read the next frame
            capture >> frame;
            if (frame.empty())
                return false;
            f++;   
        }
        return 0;
    }
}
