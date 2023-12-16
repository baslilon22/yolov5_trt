#ifndef YOLOV5INFERENCE_H_
#define YOLOV5INFERENCE_H_

#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"



#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "yololayer.h"




#define BATCH_SIZE 1

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1



class YOLOv5Infer
{
public:
    YOLOv5Infer();
    void run(cv::Mat& img,std::vector<cv::Rect>& boxex,std::vector<float> conf_res,std::vector<float> cls_id,float conf_th);

    ~YOLOv5Infer();

    float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    float prob[BATCH_SIZE * OUTPUT_SIZE];

    void* buffers[2];


    int inputIndex;
    int outputIndex;

};

#endif