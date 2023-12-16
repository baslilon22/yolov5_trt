#include "yoloInference.h"

#include "common.hpp"
#include "preprocess.h"



#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
// #define NMS_THRESH 0.4
// #define CONF_THRESH 0.5
#define NMS_THRESH 0.01
#define CONF_THRESH 0.01

#define NET s  // s m l x
#define NETSTRUCT(str) createEngine_##str
#define CREATENET(net) NETSTRUCT(net)
#define STR1(x) #x
#define STR2(x) STR1(x)


IRuntime* runtime;
ICudaEngine* engine;
IExecutionContext* context;
cudaStream_t stream;


const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;


void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

YOLOv5Infer::YOLOv5Infer()
{
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{ nullptr };
    size_t size{ 0 };
    std::string engine_name = STR2(NET);
    engine_name = "../yolov5" + engine_name + ".engine";
    // engine_name = "../yolov5-trt/yolov5" + engine_name + "-tx2.engine";
    std::cout<<"loading engine:"<<engine_name<<std::endl;
    {
        std::ifstream file(engine_name, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } 
    

    // prepare input data ---------------------------
    // static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    // //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    // //    data[i] = 1.0;
    // static float prob[BATCH_SIZE * OUTPUT_SIZE];
    // IRuntime* runtime = createInferRuntime(gLogger);
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    // ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);

    // IExecutionContext* context = engine->createExecutionContext();
    context = engine->createExecutionContext();
    assert(context != nullptr);

    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);

    // void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    // const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    // const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    // cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
        std::cout<<"yolo init finished!"<<std::endl;

}

YOLOv5Infer::~YOLOv5Infer()
{
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

static int imgConter = 0;
void YOLOv5Infer::run(cv::Mat& img,std::vector<cv::Rect>& boxex,std::vector<float> conf_res,std::vector<float> cls_id,float conf_th)
{
    

    int fcount = BATCH_SIZE;
    {
        auto before = std::chrono::system_clock::now();
        //preprocess
        for (int b = 0; b < fcount; b++) 
        {
            //cv::Mat img = cv::imread(std::string(argv[2]) + "/" + file_names[f - fcount + 1 + b]);
            if (img.empty()) continue;
            cv::Mat pr_img = preprocess_img(img); // letterbox BGR to RGB

            // int i = 0;
            // for (int row = 0; row < INPUT_H; ++row) {
            //     uchar* uc_pixel = pr_img.data + row * pr_img.step;
            //     for (int col = 0; col < INPUT_W; ++col) {
            //         data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
            //         data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
            //         data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
            //         uc_pixel += 3;
            //         ++i;
            //     }
            // }
            uchar* image_data = pr_img.data;
            int step = pr_img.step;
            ToChannelLast_GPU(data,image_data,step,INPUT_H, INPUT_W,b,fcount);
        

            // // pr_img.convertTo(pr_img,CV_32FC3,1/255.0);
            // std::vector<cv::Mat> bgrChannels(3);
            // cv::split(pr_img,bgrChannels);
            // for(auto i=0;i<bgrChannels.size();i++)
            // {
            //     bgrChannels[i] = bgrChannels[i]/255.0;
            //     std::vector<float> imageData = std::vector<float>(bgrChannels[i].reshape(1,1));
            //     memcpy(data,&imageData[0],imageData.size()*sizeof(float));
            // }

        }
        
        auto preend = std::chrono::system_clock::now();
        std::cout <<"        preprocess time:" << std::chrono::duration_cast<std::chrono::milliseconds>(preend - before).count() << "ms" << std::endl;

        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        std::cout <<"        net time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
        }
        auto final = std::chrono::system_clock::now();
        // std::cout << "nms time:"<< std::chrono::duration_cast<std::chrono::milliseconds>(final - end).count() << "ms" << std::endl;
        
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            //std::cout << res.size() << std::endl;
            //cv::Mat img = cv::imread(std::string(argv[2]) + "/" + file_names[f - fcount + 1 + b]);
            for (size_t j = 0; j < res.size(); j++) {
                if (res[j].conf<conf_th) continue;
                cv::Rect r = get_rect(img, res[j].bbox);
                r.x = std::max(0,r.x);
                r.y = std::max(0,r.y);
                r.width = std::min(img.cols-r.x,r.width);
                r.height = std::min(img.rows-r.y,r.height);

                boxex.push_back(r);
                conf_res.push_back(res[j].conf);
                cls_id.push_back(res[j].class_id);
                // cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                // cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
            // cv::imwrite("_" + file_names[f - fcount + 1 + b], img);
            // cv::imwrite(std::to_string(imgConter)+"_output.png", img);
            imgConter ++;

            // cv::imshow("Result"+std::to_string(b),img);
            // cv::waitKey(10);
        }
        fcount = 0;
    }

}


