#include "yoloInference.h"



int read_files_in_dirs(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        printf("open  %s fail \n",p_dir_name);
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }
    sort(file_names.begin(),file_names.end());
    closedir(p_dir);
    return 0;
}



int main(int argc, char** argv) 
{

    YOLOv5Infer yolov5Detector = YOLOv5Infer();
    std::vector<std::string> file_names;
    if (read_files_in_dirs(argv[1], file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        // return -1;
    }
    else if(read_files_in_dirs("../samples", file_names) < 0) 
    {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    float conf_th = 0.0;
    if(argc > 2){
        std::string th = std::string(argv[2]);
        conf_th = std::atof(th.c_str());
        printf("set conf_threshold as :%f \n",conf_th);
    }
 
    int fcount = 0;
    for (int f = 0; f < (int)file_names.size(); f++) 
    {
        fcount++;
        if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size()) continue;
        for (int b = 0; b < fcount; b++) 
        {
            cv::Mat img = cv::imread(std::string(argv[2]) + "/" + file_names[f - fcount + 1 + b]);
            // cv::cvtColor(img,img,cv::COLOR_BGR2GRAY);
            // cv::Rect rect(0,0,1280,720);
            std::vector<cv::Rect> boxes;
            std::vector<float> confs;
            std::vector<float> cls_id;
            // cv::Mat dst = img(rect,boxex);

            // img = img(rect);
            // cv::Mat dst;
            // dst.create(720*2,1280,img.type());
            // cv::Mat up = dst(cv::Rect(0,0,1280,720));
            // cv::Mat down = dst(cv::Rect(0,720,1280,720));
            // img.copyTo(up);
            // img.copyTo(down);

            yolov5Detector.run(img,boxes,confs,cls_id,conf_th);
            // if(boxes.size() != confs.size())
            // (confs.size() == cls_id.size());
            printf("the %d-th image result are : \n",b);
            for(int i=0;i<confs.size();i++){
                printf("rect = %d,%d,%d,%d  conf = %f\n",boxes[i].x,boxes[i].y,boxes[i].width,boxes[i].height,confs[i]);
                // cv::rectangle(img, cv::Rect(boxes[i].x,boxes[i].y,boxes[i].width,boxes[i].height), cv::Scalar(0x27, 0xC1, 0x36), 2);
                // cv::putText(img, std::to_string((int)cls_id[i]), cv::Point(boxes[i].x,boxes[i].y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
            // cv::imwrite("_" + file_names[f - fcount + 1 + b], img);
        }

        fcount = 0;
    }

    return 0;
}