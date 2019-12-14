#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include "stdio.h"

cv::Mat SRC;
cv::Mat TMP;
cv::Mat DST;
int     ID ;

int c(int i,int j){
    if(i<0||i>255||j<0||j>255) return 0;
    if(SRC.at<uint8_t>(i,j) != 0) return 0;
    if(TMP.at<uint16_t>(i,j) != 0) return 0;
    TMP.at<uint16_t>(i,j) = ID;

    return c(i-1,j+1) + c(i+0,j+1) + c(i+1,j+1) +\
           c(i-1,j+0) +      1     + c(i+1,j+0) +\
           c(i-1,j-1) + c(i+0,j-1) + c(i+1,j-1) ;

//    return c(i-2,j+2) + c(i-1,j+2) + c(i+0,j+2) + c(i+1,j+2) + c(i+2,j+2) +\
//           c(i-2,j+1) + c(i-1,j+1) + c(i+0,j+1) + c(i+1,j+1) + c(i+2,j+1) +\
//           c(i-2,j+0) + c(i-1,j+0) +      1     + c(i+1,j+0) + c(i+2,j+0) +\
//           c(i-2,j-1) + c(i-1,j-1) + c(i+0,j-1) + c(i+1,j-1) + c(i+2,j-1) +\
//           c(i-2,j-2) + c(i-1,j-2) + c(i+0,j-2) + c(i+1,j-2) + c(i+2,j-2) ;
}


int find(std::vector<std::pair<int, int> > dict, int id){
    for(int i=0;i<dict.size();i++){
        if(dict[i].first == id) return i;
    }
    return dict.size();
}

void process(std::string basename, std::string filename, int ratio){
    SRC = cv::imread(basename+"/"+filename+".png", 0);;
    TMP = cv::Mat::zeros(256, 256, CV_16U);
    DST = cv::Mat::ones(256, 256, CV_8U)*255;
    std::vector<std::pair<int, int> > dict;
    ID  = 1;
    for(int i=0;i<256;i++){for(int j=0;j<256;j++){
        int num = c(i,j);
        if (num > 0){
            dict.push_back(std::make_pair(ID,num));
            ID++;
        }
    }}

    std::sort(dict.begin(), dict.end(),[](std::pair<int, int> a, std::pair<int, int> b){return a.second > b.second;});

    for(int i=0;i<256;i++){for(int j=0;j<256;j++){
        int id = TMP.at<uint16_t>(i,j);
        if(id != 0){
            int pos = find(dict,id);
            if(pos < int(float(ratio)/100*dict.size())){
                DST.at<uint8_t>(i,j) = 0;
            }
        }
    }}

    std::ostringstream oss;
    oss << basename << ratio << "x1/" << filename << ".png";
    cv::imwrite(oss.str(), DST);
}

int main(int argc, char *argv[]){
    for(int i=1;i<argc;i++){
        process("./coco_stuff_hed/train_hed", argv[i], 50);
        std::cout << argv[i] << std::endl;
    }
}
