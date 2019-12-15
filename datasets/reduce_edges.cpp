#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include "stdio.h"
#include <cstdlib>
using namespace std;

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


int find(vector<pair<int, int> > dict, int id){
    for(int i=0;i<dict.size();i++){
        if(dict[i].first == id) return i;
    }
    return dict.size();
}

tuple<int, int, int, int> process(string basename, string filename, int ratio){
    SRC = cv::imread(basename+"/"+filename+".png", 0);;
    TMP = cv::Mat::zeros(256, 256, CV_16U);
    DST = cv::Mat::zeros(256, 256, CV_8U);
    vector<pair<int, int> > dict;
    map<int, int> rdict;
    ID  = 1;
    for(int i=0;i<256;i++){for(int j=0;j<256;j++){
        int num = c(i,j);
        if (num > 0){
            dict.push_back(make_pair(ID,num));
            ID++;
        }
    }}

    sort(dict.begin(), dict.end(),[](pair<int, int> a, pair<int, int> b){return a.second > b.second;});

    //reduce
    for(int pos=0;pos<dict.size();pos++){
        if(pos < int(float(ratio)/100*dict.size())){
            rdict[dict[pos].first] = dict[pos].second;
        }
    }

    for(int i=0;i<256;i++){for(int j=0;j<256;j++){
        int id = TMP.at<uint16_t>(i,j);
        if(rdict.find(id) == rdict.end()){
            DST.at<uint8_t>(i,j) = 255;
        }
    }}

    ostringstream oss;
    oss << basename << ratio << "x1/" << filename << ".png";
    cv::imwrite(oss.str(), DST);

    int dsum = accumulate(dict.begin(), dict.end(), 0, [](int acc, pair<int, int> a){return acc + a.second;});
    int rsum = accumulate(rdict.begin(), rdict.end(), 0, [](int acc, pair<int, int> a){return acc + a.second;});
    return forward_as_tuple(dict.size(), dsum, rdict.size(), rsum);
}

int main(int argc, char *argv[]){
    int ratio = atoi(getenv("ratio"));
    string basename = getenv("basename");
    for(int i=1;i<argc;i++){
        int dnum, dsum, rnum, rsum;
        tie(dnum, dsum, rnum, rsum) = process(basename , argv[i], ratio);
        cout << argv[i] << " " <<  dnum << " " << dsum << " " << rnum << " " << rsum << endl;
    }
}
