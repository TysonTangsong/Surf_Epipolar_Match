#include <stdio.h>  
#include "opencv2/core/core.hpp"  
#include "opencv2/features2d/features2d.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/nonfree/nonfree.hpp"  
#include <opencv2/calib3d/calib3d.hpp>  
#include <iostream>  
using namespace std;  
  
using namespace cv;  
  
static void help()  
{  
    printf("\nThis program demonstrates using features2d detector, descriptor extractor and simple matcher\n"  
            "Using the SURF desriptor:\n"  
            "\n"  
            "Usage:\n matcher_simple <image1> <image2>\n");  
}  
  
int main(int argc, char** argv)  
{  
    //if(argc != 3)  
    //{  
    //    help();  
    //    return -1;  
    //}  
  
    Mat img1 = imread("D:\\faceData\\stereopairs\\cones\\imL.jpg", CV_LOAD_IMAGE_GRAYSCALE);  
    Mat img2 = imread("D:\\faceData\\stereopairs\\cones\\imR.jpg", CV_LOAD_IMAGE_GRAYSCALE);  
    if(img1.empty() || img2.empty())  
    {  
        printf("Can't read one of the images\n");  
        return -1;  
    }  
  
    // detecting keypoints  
    SurfFeatureDetector detector(400);  
    vector<KeyPoint> keypoints1, keypoints2;  
    detector.detect(img1, keypoints1);  
    detector.detect(img2, keypoints2);  
  
    // computing descriptors  
    SurfDescriptorExtractor extractor;  
    Mat descriptors1, descriptors2;  
    extractor.compute(img1, keypoints1, descriptors1);  
    extractor.compute(img2, keypoints2, descriptors2);  
  
    // matching descriptors  
    BFMatcher matcher(NORM_L2);  
    vector<DMatch> matches;  
    matcher.match(descriptors1, descriptors2, matches);  
  
    // drawing the results  
    namedWindow("matches", 1);  
    Mat img_matches0,img_matches1;  
    drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches0);  
    //for(int i=0;i<matches.size();i++)  
    //{       
    //  //if(abs((keypoints1[matches[i].queryIdx].pt.y-keypoints2[matches[i].trainIdx].pt.y)>5)  
    //  //{  
  
    //  //}  
    //  //cout<<keypoints1[matches[i].queryIdx].pt.x<<endl;  
    //  Point2f pt1,pt2;  
    //  pt1=keypoints1[matches[i].queryIdx].pt;  
    //  pt2=keypoints2[matches[i].trainIdx].pt;  
    //  if(abs(pt1.y-pt2.y)>5)  
    //  {  
    //      matches[i].distance=100;  
    //  }  
  
  
    //}  
 //   drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches1);  
    // 分配空间  
    int ptCount = (int)matches.size();  
    Mat p1(ptCount, 2, CV_32F);  
    Mat p2(ptCount, 2, CV_32F);  
  
    // 把Keypoint转换为Mat  
    Point2f pt;  
    for (int i=0; i<ptCount; i++)  
    {  
        pt = keypoints1[matches[i].queryIdx].pt;  
        p1.at<float>(i, 0) = pt.x;  
        p1.at<float>(i, 1) = pt.y;  
  
        pt = keypoints2[matches[i].trainIdx].pt;  
        p2.at<float>(i, 0) = pt.x;  
        p2.at<float>(i, 1) = pt.y;  
    }  
    // 用RANSAC方法计算F  
    // Mat m_Fundamental;  
    // 上面这个变量是基本矩阵  
     vector<uchar> m_RANSACStatus;  
    // 上面这个变量已经定义过，用于存储RANSAC后每个点的状态  
    //m_Fundamental = findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);  
     Mat m_Fundamental=findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);  
     cout<<m_Fundamental<<endl;  
    // 计算野点个数  
    int OutlinerCount = 0;  
    for (int i=0; i<ptCount; i++)  
    {  
        if (m_RANSACStatus[i] == 0) // 状态为0表示野点  
        {  
            OutlinerCount++;  
        }  
    }  
  
    // 计算内点  
    vector<Point2f> m_LeftInlier;  
    vector<Point2f> m_RightInlier;  
    vector<DMatch> m_InlierMatches;  
    // 上面三个变量用于保存内点和匹配关系  
    int InlinerCount = ptCount - OutlinerCount;  
    m_InlierMatches.resize(InlinerCount);  
    m_LeftInlier.resize(InlinerCount);  
    m_RightInlier.resize(InlinerCount);  
    InlinerCount = 0;  
    for (int i=0; i<ptCount; i++)  
    {  
        if (m_RANSACStatus[i] != 0)  
        {  
            m_LeftInlier[InlinerCount].x = p1.at<float>(i, 0);  
            m_LeftInlier[InlinerCount].y = p1.at<float>(i, 1);  
            m_RightInlier[InlinerCount].x = p2.at<float>(i, 0);  
            m_RightInlier[InlinerCount].y = p2.at<float>(i, 1);  
            m_InlierMatches[InlinerCount].queryIdx = InlinerCount;  
            m_InlierMatches[InlinerCount].trainIdx = InlinerCount;  
            InlinerCount++;  
        }  
    }  
  
    // 把内点转换为drawMatches可以使用的格式  
    vector<KeyPoint> key1(InlinerCount);  
    vector<KeyPoint> key2(InlinerCount);  
    KeyPoint::convert(m_LeftInlier, key1);  
    KeyPoint::convert(m_RightInlier, key2);  
    for(int i=0;i<10;i++)  
    cout<<key1[i].pt<<" "<<key2[i].pt<<" "<<key1[i].pt.x-key2[i].pt.x<<endl;  
  
    drawMatches(img1, key1, img2, key2, m_InlierMatches, img_matches1);  
    imshow("matches0", img_matches0);  
     imshow("matches1", img_matches1);  
     Mat test(img1.rows,img1.cols,CV_32FC1);  
     float a=0.0105591872,b=0.1024243227,c=10.106085341;  
     for(int i=0;i<test.rows;i++)  
     {  
         float* pt=test.ptr<float>(i);          
         for(int j=0;j<test.cols;j++)  
         {  
             float val=j*a+i*b+c;  
             pt[j]=val;  
         }  
     }  
     //cout<<test<<endl;  
     cv::normalize(test,test,0,1,CV_MINMAX);  
     cv::convertScaleAbs(test,test,255);      
     test.convertTo(test,CV_8UC1);  
     imshow("test",test);  
    waitKey(0);  
  
    return 0;  
}  