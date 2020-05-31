#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <opencv2/core.hpp>
#include <stdio.h>
#include <iostream>
#include <sstream>

struct DataFrame { // represents the available sensor information at the same time instance
    
    cv::Mat cameraImg; // camera image
    
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
};

struct Results
{
    std::string detector;
    std::string descriptor;
    std::string matcher;
    std::string selector;
    double detector_time_ms;
    double descriptor_time_ms;
    double matcher_time_ms;
    double avg_time_ms;
    int no_detected_points;
    int no_detected_points_on_vehicle;
    int no_descriptors;
    int no_matched_points;
    int no_images;
};


#endif /* dataStructures_h */
