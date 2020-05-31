# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load. 
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed. 
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson. 
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures. 

See the classroom instruction and code comments for more details on each of these parts. Once you are finished with this project, the keypoint matching part will be set up and you can proceed to the next lesson, where the focus is on integrating Lidar points and on object detection using deep-learning. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.

## Mid-Term Report

### MP.1:Data Buffer Optimization
Already the exist vector is used. When the size exceed the data buffer size the old data
is removed.

### MP.2 : Keypoint Detection
Implemented detectors HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT and can be selectable by setting a string accordingly.

### MP.3 Keypoint Removal
All keypoints outside of a pre-defined rectangle had been removed and only the keypoints within the rectangle are used.

### MP.4 Keypoint Descriptors
Implemented descriptors BRIEF, ORB, FREAK, AKAZE and SIFT and can be selectable by setting a string accordingly.

### MP.6 Descriptor Distance Ratio
The K-Nearest-Neighbor matching is usef to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints.

### MP.7 Performance Evaluation 1
Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented.
![keypoint_veh](images/keypoint_veh.PNG)

### MP.8 Performance Evaluation 2
Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.
![match_point](images/match_point.PNG)

### MP.9 Performance Evaluation 3
Log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet and based on this data, the T
![time_perf](images/time_perf.PNG)


TOP3 detector / descriptor:
- FAST/ORB
- FAST/BRIEF
- FAST/BRISK

# NOTES:
- AKAZE descriptors will only work with AKAZE keypoints.
- SIFT detector with ORB descriptor ran out of memory.
- The spreadsheet in pdf format can be found [here.](https://github.com/m-loay/Sensor-Fusion-Self-Driving-Car/blob/master/SFND_2D_Feature_Tracking/Results.pdf)