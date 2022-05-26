#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

int main() {
    cv::Mat picture;
    const string filename = "/home/zzd/code/assignments/data/two_image_pose_estimation/1403637188088318976.png";
    picture = cv::imread(filename, 0);
    cv::Mat imageReverse;
    picture.copyTo(imageReverse);
    for (int i = 0; i < picture.rows; i++) {
        for (int j = 0; j < picture.cols; j++) {
            if (picture.channels() == 1) {
                int gray = picture.at<uchar>(i, j);
                imageReverse.at<uchar>(i, j) = 255 - gray;
            }
        }
    }

    cv::imshow("image", picture);
    cv::imshow("imageReversed", imageReverse);
    cv::waitKey(0);
    return 0;
}