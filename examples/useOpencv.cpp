#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace std;

double fx = 458.654;
double fy = 457.296;
double cx = 367.215;
double cy = 248.375;
const cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
const cv::Mat D = ( cv::Mat_<double>(5, 1) << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0);

void UndistortKeyPoints(vector<cv::Point2f> &points) {
    uint N = points.size();
    cv::Mat mat(N, 2, CV_32F);
    for (int i = 0; i < N; i++) {
        mat.at<float>(i, 0) = points[i].x;
        mat.at<float>(i, 1) = points[i].y;
    }
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, K, D, cv::Mat(), K);
    mat = mat.reshape(1);

    for (int i = 0; i < N; i++) {
        cv::Point2f kp;
        kp.x = mat.at<float>(i, 0);
        kp.y = mat.at<float>(i, 1);
        points[i] = kp;
    }
}

int main() {
    cv::Mat picture;
    const string filefolder = "/home/zzd/code/assignments/data/two_image_pose_estimation/";
    // picture = cv::imread(filename, 0);
    // cv::Mat imageReverse;
    // picture.copyTo(imageReverse);
    // for (int i = 0; i < picture.rows; i++) {
    //     for (int j = 0; j < picture.cols; j++) {
    //         if (picture.channels() == 1) {
    //             int gray = picture.at<uchar>(i, j);
    //             imageReverse.at<uchar>(i, j) = 255 - gray;
    //         }
    //     }
    // }
    const int nImage = 2;
    const int ImgWidth = 752;
    const int ImgHeight = 480;
    const int MAX_CNT = 150;
    const int MIN_DIST = 30;

    cv::Mat map1, map2;
    cv::Size imageSize(ImgWidth, ImgHeight);
    const double alpha = 1;
    cv::Mat NewCameraMatrix = getOptimalNewCameraMatrix(K, D, imageSize, alpha, imageSize, 0);
    cv::initUndistortRectifyMap(K, D, cv::Mat(), NewCameraMatrix, imageSize, CV_16SC2, map1, map2);
    for (int i = 0; i < nImage; i++) {
        string filename = filefolder + to_string(i) + ".png";
        cv::Mat RawImage = cv::imread(filename);

        vector<cv::Point2f> pts;
        cv::Mat RawImage_Gray;
        cv::cvtColor(RawImage, RawImage_Gray, CV_RGB2GRAY);
        cv::goodFeaturesToTrack(RawImage_Gray, pts, MAX_CNT, 0.01, MIN_DIST);
        for (auto pt : pts) {
            circle(RawImage, pt, 2, cv::Scalar(255, 0, 0), 2);
        }
        cv::imshow("pts", RawImage);

        UndistortKeyPoints(pts);

        cv::Mat UndistortImage;
        // cv::remap(RawImage, UndistortImage, map1, map2, cv::INTER_LINEAR);
        cv::undistort(RawImage, UndistortImage, K, D, K);
        for(auto pt : pts) {
            circle(UndistortImage, pt, 2, cv::Scalar(0, 0, 255), 2);
        }
        cv::imshow("pts_un", UndistortImage);

        string OutputPath = filefolder + to_string(i) + "_pts_un.png";
        cv::imwrite(OutputPath, UndistortImage);
        cv::waitKey(0);
    }

    // cv::imshow("image", picture);
    // cv::imshow("imageReversed", imageReverse);
    // cv::waitKey(0);
    return 0;
}