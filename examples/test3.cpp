#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

using namespace std;
using namespace cv;

// void pose_estimation_2d2d(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, vector<DMatch> matches, Mat& R, Mat& t) {
//     Mat K = (Mat_<double>(3, 3) << 458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0);
//     Eigen::MatrixXf A(matches.size(), 9);
//     for(int i = 0; i < matches.size(); i++) {
//         A(i, 0) = keypoints_1[i].pt.x * keypoints_2[i].pt.x;
//         A(i, 1) = keypoints_1[i].pt.y * keypoints_2[i].pt.x;
//         A(i, 2) = keypoints_2[i].pt.x;
//         A(i, 3) = keypoints_1[i].pt.x * keypoints_2[i].pt.y;
//         A(i, 4) = keypoints_1[i].pt.y * keypoints_2[i].pt.y;
//         A(i, 5) = keypoints_2[i].pt.y;
//         A(i, 6) = keypoints_1[i].pt.x;
//         A(i, 7) = keypoints_1[i].pt.y;
//         A(i, 8) = 1;
//     }
//     Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
//     Eigen::MatrixXf U = svd.matrixU();
//     Eigen::MatrixXf V = svd.matrixV();
//     Eigen::Matrix3f F_pre;
//     int row = V.rows();
//     F_pre(0,0) = V(row - 1, 0);
//     F_pre(0,1) = V(row - 1, 1);
//     F_pre(0,2) = V(row - 1, 2);
//     F_pre(1,0) = V(row - 1, 3);
//     F_pre(1,1) = V(row - 1, 4);
//     F_pre(1,2) = V(row - 1, 5);
//     F_pre(2,0) = V(row - 1, 6);
//     F_pre(2,1) = V(row - 1, 7);
//     F_pre(2,2) = V(row - 1, 8);
//     Eigen::JacobiSVD<Eigen::MatrixXf> svd_F(F_pre, Eigen::ComputeThinU | Eigen::ComputeThinV);
//     U = svd_F.matrixU();
//     V = svd_F.matrixV();
//     Eigen::Matrix3f S = U.inverse() * F_pre * V;
//     S(2,2) = 0;
//     Eigen::Matrix3f F = U * S * V.transpose();
//     cout << "F = " << endl << F << endl;
// }

void triangulation(
    const vector<KeyPoint> &keypoint_0,
    const vector<KeyPoint> &keypoint_1,
    const Mat &R, const Mat &t,
    vector<Point3d> &points
);

Point2f pixel2cam(const Point2d &p, const Mat &K);

int main() {

    // read img
    Mat img0 = imread("/home/zzd/code/assignments/data/two_image_pose_estimation/0_un.png");
    Mat img1 = imread("/home/zzd/code/assignments/data/two_image_pose_estimation/1_un.png");

    // ORB
    vector<KeyPoint> keypoint0, keypoint1;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // detect fast
    detector->detect(img0, keypoint0);
    detector->detect(img1, keypoint1);

    // compute brief
    Mat descriptor0, descriptor1;
    descriptor->compute(img0, keypoint0, descriptor0);
    descriptor->compute(img1, keypoint1, descriptor1);

    // match keypoints
    vector<DMatch> matches;
    matcher->match(descriptor0, descriptor1, matches);

    // filter
    auto min_max = minmax_element(matches.begin(), matches.end(),
                                [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    std::vector<DMatch> good_matches;
    for (int i = 0; i < descriptor0.rows; i++) {
        if (matches[i].distance <= max(2 * min_dist, 30.0)) {
            good_matches.push_back(matches[i]);
        }
    }

    // draw
     Mat img_match, img_goodmatch;
      drawMatches(img0, keypoint0, img1, keypoint1, matches, img_match);
    //  drawMatches(img0, keypoint0, img1, keypoint1, good_matches, img_goodmatch);
      imshow("all matches", img_match);
    //  imshow("good matches", img_goodmatch);
    // waitKey(0);

    // ransac
    vector<KeyPoint> R_keypoint0, R_keypoint1;
    for (int i = 0; i < matches.size(); i++) {
        R_keypoint0.push_back(keypoint0[matches[i].queryIdx]);
        R_keypoint1.push_back(keypoint1[matches[i].trainIdx]);
    }

    vector<Point2f> p0, p1;
    for (int i = 0; i < matches.size(); i++) {
        p0.push_back(R_keypoint0[i].pt);
        p1.push_back(R_keypoint1[i].pt);
    }

    vector<uchar> RansacStatus;
    Mat Fundamental = findFundamentalMat(p0, p1, RansacStatus, FM_RANSAC);
    cout << "fundamental_matrix is " << endl << Fundamental << endl;

    Point2d principal_point(367.215, 248.375);
    double focal_length = 458;      //相机焦距
    Mat essential_matrix;
    essential_matrix = findEssentialMat(p0, p1, focal_length, principal_point);
    cout << "essential_matrix is " << endl << essential_matrix << endl;
    Mat R, t;
    recoverPose(essential_matrix, p0, p1, R, t, focal_length, principal_point);
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;

    // new matches
    vector<KeyPoint> Ransac_keypoint0,Ransac_keypoint1;
    vector<DMatch> ransac_matches;
    int index = 0;
    for (int i = 0; i < matches.size(); i++) {
        if (RansacStatus[i] != 0) {
            Ransac_keypoint0.push_back(R_keypoint0[i]);
            Ransac_keypoint1.push_back(R_keypoint1[i]);
            matches[i].queryIdx=index;
            matches[i].trainIdx=index;
            ransac_matches.push_back(matches[i]);
            index++;
        }
    }

    Mat img_ransac_matches;
    drawMatches(img0, Ransac_keypoint0, img1, Ransac_keypoint1, ransac_matches, img_ransac_matches);
    // imshow("ransac", img_ransac_matches);
    waitKey(0);
}

void triangulation(
    const vector<KeyPoint> &keypoint_0,
    const vector<KeyPoint> &keypoint_1,
    const Mat &R, const Mat &t,
    vector<Point3d> &points) {
    Mat T0 = (Mat_<float>(3, 4) <<
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0);
    Mat T1 = (Mat_<float>(3, 4) <<
    R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
    R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
    R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));

    Mat K = (Mat_<double>(3, 3) << 458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0);
    vector<Point2f> pts_0, pts_1;
    for (int i = 0; i < keypoint_0.size(); i++) {
        pts_0.push_back(pixel2cam(keypoint_0[i].pt, K));
        pts_1.push_back(pixel2cam(keypoint_1[i].pt, K));
    }

    Mat pts_4d;
    triangulatePoints(T0, T1, pts_0, pts_1, pts_4d);

    for(int i = 0; i < pts_4d.cols; i++) {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0);
        Point3d pw(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
        points.push_back(pw);
    }
}

Point2f pixel2cam(const Point2d &p, const Mat &K) {
  return Point2f
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}