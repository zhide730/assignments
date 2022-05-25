#include "include/useEigen.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"
#include <vector>
#include <fstream>

Eigen::Matrix3d Ric_l, Ric_r;
Eigen::Vector3d Tic_l, Tic_r;

void readParameters(std::string config_file) {
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cout << "ERROR: Wrong path to settings" << std::endl;
    }

    cv::Mat imu_p_cam_left, imu_p_cam_right;
    fsSettings["imu_p_cam_left"] >> imu_p_cam_left;
    fsSettings["imu_p_cam_right"] >> imu_p_cam_right;
    cv::cv2eigen(imu_p_cam_left, Tic_l);
    cv::cv2eigen(imu_p_cam_right, Tic_r);

    double w, x, y, z;
    cv::FileNode n = fsSettings["imu_q_cam_left"];
    w = static_cast<double>(n["w"]);
    x = static_cast<double>(n["x"]);
    y = static_cast<double>(n["y"]);
    z = static_cast<double>(n["z"]);
    Eigen::Quaterniond Qic_l(w, x, y, z);
    Ric_l = Qic_l;

    n = fsSettings["imu_q_cam_right"];
    w = static_cast<double>(n["w"]);
    x = static_cast<double>(n["x"]);
    y = static_cast<double>(n["y"]);
    z = static_cast<double>(n["z"]);
    Eigen::Quaterniond Qic_r(w, x, y, z);
    Ric_r = Qic_r;
}

int main() {
    std::string config_file = "/home/zzd/code/assignments/config/useEigen.yaml";
    readParameters(config_file);
    Eigen::Quaterniond Q_cl_cr;
    Eigen::Vector3d p_cl_cr;
    ComputeExtrinsic(Ric_l, Ric_r, Tic_l, Tic_r, Q_cl_cr, p_cl_cr);
    std::cout << "Q_cl_cr = " << Q_cl_cr.toRotationMatrix() << std::endl;
    std::cout << "p_cl_cr = " << p_cl_cr.transpose() << std::endl;
    return 0;
}