#include <include/useEigen.h>

void ComputeExtrinsic(const Eigen::Matrix3d& Ric_l, const Eigen::Matrix3d& Ric_r,
             const Eigen::Vector3d& Tic_l, const Eigen::Vector3d& Tic_r,
             Eigen::Quaterniond& Q_cl_cr, Eigen::Vector3d& p_cl_cr) {
    Q_cl_cr = Ric_l.transpose() * Ric_r;
    p_cl_cr = Ric_l.transpose() * (Tic_r - Tic_l);
}

void LinearSolver(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, Eigen::VectorXd& x) {
    Eigen::VectorXd tmp_b = A.transpose() * b;
    Eigen::MatrixXd tmp_A = A.transpose() * A;
    x = tmp_A.ldlt().solve(tmp_b);
}