#include "include/useEigen.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"
#include <vector>
#include <fstream>
#include <sstream>

using namespace std;

void readdata2eigen(const string data_file, Eigen::MatrixXd& A, Eigen::VectorXd& b) {
    ifstream in(data_file);
    string line;
    if (in) {
        getline(in, line);
        int i = 0;
        int j = 0;
        while(getline(in, line)) {
            stringstream ss(line);
            double x;
            while (ss >> x)
            {
                if (j % 2 == 0) {
                    A(i, 0) = x;
                    A(i, 1) = 1;
                } else {
                    b(i) = x;
                }
                j++;
            }
            i++;
        }
    }
}

int main() {
    const string data_file = "/home/zzd/code/assignments/data/data.txt";
    const string data2_file = "/home/zzd/code/assignments/data/data2.txt";
    Eigen::MatrixXd A(100, 2);
    Eigen::VectorXd b(100, 1);
    Eigen::VectorXd x(2, 1);
    readdata2eigen(data_file, A, b);
    LinearSolver(A, b, x);
    cout << "m = " << x(0) << ",   " << "n = " << x(1) << endl;

    readdata2eigen(data2_file, A, b);
    LinearSolver(A, b, x);
    cout << "m = " << x(0) << ",   " << "n = " << x(1) << endl;
    return 0;
}