//#define EIGEN_DONT_PARALLELIZE
#define EIGEN_NO_DEBUG
#define EIGEN_VECTORIZE_SSE4_2
#include <iostream>
#include "Eigen/Dense"
#include <chrono>
#include "file_read_write.h"
#include "common.h"
using namespace disk_hivf;

int main(int argc, char* argv[]) {
    // 使用 Eigen 库
    using namespace Eigen;
    // 限制Eigen使用单线程
    Eigen::setNbThreads(1);
    int rows2 = 128;
    int cols2 = 128;
    RowVectorXf A = RowVectorXf::Random(cols2);
    RMatrixDf B = RMatrixDf::Random(rows2, cols2+2);

    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();

    RMatrixDf D(rows2, cols2);
    for (int i = 0; i < 10000; i ++) {
        // 进行矩阵乘法运算
        
        D = B.block(0, 2, rows2, cols2).transpose();
        RowVectorXf C = A * D;
        
        /*
        RowVectorXf C(rows2);
        for (int j = 0; j < rows2; j++) {
            C(j) = A * B.row(j).transpose();
        }
        */
    }
    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();

    // 计算耗时
    std::chrono::duration<double> elapsed = end - start;

    // 输出耗时
    std::cout << "Matrix multiplication took " << elapsed.count() << " seconds." << std::endl;

    return 0;
}