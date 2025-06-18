//#define EIGEN_DONT_PARALLELIZE
#define EIGEN_NO_DEBUG
#define EIGEN_VECTORIZE_SSE4_2
#include <iostream>
#include "Eigen/Dense"
#include <chrono>
#include "file_read_write.h"
using namespace disk_hivf;

int main(int argc, char* argv[]) {
    // 使用 Eigen 库
    using namespace Eigen;
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_filename> " << std::endl;
        return 1;
    }

    std::string inputFilename = argv[1];
    // 创建两个随机矩阵
    std::vector<float> data;
    Eigen::Map<Eigen::MatrixXf> A = readMatrixFromDimVecs(inputFilename, data);
    std::cout << 'MatrixXf A rows and cols' << A.rows() << " " << A.cols() << std::endl;
    int rows2 = 1000;
    int cols2 = A.cols();
    MatrixXf B = MatrixXf::Random(rows2, cols2);

    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();

    // 进行矩阵乘法运算
    MatrixXf C = A * B.transpose();

    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();

    // 计算耗时
    std::chrono::duration<double> elapsed = end - start;

    // 输出耗时
    std::cout << "Matrix multiplication took " << elapsed.count() << " seconds." << std::endl;

    return 0;
}