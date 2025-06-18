#include "kmeans.h"
#include <iostream>
#include <vector>
#include "Eigen/Dense"
#include "file_read_write.h"
#include "common.h"

using namespace disk_hivf;

void test_kmeans_core() {
    // 创建示例数据
    std::vector<float> batch_features = 
                     {1.0, 2.0, 3.0,
                      4.0, 5.0, 6.0,
                      7.0, 8.0, 9.0,
                      10.0, 11.0, 12.0};
    for (auto a: batch_features) {
        std::cout << a << std::endl;
    }
    RMatrixDf centers(3, 3);
    centers << 1.0, 0.0, 0.0,
               1.0, 0.0, -1.0,
               2.0, 1.0, -1.0;

    // 创建 Eigen::Map 对象
    Eigen::Map<RMatrixDf> batch_features_map(batch_features.data(), 4, 3);

    std::cout << "batch_features_map=\n" << batch_features_map << std::endl;

    Eigen::Map<RMatrixDf> centers_map(centers.data(), centers.rows(), centers.cols());
    Eigen::VectorXf centers_squa_norm = centers_map.rowwise().squaredNorm();
    // 创建一个向量来存储分配结果
    std::vector<Int> assign;

    // 调用 kmeans_core 函数
    kmeans_core(batch_features_map, centers_map, centers_squa_norm, assign);

    // 输出分配结果
    std::cout << "Assignment of each point to the nearest center:" << std::endl;
    for (size_t i = 0; i < assign.size(); ++i) {
        std::cout << "Point " << i << " is assigned to center " << assign[i] << std::endl;
    }
}


void test_kmeans(std::string & input_file) {
    std::vector<float> features_data;
    int dimension;
    Int numVecs;
    readDimVecs(input_file, features_data, dimension, numVecs);
    std::vector<float> centers_data;
    std::vector<Int> assign;
    double ret_loss;
    kmeans(features_data, dimension, 300, 40, 256, 3, 50000, centers_data, assign, ret_loss);
    std::cout << "ret_loss " << ret_loss << std::endl;
    //for (auto a: assign) {
    //    std::cout << a << std::endl;
    //}
}

int main(int argc, char* argv[]) {
    using namespace Eigen;
    Eigen::initParallel();
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_filename> " << std::endl;
        return 1;
    }
    //test_kmeans_core();
    std::string inputFilename = argv[1];
    test_kmeans(inputFilename);

    return 0;
}
