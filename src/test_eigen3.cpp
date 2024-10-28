#include "Eigen/Dense"
#include <iostream>

// 使用 Eigen::Ref 的函数，可以接受 Eigen::RowVectorXf 和 Eigen::Map<Eigen::RowVectorXf>
void processRowVector(const Eigen::Ref<const Eigen::RowVectorXf>& vec) {
    std::cout << "RowVectorXf (Ref): " << vec << std::endl;
}

int main() {
    // 创建一个 RowVectorXf
    Eigen::RowVectorXf rowVec(3);
    rowVec << 1, 2, 3;

    // 创建一个 Eigen::Map<Eigen::RowVectorXf>
    float data[] = {4, 5, 6};
    Eigen::Map<Eigen::RowVectorXf> mapVec(data, 3);

    // 传递 RowVectorXf
    processRowVector(rowVec);

    // 传递 Eigen::Map<Eigen::RowVectorXf>
    processRowVector(mapVec);

    return 0;
}