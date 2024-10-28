#pragma once
#include <vector>
#include <string>
#include <cstring>
#include "Eigen/Dense"
#include "common.h"

namespace disk_hivf {
    Int init_centers(Eigen::Map<RMatrixXf> & features, 
    Eigen::Map<RMatrixXf> & centers, Int type);
    
    Int kmeans(std::vector<float> &features_data, int dim, Int k, Int epoch, 
        Int batch_size, Int centers_select_type,
        std::vector<float> & centers_data,
        std::vector<Int> & assign, double& ret_loss);

    double kmeans_core(Eigen::Map<RMatrixXf> & batch_features,
        Eigen::Map<RMatrixXf>& centers,
        Eigen::VectorXf & centers_squa_norm,
        std::vector<Int> & assign);

}