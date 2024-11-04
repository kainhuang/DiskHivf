#pragma once

//#define EIGEN_USE_MKL_ALL
//#define EIGEN_VECTORIZE_SSE4_2

#include <inttypes.h>
#include "Eigen/Dense"


namespace disk_hivf {
    typedef float Float;
    typedef uint64_t Uint;
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RMatrixXf;
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RMatrixDf;
    typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RMatrixDi;
    typedef Eigen::MatrixXf CMatrixDf;
    typedef int64_t Int;
    typedef uint32_t FeatureId;
}