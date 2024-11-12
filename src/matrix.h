#pragma once
#include <iostream>
#include "Eigen/Dense"
#include <algorithm>
#include <vector>
#include "common.h"

namespace disk_hivf {
    // 根据行向量的模对矩阵进行排序
    template <typename Derived>
    void sort_rows_by_squa_norm_desc(Eigen::MatrixBase<Derived>& matrix) {
        // 获取矩阵的行数和列数
        Int rows = matrix.rows();

        // 创建一个向量来存储每一行的模和行索引
        std::vector<std::pair<double, Int>> norms(rows);

        // 计算每一行的模
        for (Int i = 0; i < rows; ++i) {
            norms[i] = {matrix.row(i).squaredNorm(), i};
        }

        // 根据模的大小进行排序
        sort(norms.begin(), norms.end(), 
            [](const std::pair<double, Int>& a, const std::pair<double, Int>& b) {
                return a.first > b.first;
            });

        // 按照排序后的索引重新排列矩阵的行
        RMatrixXf tmp(matrix.rows(), matrix.cols());
        for (Int i = 0; i < rows; ++i) {
            Int target = norms[i].second;
            tmp.row(i) = matrix.row(target);
        }
        matrix = tmp;
    }


    template <typename Derived>
    void sort_rows_by_vec(Eigen::MatrixBase<Derived>& matrix, std::vector<Int> & vec) {
        // 获取矩阵的行数和列数
        Int rows = matrix.rows();

        // 创建一个向量来存储每一行的模和行索引
        std::vector<std::pair<double, Int>> norms(rows);

        // 计算每一行的模
        for (Int i = 0; i < rows; ++i) {
            norms[i] = {vec[i], i};
        }

        // 根据模的大小进行排序
        sort(norms.begin(), norms.end(), 
            [](const std::pair<double, Int>& a, const std::pair<double, Int>& b) {
                return a.first > b.first;
            });

        // 按照排序后的索引重新排列矩阵的行
        RMatrixXf tmp(matrix.rows(), matrix.cols());
        for (Int i = 0; i < rows; ++i) {
            Int target = norms[i].second;
            tmp.row(i) = matrix.row(target);
        }
        matrix = tmp;
    }


    // 计算距离矩阵
    template <typename DerivedA, typename DerivedB>
    RMatrixDf computeDistanceMatrix(const Eigen::MatrixBase<DerivedA>& A,
        const Eigen::MatrixBase<DerivedB>& B, bool b_had_transpose = false) {
        Eigen::VectorXf A_sq_sum = A.rowwise().squaredNorm();
        RMatrixDf distances;
        if (b_had_transpose) {
            distances = A * B * -2;
            Eigen::RowVectorXf B_sq_sum = B.colwise().squaredNorm();
            distances.rowwise() += B_sq_sum;
        } else {
            distances = A * B.transpose() * (-2);
            Eigen::VectorXf B_sq_sum = B.rowwise().squaredNorm();
            distances.rowwise() += B_sq_sum.transpose();
        }
        distances.colwise() += A_sq_sum;
        return distances;
    }


    // 计算距离矩阵
    template <typename DerivedA, typename DerivedB>
    RMatrixDf computeDistanceMatrix_Bsq(const Eigen::MatrixBase<DerivedA>& A,
        const Eigen::MatrixBase<DerivedB>& B,
        Eigen::VectorXf& B_squa_norm) {
        Eigen::VectorXf A_sq_sum = A.rowwise().squaredNorm();
        RMatrixDf distances;
        distances = A * B.transpose() * (-2);
        distances.rowwise() += B_squa_norm.transpose();
        distances.colwise() += A_sq_sum;
        return distances;
    }


    template <typename DerivedA, typename DerivedB>
    RMatrixDf computeDistanceMatrix_BTsq(const Eigen::MatrixBase<DerivedA>& A,
        const Eigen::MatrixBase<DerivedB>& B,
        Eigen::RowVectorXf& B_squa_norm) {
        Eigen::VectorXf A_sq_sum = A.rowwise().squaredNorm();
        RMatrixDf distances;
        distances = A * B * (-2);
        distances.rowwise() += B_squa_norm;
        distances.colwise() += A_sq_sum;
        return distances;
    }

    // 找到每个向量的 topk 最近邻
    inline std::vector<std::vector<std::pair<float, Int>>> findTopKNeighbors(const RMatrixDf& distances, Int topk) {
        Int m = distances.rows();
        Int n = distances.cols();
        std::vector<std::vector<std::pair<float, Int>>> topk_indices(m, std::vector<std::pair<float, Int>>(topk));

        for (Int i = 0; i < m; ++i) {
            std::vector<std::pair<float, Int>> dist_indices(n);
            for (Int j = 0; j < n; ++j) {
                dist_indices[j] = {distances(i, j), j};
            }
            std::partial_sort(dist_indices.begin(), dist_indices.begin() + topk, dist_indices.end());
            for (Int k = 0; k < topk; ++k) {
                topk_indices[i][k] = dist_indices[k];
            }
        }
        return topk_indices;
    }

    template <typename DerivedA, typename DerivedB>
    std::vector<std::vector<std::pair<float, Int>>> findTopKNeighbors(
        const Eigen::MatrixBase<DerivedA>& A,
        const Eigen::MatrixBase<DerivedB>& B, Int topk) {
        // 计算距离矩阵
        Eigen::MatrixXf distances = computeDistanceMatrix(A, B);
        // 找到每个向量的 topk 最近邻
        std::vector<std::vector<std::pair<float, Int>>> topk_neighbors = findTopKNeighbors(distances, topk);
        return topk_neighbors;
    }
    /*
    int partition(std::vector<int>& vec1, std::vector<float>& vec2,
        int left, int right, int pivotIndex);

    void quickSelect(std::vector<int>& vec1, std::vector<float>& vec2,
        int left, int right, int k);
    
    void topKByVec2(std::vector<int>& vec1, std::vector<float>& vec2,
        int k);

    inline int partition(std::vector<int>& vec1, std::vector<float>& vec2,
        int left, int right, int pivotIndex) {
        float pivotValue = vec2[pivotIndex];
        std::swap(vec2[pivotIndex], vec2[right]);
        std::swap(vec1[pivotIndex], vec1[right]);
        int storeIndex = left;

        for (int i = left; i < right; ++i) {
            if (vec2[i] < pivotValue) { // 注意这里是小于号，因为我们要取top k 小的元素
                if (i != storeIndex) {
                    std::swap(vec2[storeIndex], vec2[i]);
                    std::swap(vec1[storeIndex], vec1[i]);
                }
                ++storeIndex;
            }
        }
        std::swap(vec2[right], vec2[storeIndex]);
        std::swap(vec1[right], vec1[storeIndex]);
        return storeIndex;
    }
    */

    inline int partition(std::vector<int>& vec1, std::vector<float>& vec2,
        int left, int right, int pivotIndex) {
        int pivotValue1 = vec1[pivotIndex];
        float pivotValue2 = vec2[pivotIndex];
        std::swap(vec2[pivotIndex], vec2[left]);
        std::swap(vec1[pivotIndex], vec1[left]);
        int i = left;
        int j = right;

        while (i < j) {
            while (i < j && vec2[j] >= pivotValue2) j--;
            vec1[i] = vec1[j];
            vec2[i] = vec2[j];
            while (i < j && vec2[i] <= pivotValue2) i++;
            vec1[j] = vec1[i];
            vec2[j] = vec2[i];
        }
        vec1[i] = pivotValue1;
        vec2[i] = pivotValue2;

        return i;
    }

    inline void quickSelect(std::vector<int>& vec1, std::vector<float>& vec2,
        int left, int right, int k) {
        if (left == right) return;

        int pivotIndex = left + (right - left) / 2;
        pivotIndex = partition(vec1, vec2, left, right, pivotIndex);

        if (k == pivotIndex) {
            return;
        } else if (k < pivotIndex) {
            quickSelect(vec1, vec2, left, pivotIndex - 1, k);
        } else {
            quickSelect(vec1, vec2, pivotIndex + 1, right, k);
        }
    }

    inline void topKByVec2(std::vector<int>& vec1, std::vector<float>& vec2, int k) {
        if (k <= 0 || k > (int)vec2.size()) return;

        quickSelect(vec1, vec2, 0, vec2.size() - 1, k);

        // 现在 vec1 和 vec2 的前 k 个元素是 vec2 中最小的 k 个元素
        vec1.resize(k);
        vec2.resize(k);
    }


}