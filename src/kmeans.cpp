#include "common.h"
#include <iostream>
#include "kmeans.h"
#include "Eigen/Dense"
#include "random.h"
#include <random>
#include <algorithm>
#include <vector>
#include "unity.h"
#include "matrix.h"

namespace disk_hivf {
    Int init_centers(Eigen::Map<RMatrixXf> & features, 
            Eigen::Map<RMatrixXf> & centers, Int type) {
        //TimeStat ts("init_centers ");
        Kiss32Random ks(rand());
        Int ret = 0;
        if (1 == type) {
            memcpy(centers.data(), features.data(), 
                centers.rows() * centers.cols() * sizeof(float));
        } else if (2 == type) {
            std::vector<uint32_t> m_nums;
            ret = rand_m_nums(ks, features.rows(), centers.rows(), m_nums);
            if (ret != 0) {
                return -1;
            }
            for (Int i = 0; i < centers.rows(); i++) {
                centers.row(i) = features.row(m_nums[i]);
            }
        } else if (3 == type) {
            Int n_sample = std::min(features.rows(), centers.rows() * 8);
            //Int n_sample  = features.rows();
            RMatrixXf sample_mat(n_sample, features.cols());
            std::vector<uint32_t> m_nums;
            ret = rand_m_nums(ks, features.rows(), n_sample, m_nums);
            for (Int i = 0; i < n_sample; i++) {
                sample_mat.row(i) = features.row(m_nums[i]);
            }
            centers.row(0) = sample_mat.row(0);
            std::vector<float> min_distance(n_sample,
                            std::numeric_limits<float>::max());
            for (Int i = 1; i < centers.rows(); i++) {
                #pragma omp parallel for
                for (Int j = 0; j < n_sample; j ++) {
                    float distance = (sample_mat.row(j) 
                                - centers.row(i - 1)).squaredNorm();
                    //distance = std::pow(distance, 1);
                    if (distance < min_distance[j]) {
                        min_distance[j] = distance;
                    }
                }
                randDist<float> rand_dist(ks, min_distance);
                centers.row(i) = sample_mat.row(rand_dist.sample());
            }
        } else {
            return -1;
        }
        return 0;
    }

    double kmeans_core(Eigen::Map<RMatrixXf> & batch_features,
        Eigen::Map<RMatrixXf>& centers,
        Eigen::VectorXf & centers_squa_norm,
        std::vector<Int> & assign) {
        /*
        Eigen::VectorXf batch_features_squa_norm = batch_features.rowwise().squaredNorm();
        //Eigen::VectorXf centers_squa_norm = centers.rowwise().squaredNorm();
        RMatrixDf A = batch_features * centers.transpose();
        A *= -2;
        A.colwise() += batch_features_squa_norm;
        A.rowwise() += centers_squa_norm.transpose();
        */
        
        RMatrixDf A = computeDistanceMatrix_Bsq(batch_features, centers, centers_squa_norm);
        
        Eigen::VectorXi minIndices(A.rows());
        for (Int i = 0; i < A.rows(); ++i) {
            A.row(i).minCoeff(&minIndices(i));
        }
        double loss = 0;
        assign.resize(minIndices.size());
        for (Int i = 0; i < minIndices.size(); i++) {
            assign[i] = minIndices(i);
            loss += A(i, minIndices(i));
        }

        return loss;
    }

    double kmeans_inference(std::vector<float> &features_data,
        std::vector<float> & centers_data, int dim, Int batch_size,
        std::vector<Int> & assign) {
        Int k = centers_data.size() / dim;
        Int numVecs = features_data.size() / dim;
        double loss = 0;
        Int ret = 0;
        Eigen::Map<RMatrixXf> centers(centers_data.data(), k, dim);
        Eigen::Map<RMatrixXf> features(features_data.data(), numVecs, dim);
        Eigen::VectorXf centers_squa_norm = centers.rowwise().squaredNorm();
        assign.resize(numVecs, 0);
        #pragma omp parallel for reduction(+:loss)
        for (Int i = 0; i < numVecs; i += batch_size) {
            Int curr_batch_size = std::min(batch_size, numVecs - i);
            float * data_ptr = features.data() + i * dim;
            Eigen::Map<RMatrixXf> batch_features(data_ptr, curr_batch_size, dim);
            std::vector<Int> tmp_assign;
            double tmp_loss = kmeans_core(batch_features, centers, centers_squa_norm, tmp_assign);
            loss += tmp_loss;
            if (ret != 0) {
                std::cerr << "kmeans kmeans_core err" << std::endl;
            }
            for (Int j = 0; j < curr_batch_size; j++) {
                Int idx = i + j;
                assign[idx] = tmp_assign[j];
            }
        }
        return loss / numVecs;
    }

    Int kmeans(std::vector<float> &features_data, int dim, Int k, Int epoch, 
        Int batch_size, Int centers_select_type, Int sample_num,
        std::vector<float> & centers_data,
        std::vector<Int> & assign, double& ret_loss) {
        Int vecs_num = features_data.size() / dim;

        std::cout << " sample_num=" << sample_num << " vecs_num=" << vecs_num << std::endl;
        if (sample_num <= 0 || sample_num >= vecs_num) {
            std::cout << "no sample kmeans" << std::endl; 
            return kmeans(features_data, dim, k, epoch, 
                batch_size, centers_select_type,
                centers_data, assign, ret_loss);
        } else {
            std::cout << "sample kmeans" << std::endl;
            std::vector<float> sample_features_data(sample_num * dim);
            Kiss32Random ks(rand());
            std::vector<uint32_t> m_nums;
            Int ret = rand_m_nums(ks, vecs_num, sample_num, m_nums);
            if (ret != 0) {
                std::cerr << "kmeans::rand_m_nums fail" << std::endl;
                return ret; 
            }
            for (size_t i = 0; i < m_nums.size(); i++) {
                memcpy(sample_features_data.data() + i * dim,
                    features_data.data() + m_nums[i] * dim,
                    dim * sizeof(float));
            }
            std::vector<Int> __assign;
            double __ret_loss;

            ret = kmeans(sample_features_data, dim, k, epoch, batch_size,
                centers_select_type, centers_data, __assign, __ret_loss);
            std::cout << " __ret_loss=" << __ret_loss << std::endl;
            if (ret != 0) {
                std::cerr << "kmeans::kmeans fail" << std::endl;
                return ret; 
            }
            ret_loss = kmeans_inference(features_data, centers_data, dim, batch_size, assign);
        }
        
        return 0;
    }

    Int kmeans(std::vector<float> &features_data, int dim, Int k, Int epoch, 
        Int batch_size, Int centers_select_type,
        std::vector<float> & centers_data,
        std::vector<Int> & assign, double& ret_loss) {
        TimeStat ts("kmeans ");
        Int numVecs = features_data.size() / dim;
        if (k > numVecs || k <= 0 || numVecs <= 0) {
            std::cerr << "k and numVecs para err k > numVecs k=" 
                << k << " numVecs=" << numVecs << std::endl;
            return -1;
        }
        Eigen::Map<RMatrixXf> features(features_data.data(), numVecs, dim);
        // std::cout << features << std::endl;
        centers_data.resize(k * dim, 0);
        Eigen::Map<RMatrixXf> centers(centers_data.data(), k, dim);
        assign.resize(numVecs);
        ret_loss = std::numeric_limits<double>::max();
        
        std::vector<float> training_centers_data(k * dim, 0);
        Eigen::Map<RMatrixXf> training_centers(training_centers_data.data(), k, dim);
        Int ret = 0;
        ret = init_centers(features, training_centers, centers_select_type);
        if (ret != 0) {
            std::cerr << "kmeans init_centers err" << std::endl;
            return -1;
        }
        
        Kiss32Random ks(rand());
        
        std::vector<float> tmp_centers_data(k * dim, 0);
        std::vector<Int> training_assign;
        
        for (Int _ = 0; _ < epoch; _++) {
            //TimeStat ts("epoch=" + num2str(_));
            tmp_centers_data.resize(k * dim, 0);
            Eigen::Map<RMatrixXf> tmp_centers(tmp_centers_data.data(), k, dim);
            std::vector<Int> nassign(k, 0);
            double loss = 0;
            training_assign.resize(numVecs, -1);
            Eigen::VectorXf centers_squa_norm = training_centers.rowwise().squaredNorm();
            #pragma omp parallel for reduction(+:loss)
            for (Int i = 0; i < numVecs; i += batch_size) {
                Int curr_batch_size = std::min(batch_size, numVecs - i);
                float * data_ptr = features.data() + i * dim;
                Eigen::Map<RMatrixXf> batch_features(data_ptr, curr_batch_size, dim);
                std::vector<Int> tmp_assign;
                double tmp_loss = kmeans_core(batch_features, training_centers, centers_squa_norm, tmp_assign);
                loss += tmp_loss;
                if (ret != 0) {
                    std::cerr << "kmeans kmeans_core err" << std::endl;
                }
                
                for (Int j = 0; j < curr_batch_size; j++) {
                    Int idx = i + j;
                    training_assign[idx] = tmp_assign[j];
                    //#pragma omp critical
                    {
                        tmp_centers.row(training_assign[idx]) += features.row(idx);
                        nassign[training_assign[idx]]++;
                    }
                }
                

            }
            double avg_center_size = 0;
            #pragma omp parallel for reduction(+:avg_center_size)
            for (Int i = 0; i < k; i++) {
                double tmp_avg_center_size = nassign[i] * 1.0 - numVecs * 1.0 / k;
                tmp_avg_center_size *= tmp_avg_center_size;
                avg_center_size += tmp_avg_center_size;
                if (nassign[i] > 0) {
                    tmp_centers.row(i) /= nassign[i];
                } else {
                    Int rd = ks.kiss() % features.rows();
                    tmp_centers.row(i) = features.row(rd);
                }
            }
            avg_center_size /= k;
            avg_center_size = std::sqrt(avg_center_size);
            training_centers = tmp_centers;
            loss /= numVecs;
            //std::cout << "kmeans epoch=" << _ << " loss=" << loss 
            //    << " avg_center_size = " << avg_center_size << std::endl;
            if (loss < ret_loss) {
                centers = training_centers;
                ret_loss = loss;
                memcpy(assign.data(), training_assign.data(), training_assign.size() * sizeof(Int));
            }
        }
        return 0;
    }
}
