#pragma once
#include "common.h"
#include "kmeans.h"
#include <iostream>
#include <vector>
#include "Eigen/Dense"
#include "file_read_write.h"
#include "conf.h"
#include "heap.h"
#include <cmath>
#include <algorithm>
#include <queue>
#include <future>
#include <stdexcept>
#include <mutex>
#include "matrix.h"
#include "thread_pool.h"
#include "lru_cache.h"

namespace disk_hivf {
    struct DataIndex{
        DataIndex() {
            m_offset = std::numeric_limits<Int>::max();
            m_len = 0;
            m_radius = 0;
        }
        DataIndex(Int offset, Int len, float radius): 
            m_offset(offset), m_len(len), m_radius(radius) {}
        inline void print() {
            std::cout << m_offset << " " << m_len << " " << m_radius << std::endl;
        }
        Int m_offset;
        int m_len;
        float m_radius;
    };

    struct FeatureAssign {
        FeatureAssign(): m_feature_id(0), m_first_center_id(0), m_second_center_id(0) {
            m_distance = std::numeric_limits<float>::max();
        }
        FeatureAssign(FeatureId feature_id, Uint first_center_id, Uint second_center_id, float distance):
            m_feature_id(feature_id), m_first_center_id(first_center_id),
            m_second_center_id(second_center_id), m_distance(distance) {}
        bool operator < (const FeatureAssign & other) const {
            return m_distance < other.m_distance;
        }
        inline void print() {
            std::cout << " m_feature_id = " << m_feature_id
            << " m_first_center_id = " << m_first_center_id
            << " m_second_center_id = " << m_second_center_id
            << " m_distance = " << m_distance
            << std::endl;
        }
        
        FeatureId m_feature_id;
        Uint m_first_center_id;
        Uint m_second_center_id;
        float m_distance;
    };

    struct DiskOrderRankMember {
        DiskOrderRankMember(Int id, Int first_center_id,
            Int second_center_id,
            Int second_center_disk_order,
            float distance): 
            m_id(id), m_first_center_id(first_center_id),
            m_second_center_id(second_center_id),
            m_second_center_disk_order(second_center_disk_order),
            m_distance(distance){
        }
        Int m_id;
        Int m_first_center_id;
        Int m_second_center_id;
        Int m_second_center_disk_order;
        float m_distance;
        inline bool operator < (const DiskOrderRankMember & other) const {
            if (m_first_center_id != other.m_first_center_id) {
                return m_first_center_id < other.m_first_center_id;
            } else if (m_second_center_disk_order != other.m_second_center_disk_order) {
                return m_second_center_disk_order < other.m_second_center_disk_order;
            } else {
                return m_distance > other.m_distance;
            }
            
        }
    };

    struct SearchingCell {
        SearchingCell(Int file_id, Int cell_id, float distance, Int offset,
            Int len, double radius
            ): m_file_id(file_id),
            m_cell_id(cell_id), m_distance(distance),
            m_offset(offset), m_len(len), m_radius(radius) {
        }

        inline bool operator < (const SearchingCell & other) const {
            if (m_file_id != other.m_file_id) {
                return m_file_id < other.m_file_id;
            } else {
                return m_offset < other.m_offset;
            }
        }

        inline bool is_continuous(const SearchingCell & other, Int item_size) const {
            if (m_file_id == other.m_file_id && other.m_offset - (m_offset + item_size * m_len) <= 0) {
                return true;
            } else {
                return false;
            }
        }

        inline void print() {
            std::cout << " m_file_id = " << m_file_id
            << " m_cell_id = " << m_cell_id
            << " m_distance = " << m_distance
            << " m_offset = " << m_offset
            << " m_len = " << m_len
            << " m_radius = " << m_radius << std::endl;
        }

        Int m_file_id;
        Int m_cell_id;
        float m_distance;
        Int m_offset;
        Int m_len;
        double m_radius;
    };

    struct CellData {
        CellData(std::vector<FeatureId> & ids, std::vector<char> & data) {
            m_ids = std::move(ids);
            m_data = std::move(data);
        }
        std::vector<FeatureId> m_ids;
        std::vector<char> m_data;
    };

    struct SearchingBlock {
        SearchingBlock(): m_file_id(-1), m_offset(-1), m_max_offset(0) {
            m_min_distance = std::numeric_limits<float>::max();
        }
        inline void push_back(SearchingCell & search_cell, Int item_size) {
            if (-1 == m_file_id) {
                m_file_id = search_cell.m_file_id;
            }
            if (-1 == m_offset) {
                m_offset = search_cell.m_offset;
            }
            m_max_offset = search_cell.m_offset + item_size * search_cell.m_len;
            m_min_distance = std::min(m_min_distance, search_cell.m_distance);
            m_cell_vecs.emplace_back(search_cell.m_cell_id, search_cell.m_len);
        }

        inline bool operator < (const SearchingBlock & other) const {
            return m_min_distance < other.m_min_distance;
        }

        Int m_file_id;
        Int m_offset;
        Int m_max_offset;
        float m_min_distance;
        //std::vector<char> m_data;
        std::vector<std::pair<Int, Int>> m_cell_vecs; 
    };


    class HierachicalCluster{
        public:
            HierachicalCluster(Conf & conf);
            Int init();
            Int train_model();
            Int save_model();
            Int load_model();
            Int build_index();
            Int save_index();
            Int load_index();
            Int search(const Eigen::Ref<const Eigen::RowVectorXf> & feature, Int topk,
                std::vector<std::pair<FeatureId, float>> & result, Int use_cache = 0);
            Int search(const Eigen::Ref<const Eigen::RowVectorXf> & feature, Int topk,
                std::vector<FeatureId> & result, Int use_cache = 0);
            Int search(std::vector<float> & feature_data, Int topk,
                std::vector<std::pair<FeatureId, float>> & result, Int use_cache = 0);
            Int search(std::vector<float> & feature_data, Int topk,
                std::vector<FeatureId> & result, Int use_cache = 0);
        public:
            std::vector<Int> m_time_stat;    

        private:
            inline Int get_tot_offset(Int file_id, Int offset, Int item_size) const {
                return m_file_tot_offset[file_id] + offset / item_size;
            }

            inline Int get_file_id(Int x) {
                Int n = m_conf.m_first_cluster_num;
                Int m = m_conf.m_index_file_num;
                Int bucket_size = n / m;
                if (x / bucket_size < m) {
                    return x / bucket_size;
                } else {
                    return x % m;
                }
            }

            std::future<std::vector<char>> read_file_async(Int file_id, Int offset, Int len);

            Int init_edge_info();
            
            std::vector<Int> make_centers_disk_order(Eigen::Map<RMatrixXf> & centers, Int centers_num);

            Int rerank_disk_order(const std::vector<FeatureAssign>& features_assign,
                const std::vector<Int>& disk_order);

            void make_search_block(
                std::vector<SearchingCell> & search_cells,
                std::vector<SearchingBlock> & search_blocks, Int item_size);

            template <typename Derived>
                Int findTopkSecondCenters(const Eigen::MatrixBase<Derived>& batch_features,
                    std::vector<std::vector<std::pair<float, Int>>> & topkfirst_center,
                    std::vector<LimitedMaxHeap<FeatureAssign>> & heap_vecs,
                    bool empty_cell_fillter = false) {
                    //TimeStat ts("findTopkSecondCenters");
                    Eigen::Map<RMatrixXf> second_centers(m_second_centers_data.data(), m_conf.m_second_cluster_num, m_conf.m_dim);
                    RMatrixDf qt = batch_features * second_centers.transpose() * (-2);
                    //ts.TimeMark("make qt");
                    for (Int features_id = 0; features_id < batch_features.rows(); features_id++) {
                        std::vector<std::pair<float, Int>> & first_center_dists
                            = topkfirst_center[features_id];
                        LimitedMaxHeap<FeatureAssign> & heap = heap_vecs[features_id];
                        //Eigen::Map<RMatrixXf> second_centers(m_second_centers_data.data(), m_conf.m_second_cluster_num, m_conf.m_dim);
                        Eigen::RowVectorXf query2second_centers_dist(m_conf.m_second_cluster_num);
                        
                        for (size_t i = 0; i < first_center_dists.size(); i++) {
                            float first_center_dist = first_center_dists[i].first;
                            Int first_center_id = first_center_dists[i].second;
                            
                            Int second_cut;
                            if (heap.is_full()) {
                                float cut_dist = heap.top().m_distance;

                                for (second_cut = 0; second_cut < m_conf.m_second_cluster_num; second_cut++) {
                                    if (std::sqrt(first_center_dist) - std::sqrt(m_second_centers_squa_norm(second_cut)) 
                                        > std::sqrt(cut_dist)) {
                                        break;
                                    }
                                }
                            } else {
                                second_cut = m_conf.m_second_cluster_num;
                            }
                            if (second_cut <= 0) {
                                continue;
                            }
                            for (Int j = 0; j < second_cut; j++) {
                                query2second_centers_dist(j) = qt(features_id, j) +
                                    (first_center_dist + m_first2second_edges_stationary_dist(first_center_id, j)) + m_second_centers_squa_norm(j);
                                if (empty_cell_fillter) {
                                    Int cell_id = first_center_id * m_conf.m_second_cluster_num + j;
                                    if (m_first2second_cells[cell_id].m_len <= 0) {
                                        continue;
                                    }
                                }
                                heap.push(FeatureAssign(features_id, first_center_id, j, query2second_centers_dist(j)));
                            }
                        }
                    }
                    return 0;
                }


            template <typename Derived>
                Int findTopkSecondCenters2(const Eigen::MatrixBase<Derived>& batch_features,
                    std::vector<std::vector<std::pair<float, Int>>> & topkfirst_center,
                    Int topk,
                    std::vector<std::vector<int>> & batch_cell_ids,
                    std::vector<std::vector<float>> & batch_dists_data) {
                    //TimeStat ts("findTopkSecondCenters2");
                    Eigen::Map<RMatrixXf> second_centers(m_second_centers_data.data(), m_conf.m_second_cluster_num, m_conf.m_dim);
                    RMatrixDf qt = batch_features * second_centers.transpose() * (-2);
                    std::vector<float> first_center_dists_data;
                    std::vector<float> alpha_data;
                    //ts.TimeMark("findTopkSecondCenters2 qt");
                    for (Int features_id = 0; features_id < batch_features.rows(); features_id++) {
                        auto & cell_ids = batch_cell_ids[features_id];
                        auto & dists_data = batch_dists_data[features_id];
                        std::vector<std::pair<float, Int>> & first_center_dists
                            = topkfirst_center[features_id];
                        first_center_dists_data.reserve(first_center_dists.size());
                        for (auto & pair: first_center_dists) {
                            first_center_dists_data.push_back(pair.first);
                        }
                        Eigen::Map<Eigen::VectorXf> first_center_dists_vec(
                            first_center_dists_data.data(),
                            first_center_dists_data.size()
                        );
                        dists_data.resize(first_center_dists.size() * m_conf.m_second_cluster_num);
                        alpha_data.resize(first_center_dists.size() * m_conf.m_second_cluster_num);
                        for (size_t i = 0; i < first_center_dists.size(); i++) {
                            Int first_center_id = first_center_dists[i].second;
                            memcpy(dists_data.data() + i * m_conf.m_second_cluster_num,
                                m_first2second_edges_stationary_dist_data.data() 
                                + first_center_id * m_conf.m_second_cluster_num,
                                m_conf.m_second_cluster_num * sizeof(float)
                            );

                            memcpy(alpha_data.data() + i * m_conf.m_second_cluster_num,
                                m_alpha_data.data() 
                                + first_center_id * m_conf.m_second_cluster_num,
                                m_conf.m_second_cluster_num * sizeof(float)
                            );
                        }
                        //ts.TimeMark("findTopkSecondCenters2 memcpy");
                        Eigen::Map<RMatrixXf> dists(
                            dists_data.data(), 
                            first_center_dists.size(),
                            m_conf.m_second_cluster_num
                        );

                        Eigen::Map<RMatrixXf> alpha(
                            alpha_data.data(), 
                            first_center_dists.size(),
                            m_conf.m_second_cluster_num
                        );
                        // std::cout << "alpha = \n" << alpha << std::endl;
                        RMatrixXf alpha_squa_second_squa_norm = alpha.array().square().rowwise() * m_second_centers_squa_norm.transpose().array();
                        RMatrixXf alpha_qt = alpha.array().rowwise() * qt.row(features_id).array();
                        //dists.rowwise() += m_second_centers_squa_norm.transpose();
                        //dists.rowwise() += qt.row(features_id);
                        dists.colwise() += first_center_dists_vec;
                        dists += alpha_squa_second_squa_norm;
                        dists += alpha_qt;
                        //ts.TimeMark("findTopkSecondCenters2 calc");
                        cell_ids.resize(
                            first_center_dists.size() * m_conf.m_second_cluster_num);
                        //ts.TimeMark("findTopkSecondCenters2 resize");
                        for (size_t i = 0; i < first_center_dists.size(); i++) {
                            Int first_center_id = first_center_dists[i].second;
                            for (Int j = 0; j < m_conf.m_second_cluster_num; j++) {
                                Int cell_id = first_center_id * m_conf.m_second_cluster_num + j;
                                Int id = i * m_conf.m_second_cluster_num + j;
                                cell_ids[id] = cell_id;
                            }
                        }
                        //ts.TimeMark("findTopkSecondCenters2 make fa_vec");
                        topKByVec2(cell_ids, dists_data, topk);
                        //ts.TimeMark("findTopkSecondCenters2 push 2 heap");
                    }
                    return 0;
                }

        inline float dynamic_prune_func(float x) {
            return m_conf.m_dynamic_prune_a * x * x + 
                m_conf.m_dynamic_prune_b * x + 
                m_conf.m_dynamic_prune_c;
        }

        private:
            Conf m_conf;
            std::vector<float> m_first_centers_data;
            std::vector<float> m_second_centers_data;
            Eigen::Map<RMatrixXf> m_first_centers;
            Eigen::Map<RMatrixXf> m_second_centers;
            Eigen::VectorXf m_first_centers_squa_norm;
            Eigen::VectorXf m_second_centers_squa_norm;

            // len = first_centers_num * second_centers_num
            std::vector<float> m_first2second_edges_stationary_dist_data;

            //一级中心到二级中心的距离公共计算值 2S * T 
            //方便|q-S-T|的计算 |q-S-T| = q*q-2qS-2qT+(S*S+ T*T+2ST)
            Eigen::Map<RMatrixDf> m_first2second_edges_stationary_dist;

            std::vector<float> m_alpha_data;
            Eigen::Map<RMatrixDf> m_alpha;

            // len = first_centers_num
            // m_first_min_stationary_dist =
            // m_first2second_edges_stationary_dist.rowwise().min();
            // Eigen::VectorXf m_first_min_stationary_dist;
            
            // len = first_centers_num * second_centers_num
            std::vector<DataIndex> m_first2second_cells;
            
            FileReadWriter m_file_read_writer;

            std::vector<Int> m_file_tot_offset;

            std::vector<FeatureId> m_feature_ids;

            std::vector<std::mutex> m_file_mutexs;

            Int m_data_unit_size;

            ThreadPool m_io_thread_pool;

            float m_build_index_loss;

            std::unordered_map<int, CellData> m_cache;
    };
}