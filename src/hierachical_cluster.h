#pragma once
#include "kmeans.h"
#include <iostream>
#include <vector>
#include "Eigen/Dense"
#include "file_read_write.h"
#include "common.h"
#include "conf.h"
#include "heap.h"
#include <cmath>
#include <algorithm>
#include <queue>

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
        FeatureAssign() {}
        FeatureAssign(Uint feature_id, Uint first_center_id, Uint second_center_id, float distance):
            m_feature_id(feature_id), m_first_center_id(first_center_id),
            m_second_center_id(second_center_id), m_distance(distance) {}
        bool operator < (const FeatureAssign & other) const {
            return m_distance < other.m_distance;
        }
        Uint m_feature_id;
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
            if (m_file_id == other.m_file_id && other.m_offset - (m_offset + item_size * m_len) == 0) {
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


    struct Result {
        Result(Int vec_id, float distance): m_vec_id(vec_id), m_distance(distance) {}
        inline bool operator < (const Result & other) const {
            return m_distance < other.m_distance;
        }
        Int m_vec_id;
        float m_distance;
    };

    struct SearchingBlock {
        SearchingBlock(): m_file_id(-1), m_offset(-1), m_max_offset(0) {
            m_min_distance = std::numeric_limits<float>::max();
        }
        inline void push_back(SearchingCell & search_cell, Int item_size) {
            m_search_cells.push_back(search_cell);
            if (-1 == m_file_id) {
                m_file_id = search_cell.m_file_id;
            }
            if (-1 == m_offset) {
                m_offset = search_cell.m_offset;
            }
            m_max_offset = search_cell.m_offset + item_size * search_cell.m_len;
            m_min_distance = std::min(m_min_distance, search_cell.m_distance);
        }

        inline bool operator < (const SearchingBlock & other) const {
            return m_min_distance < other.m_min_distance;
        }

        inline void pop_front() {
            m_search_cells.pop_front();
            if (!m_search_cells.empty()) {
                m_offset = m_search_cells.front().m_offset;
            } else {
                m_offset = -1;
            }
        }

        inline void pop_back(const Int item_size) {
            m_search_cells.pop_back();
            if (!m_search_cells.empty()) {
                m_max_offset = m_search_cells.back().m_offset 
                    + m_search_cells.back().m_len * item_size;
            } else {
                m_max_offset = -1;
            }
        }

        std::deque<SearchingCell> m_search_cells;
        Int m_file_id;
        Int m_offset;
        Int m_max_offset;
        float m_min_distance;
        std::vector<char> m_data;
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
                std::vector<std::pair<Int, float>> & result);
            Int search(const Eigen::Ref<const Eigen::RowVectorXf> & feature, Int topk,
                std::vector<Int> & result);
            Int search(std::vector<float> & feature_data, Int topk,
                std::vector<std::pair<Int, float>> & result);
            Int search(std::vector<float> & feature_data, Int topk,
                std::vector<Int> & result);
        public:
            std::vector<Int> m_time_stat;    

        private:
            Int init_edge_info();
            std::vector<Int> make_second_centers_disk_order();
            Int rerank_disk_order(const std::vector<FeatureAssign>& features_assign,
                const std::vector<Int>& disk_order);

            void make_search_block(std::vector<SearchingCell> & search_cells,
                std::vector<SearchingBlock> & search_blocks);

            template <typename Derived>
                Int findTopkSecondCenters(const Eigen::MatrixBase<Derived>& batch_features,
                    std::vector<std::vector<std::pair<float, Int>>> & topkfirst_center,
                    std::vector<LimitedMaxHeap<FeatureAssign>> & heap_vecs,
                    bool empty_cell_fillter = false) {
                    for (Int features_id = 0; features_id < batch_features.rows(); features_id++) {
                        const Eigen::RowVectorXf & feature = batch_features.row(features_id);
                        std::vector<std::pair<float, Int>> & first_center_dists
                            = topkfirst_center[features_id];
                        LimitedMaxHeap<FeatureAssign> & heap = heap_vecs[features_id];
                        for (Int i = 0; i < first_center_dists.size(); i++) {
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
                            Eigen::Map<RMatrixXf> second_centers_cuted(m_second_centers_data.data(), second_cut, m_conf.m_dim);
                            Eigen::RowVectorXf query2second_centers_dist = feature * second_centers_cuted.transpose() * (-2);
                            for (Int j = 0; j < second_cut; j++) {
                                query2second_centers_dist(j) +=
                                    (first_center_dist + m_first2second_edges_stationary_dist(first_center_id, j));
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

            //一级中心到二级中心的距离公共计算值 T * T + 2S * T 
            //方便|q-S-T|的计算 |q-S-T| = q*q-2qS-2qT+(S*S+ T*T+2ST)
            Eigen::Map<RMatrixDf> m_first2second_edges_stationary_dist;
            
            // len = first_centers_num
            // m_first_min_stationary_dist =
            // m_first2second_edges_stationary_dist.rowwise().min();
            Eigen::VectorXf m_first_min_stationary_dist;
            
            // len = first_centers_num * second_centers_num
            std::vector<DataIndex> m_first2second_cells;
            
            FileReadWriter m_file_read_writer;
    };
}