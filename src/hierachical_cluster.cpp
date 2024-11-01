#include "kmeans.h"
#include <iostream>
#include <vector>
#include "Eigen/Dense"
#include "file_read_write.h"
#include "common.h"
#include "conf.h"
#include <cstring>
#include "hierachical_cluster.h"
#include "matrix.h"
#include "heap.h"
#include <algorithm>

namespace disk_hivf {
    HierachicalCluster::HierachicalCluster(Conf & conf): m_time_stat(20, 0), m_conf(conf),
        m_first_centers_data(conf.m_first_cluster_num * conf.m_dim),
        m_second_centers_data(conf.m_second_cluster_num * conf.m_dim),
        m_first_centers(m_first_centers_data.data(), conf.m_first_cluster_num, conf.m_dim),
        m_second_centers(m_second_centers_data.data(), conf.m_second_cluster_num, conf.m_dim),
        m_first_centers_squa_norm(conf.m_first_cluster_num),
        m_second_centers_squa_norm(conf.m_second_cluster_num),
        m_first2second_edges_stationary_dist_data(conf.m_first_cluster_num * conf.m_second_cluster_num),
        m_first2second_edges_stationary_dist(m_first2second_edges_stationary_dist_data.data(), 
            conf.m_first_cluster_num, conf.m_second_cluster_num),
        m_first_min_stationary_dist(conf.m_first_cluster_num),
        m_first2second_cells(conf.m_first_cluster_num * conf.m_second_cluster_num),
        m_file_read_writer(conf.m_index_dir, conf.m_index_file_num, conf.m_is_disk),
        m_file_tot_offset(m_conf.m_index_file_num + 1, 0) {
    }

    Int HierachicalCluster::init() {
        Int ret = m_file_read_writer.Init();
        if (ret != 0) {
            //LOG
            return -1;
        }
        return 0;
    }

    Int HierachicalCluster::train_model() {
        int dim;
        Int num_vecs;
        Int ret = 0;
        std::vector<float> features_data;
        ret = readDimVecs(m_conf.m_train_data_file, features_data, dim, num_vecs);
        if (ret != 0) {
            // LOG
            std::cerr << "Error reading training data" << std::endl;
            return -1;
        }
        if (dim != m_conf.m_dim) {
            // LOG
            std::cerr << "Dimension mismatch" << std::endl;
            return -1;
        }
        if (num_vecs != (Int)(features_data.size() / dim)) {
            // LOG
            std::cerr << "Number of vectors mismatch" << std::endl;
            return -1;
        }
        std::vector<float> training_features_data(features_data.size());
        memcpy(training_features_data.data(),
            features_data.data(), 
            features_data.size() * sizeof(float));
        
        double ret_loss;
        double best_loss = std::numeric_limits<double>::max();
        std::vector<Int> assign(num_vecs);
        std::vector<float> first_centers_data(m_conf.m_first_cluster_num * dim);
        std::vector<float> second_centers_data(m_conf.m_second_cluster_num * dim);
        Eigen::Map<RMatrixXf> features(features_data.data(),
            num_vecs, dim);
        Eigen::Map<RMatrixXf> training_features(training_features_data.data(),
            num_vecs, dim);
        Eigen::Map<RMatrixXf> first_centers(first_centers_data.data(),
            m_conf.m_first_cluster_num, dim);
        Eigen::Map<RMatrixXf> second_centers(second_centers_data.data(),
            m_conf.m_second_cluster_num, dim);

        for (Int _ = 0; _ < m_conf.m_hierachical_cluster_epoch; _++) {
            ret = kmeans(training_features_data,
                m_conf.m_dim,
                m_conf.m_first_cluster_num,
                m_conf.m_kmeans_epoch,
                m_conf.m_batch_size,
                m_conf.m_kmeans_centers_select_type,
                first_centers_data,
                assign, ret_loss
            );
            if (ret != 0) {
                // LOG
                std::cerr << "Error in first kmeans" << std::endl;
                return -1;
            }
            std::cerr << "first kmeans epoch= " << _ << " ret_loss = " << ret_loss << std::endl;

            for (Int i = 0; i < training_features.rows(); i++) {
                training_features.row(i) = features.row(i) - first_centers.row(assign[i]);
            }
            
            ret = kmeans(training_features_data,
                m_conf.m_dim,
                m_conf.m_second_cluster_num,
                m_conf.m_kmeans_epoch,
                m_conf.m_batch_size,
                m_conf.m_kmeans_centers_select_type,
                second_centers_data,
                assign, ret_loss
            );
            if (ret != 0) {
                // LOG
                std::cerr << "Error in second kmeans" << std::endl;
                return -1;
            }
            std::cerr << "second kmeans epoch= " << _ << " ret_loss = " << ret_loss << std::endl;
            if (ret_loss < best_loss) {
                best_loss = ret_loss;
                m_first_centers = first_centers;
                m_second_centers = second_centers;
            }
            for (Int i = 0; i < training_features.rows(); i++) {
                training_features.row(i) = features.row(i) - second_centers.row(assign[i]);
            }
        }
        m_first_centers_squa_norm = m_first_centers.rowwise().squaredNorm();
        sort_rows_by_squa_norm_desc(m_second_centers);
        m_second_centers_squa_norm = m_second_centers.rowwise().squaredNorm();
        //std::cout << m_second_centers_squa_norm << std::endl;
        return 0;
    }

    Int HierachicalCluster::save_model() {
        std::ofstream outputFile(m_conf.m_model_file, std::ios::binary);
        if (!outputFile) {
            std::cerr << "Cannot open output file: " << m_conf.m_model_file << std::endl;
            return -1;
        }

        outputFile.write(reinterpret_cast<const char*>(m_first_centers_data.data()),
                        m_first_centers_data.size() * sizeof(float));
        if (!outputFile) {
            std::cerr << "Error writing first centers data to file: " << m_conf.m_model_file << std::endl;
            return -1;
        }

        outputFile.write(reinterpret_cast<const char*>(m_second_centers_data.data()),
                        m_second_centers_data.size() * sizeof(float));
        if (!outputFile) {
            std::cerr << "Error writing second centers data to file: " << m_conf.m_model_file << std::endl;
            return -1;
        }
        //std::cout << m_second_centers << std::endl;
        outputFile.close();
        return 0;
    }

    Int HierachicalCluster::load_model() {
        std::ifstream inputFile(m_conf.m_model_file, std::ios::binary);
        if (!inputFile) {
            std::cerr << "Cannot open input file: " << m_conf.m_model_file << std::endl;
            return -1;
        }

        // Ensure the vectors are of the correct size
        m_first_centers_data.resize(m_conf.m_first_cluster_num * m_conf.m_dim);
        m_second_centers_data.resize(m_conf.m_second_cluster_num * m_conf.m_dim);

        inputFile.read(reinterpret_cast<char*>(m_first_centers_data.data()),
                    m_first_centers_data.size() * sizeof(float));
        if (!inputFile) {
            std::cerr << "Error reading first centers data from file: " << m_conf.m_model_file << std::endl;
            return -1;
        }

        inputFile.read(reinterpret_cast<char*>(m_second_centers_data.data()),
                    m_second_centers_data.size() * sizeof(float));
        if (!inputFile) {
            std::cerr << "Error reading second centers data from file: " << m_conf.m_model_file << std::endl;
            return -1;
        }
        m_first_centers_squa_norm = m_first_centers.rowwise().squaredNorm();
        m_second_centers_squa_norm = m_second_centers.rowwise().squaredNorm();
        //std::cout << m_second_centers << std::endl;
        inputFile.close();
        return 0;
    }
    
    Int HierachicalCluster::init_edge_info() {
        m_first2second_edges_stationary_dist = m_first_centers * m_second_centers.transpose() * 2;
        m_first2second_edges_stationary_dist.rowwise() += m_second_centers_squa_norm.transpose();
        m_first_min_stationary_dist = m_first2second_edges_stationary_dist.rowwise().minCoeff();
        return 0;
    }


    std::vector<Int> HierachicalCluster::make_second_centers_disk_order() {
        RMatrixDf distances = computeDistanceMatrix(m_second_centers, m_second_centers);
        std::vector<Int> tmp_vec;
        std::vector<Int> ret_vec(m_conf.m_second_cluster_num, -1);
        std::vector<bool> mark(m_conf.m_second_cluster_num, false);
        if (m_conf.m_second_cluster_num <= 0) {
            return ret_vec;
        }
        mark[m_conf.m_second_cluster_num-1] = true;
        tmp_vec.push_back(m_conf.m_second_cluster_num-1);
        Int per_center_id = m_conf.m_second_cluster_num-1;
        for (Int i = 1; i < m_conf.m_second_cluster_num; i++) {
            float min_dist = std::numeric_limits<float>::max();
            float min_id = 0;
            for (Int j = 0; j < m_conf.m_second_cluster_num; j++) {
                if (mark[j]) {
                    continue;
                }
                if (distances(per_center_id, j) < min_dist) {
                    min_dist = distances(per_center_id, j);
                    min_id = j;
                }
            }
            tmp_vec.push_back(min_id);
            mark[min_id] = true;
            per_center_id = min_id;
        }
        for (Int i = 0; i < m_conf.m_second_cluster_num; i++) {
            ret_vec[tmp_vec[i]] = i;
        }
        return ret_vec;
    }

    Int HierachicalCluster::rerank_disk_order(const std::vector<FeatureAssign>& features_assign,
                const std::vector<Int>& disk_order) {
        TimeStat ts("rerank_disk_order ");
        if (m_conf.m_hs_mode) {
            m_feature_ids.resize(m_file_tot_offset[m_conf.m_index_file_num], 0);
        }
        Int nthreads = std::min(m_conf.m_index_file_num, (Int)5);
        #pragma omp parallel for num_threads(nthreads)
        for (Int file_id = 0; file_id < m_file_read_writer.get_file_num(); file_id++) {
            std::vector<char> data;
            Int size = m_file_read_writer.read(file_id, data);
            if (size < 0) {
                //LOG
                //return -1;
            }
            Int item_size = sizeof(FeatureId) + m_conf.m_dim * sizeof(float);
            Int item_num = size / item_size;
            char * ptr = data.data();
            std::vector<DiskOrderRankMember> disk_order_members;
            for (Int i = 0; i < item_num; i++) {
                FeatureId features_id = *(reinterpret_cast<FeatureId *>(ptr + i * item_size));
                Int first_center_id = features_assign[features_id].m_first_center_id;
                Int second_center_id = features_assign[features_id].m_second_center_id;
                Int second_center_disk_order = disk_order[second_center_id];
                float dist = features_assign[features_id].m_distance;
                disk_order_members.push_back(DiskOrderRankMember(i, first_center_id,
                    second_center_id, second_center_disk_order, dist));
            }
            std::sort(disk_order_members.begin(), disk_order_members.end());
            std::vector<char> tmp_data(data.size());
            char * tmp_ptr = tmp_data.data();
            for (Int i = 0; i < item_num; i++) {
                Int target_index = disk_order_members[i].m_id;
                memcpy(tmp_ptr + i * item_size, ptr + target_index * item_size, item_size);
                
            }
            
            Int ret = m_file_read_writer.clear(file_id);
            if (ret < 0) {
                //LOG
                //return -1;
            }
            for (Int i = 0; i < item_num; i++) {
                Int first2second_cells_id = 
                    disk_order_members[i].m_first_center_id * m_conf.m_second_cluster_num
                    + disk_order_members[i].m_second_center_id;
                //std::cout << "ptr1=" << tmp_ptr + i * item_size << std::endl;
                Int begin_offset;
                if (m_conf.m_hs_mode) {
                    FeatureId features_id = *(reinterpret_cast<FeatureId *>(tmp_ptr + i * item_size));
                    begin_offset = m_file_read_writer.write(file_id,
                        tmp_ptr + i * item_size + sizeof(FeatureId),
                        item_size - sizeof(FeatureId));
                    Int features_id_offset = get_tot_offset(file_id, 
                        begin_offset, item_size - sizeof(FeatureId));
                    m_feature_ids[features_id_offset] = features_id;
                    
                } else {
                    begin_offset = m_file_read_writer.write(file_id, tmp_ptr + i * item_size, item_size);
                }
                /*
                Int features_id = *(reinterpret_cast<Int *>(tmp_ptr + i * item_size));
                float * features_ptr = reinterpret_cast<float *>(tmp_ptr + i * item_size + sizeof(Int));
                Eigen::Map<Eigen::RowVectorXf> features_mp(features_ptr, m_conf.m_dim);
                std::cout << begin_offset << std::endl;
                std::cout << features_id << std::endl;
                std::cout << features_mp << std::endl;
                */
                if (begin_offset < 0) {
                    //LOG
                    //return -1;
                }
                m_first2second_cells[first2second_cells_id].m_offset = 
                    std::min(m_first2second_cells[first2second_cells_id].m_offset,
                    begin_offset);
                m_first2second_cells[first2second_cells_id].m_len++;
            }
        }
        return 0;
    }


    Int HierachicalCluster::build_index() {
        TimeStat ts("build_index ");
        if (init_edge_info() != 0) {
            //LOG
            return -1;
        } 
        if (m_file_read_writer.clear() < 0) {
            //LOG
            return -1;
        }
        std::ifstream index_data_file(m_conf.m_index_data_file, std::ios::binary);
        if (!index_data_file) {
            std::cerr << "Cannot open index_data_file: " << m_conf.m_index_data_file << std::endl;
            return -1;
        }
        int dim;
        index_data_file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (!index_data_file) {
            std::cerr << "Error reading dimension from file: " << m_conf.m_index_data_file << std::endl;
            return -1;
        }
        Int vecs_num;
        index_data_file.read(reinterpret_cast<char*>(&vecs_num), sizeof(Int));
        if (!index_data_file) {
            std::cerr << "Error reading numVecs from file: "  << m_conf.m_index_data_file << std::endl;
            return -1;
        }
        Int batch_size = m_conf.m_read_file_batch_size;
        Int offset = 0;
        std::vector<FeatureAssign> features_assign(vecs_num);
        double loss = 0;
        std::vector<Int> file_vec_nums(m_conf.m_index_file_num, 0);
        #pragma omp parallel for reduction(+:loss)
        for (Int i = 0; i < vecs_num; i += batch_size) {
            Int curr_batch_size = 0;
            std::vector<float> batch_features_data;
            Int now_offset = 0;
            #pragma omp critical
            {
                curr_batch_size = std::min(batch_size, vecs_num - i);
                batch_features_data.resize(curr_batch_size * dim);
                index_data_file.read(reinterpret_cast<char*>(batch_features_data.data()),
                    curr_batch_size * dim * sizeof(float));
                if (!index_data_file) {
                    std::cerr << "Error reading offset=" << offset << " file=" 
                        << m_conf.m_index_data_file << std::endl;
                }
                now_offset = offset;
                //std::cout << "now_offset=" << offset << std::endl;
                offset += curr_batch_size;
            }

            Eigen::Map<RMatrixXf> batch_features(batch_features_data.data(), curr_batch_size, dim);
            
            RMatrixDf query2first_distance = batch_features * m_first_centers.transpose() * (-2);
            query2first_distance.colwise() += batch_features.rowwise().squaredNorm();
            query2first_distance.rowwise() += m_first_centers_squa_norm.transpose(); 
            std::vector<std::vector<std::pair<float, Int>>> topkfirst_center = 
                findTopKNeighbors(query2first_distance, m_conf.m_build_index_search_first_center_num);
            Int topk = 1;
            std::vector<LimitedMaxHeap<FeatureAssign>>
                heap_vecs(curr_batch_size, LimitedMaxHeap<FeatureAssign>(topk));
            Int ret = findTopkSecondCenters(batch_features, topkfirst_center, heap_vecs);
            if (ret != 0) {
                std::cerr << "findTopkSecondCenters fail" << std::endl;
            }
            #pragma omp critical
            {
                for (size_t j = 0; j < heap_vecs.size(); j++) {
                    FeatureId idx = now_offset + j;
                    features_assign[idx] = heap_vecs[j].top();
                    features_assign[idx].m_feature_id = idx;
                    //write idx,feature_vec to file dir/first_center_id
                    Int file_id = features_assign[idx].m_first_center_id % m_conf.m_index_file_num;

                    Int first2second_cells_id =
                        features_assign[idx].m_first_center_id * m_conf.m_second_cluster_num 
                        + features_assign[idx].m_second_center_id;
                    //std::cout << file_id << " " << first2second_cells_id << std::endl;
                    m_first2second_cells[first2second_cells_id].m_radius =
                        std::max(m_first2second_cells[first2second_cells_id].m_radius,
                        features_assign[idx].m_distance);
                    loss += features_assign[idx].m_distance;
                    ret = m_file_read_writer.write(file_id, reinterpret_cast<char *>(&idx), sizeof(FeatureId));
                    if (ret < 0) {
                        //LOG
                    }
                    ret = m_file_read_writer.write(file_id, 
                        reinterpret_cast<char *>(batch_features.row(j).data()),
                        dim * sizeof(float));
                    if (ret < 0) {
                        //LOG
                    }
                    file_vec_nums[file_id]++;
                }
            }
        }
        loss /= vecs_num;
        std::cerr << "build index loss = " << loss << std::endl; 
        for (size_t i = 1; i < m_file_tot_offset.size(); i++) {
            m_file_tot_offset[i] = m_file_tot_offset[i-1] + file_vec_nums[i-1];

        }
        for (auto file_tot_offset: m_file_tot_offset) {
            std::cout << file_tot_offset << " ";
        }
        std::cout << std::endl;
        std::vector<Int> second_centers_disk_order = make_second_centers_disk_order();
        Int ret = rerank_disk_order(features_assign, second_centers_disk_order);
        if (ret < 0) {
            std::cerr << "rerank_disk_order fail" << std::endl;
            return -1;
        }
        /*
        {
            std::vector<Int> second_centers_disk_order_rev(second_centers_disk_order.size());
            for (Int i = 0; i < second_centers_disk_order.size(); i++) {
                second_centers_disk_order_rev[second_centers_disk_order[i]] = i;
            }
            for (Int i = 0; i < m_conf.m_first_cluster_num; i++) {
                for (Int j = 0; j < m_conf.m_second_cluster_num; j++) {
                    Int second_centers_id = second_centers_disk_order_rev[j];
                    Int cell_id = i * m_conf.m_first_cluster_num + second_centers_id;
                    m_first2second_cells[cell_id].print();
                }
            }
        }
        */
        //RMatrixDf dist = computeDistanceMatrix(m_second_centers, m_second_centers);
        //std::cout << dist << std::endl;
        return 0;
    }
    
    Int HierachicalCluster::save_index() {
        TimeStat ts("save_index ");
        std::string index_file_name = m_conf.m_index_dir + "/index"; 
        std::ofstream outputFile(index_file_name, std::ios::binary);
        if (!outputFile) {
            std::cerr << "save_index::Cannot open output file: " << index_file_name << std::endl;
            return -1;
        }
       
        outputFile.write(reinterpret_cast<const char*>(m_first2second_edges_stationary_dist_data.data()),
                        m_first2second_edges_stationary_dist_data.size() * sizeof(float));
        
        if (!outputFile) {
            std::cerr << "Error writing m_first2second_edges_stationary_dist_data to file: " 
                << index_file_name << std::endl;
            return -1;
        }
        
        outputFile.write(reinterpret_cast<const char*>(m_first2second_cells.data()),
                        m_first2second_cells.size() * sizeof(DataIndex));
        if (!outputFile) {
            std::cerr << "Error writing m_first2second_cells to file: " 
                << index_file_name << std::endl;
            return -1;
        }

        outputFile.write(reinterpret_cast<const char *>(m_file_tot_offset.data()), m_file_tot_offset.size() * sizeof(Int));
        if (!outputFile) {
            std::cerr << "Error writing m_file_tot_offset to file: " 
                << index_file_name << std::endl;
            return -1;
        }

        if (m_conf.m_hs_mode) {
            outputFile.write(reinterpret_cast<const char*>(m_feature_ids.data()),
                m_feature_ids.size() * sizeof(FeatureId));
            if (!outputFile) {
                std::cerr << "Error writing m_feature_ids to file: " 
                    << index_file_name << std::endl;
                return -1;
            }
        }
        //std::cout << m_first2second_edges_stationary_dist << std::endl;
        //for (auto & cell: m_first2second_cells) {
        //    cell.print();
        //}
        //std::cout << m_first_min_stationary_dist << std::endl;
        outputFile.close();
        return 0;
    }
    
    Int HierachicalCluster::load_index() {
        TimeStat ts("load_index ");
        std::string index_file_name = m_conf.m_index_dir + "/index";
        std::ifstream inputFile(index_file_name, std::ios::binary);
        if (!inputFile) {
            std::cerr << "load_index::Cannot open input file: " << index_file_name << std::endl;
            return -1;
        }
        m_first2second_edges_stationary_dist_data.resize(m_conf.m_first_cluster_num * m_conf.m_second_cluster_num);
        inputFile.read(reinterpret_cast<char*>(m_first2second_edges_stationary_dist_data.data()),
                    m_first2second_edges_stationary_dist_data.size() * sizeof(float));
        if (!inputFile) {
            std::cerr << "Error reading m_first2second_edges_stationary_dist_data from file: " 
                << index_file_name << std::endl;
            return -1;
        }
        m_first2second_cells.resize(m_conf.m_first_cluster_num * m_conf.m_second_cluster_num);
        inputFile.read(reinterpret_cast<char*>(m_first2second_cells.data()),
                    m_first2second_cells.size() * sizeof(DataIndex));
        if (!inputFile) {
            std::cerr << "Error reading m_first2second_cells from file: "
                << index_file_name << std::endl;
            return -1;
        }
        m_first_min_stationary_dist = m_first2second_edges_stationary_dist.rowwise().minCoeff();

        m_file_tot_offset.resize(m_conf.m_index_file_num + 1, 0);
        inputFile.read(reinterpret_cast<char *>(m_file_tot_offset.data()),
            m_file_tot_offset.size() * sizeof(Int));
        if (!inputFile) {
            std::cerr << "Error reading m_file_tot_offset from file: "
                << index_file_name << std::endl;
            return -1;
        }
        //for (auto offset: m_file_tot_offset) {
        //    std::cout << offset << std::endl;
        //}
        if (m_conf.m_hs_mode) {
            m_feature_ids.resize(m_file_tot_offset[m_conf.m_index_file_num], 0);
            inputFile.read(reinterpret_cast<char *>(m_feature_ids.data()),
                m_feature_ids.size() * sizeof(FeatureId));
            if (!inputFile) {
                std::cerr << "Error reading m_feature_ids from file: "
                    << index_file_name << std::endl;
                return -1;
            }
        }
        //std::cout << m_first2second_edges_stationary_dist << std::endl;
        //for (auto & cell: m_first2second_cells) {
        //    cell.print();
        //}
        //std::cout << m_first_min_stationary_dist << std::endl;
        inputFile.close();
        return 0;
    }


    void HierachicalCluster::make_search_block(std::vector<SearchingCell> & search_cells,
                std::vector<SearchingBlock> & search_blocks, Int item_size) {
        std::sort(search_cells.begin(), search_cells.end());
        for (size_t i = 0; i < search_cells.size(); i++) {
            if (0 == i) {
                search_blocks.push_back(SearchingBlock());
                search_blocks[search_blocks.size()-1].push_back(search_cells[i], item_size);
            } else {
                if (search_cells[i-1].is_continuous(search_cells[i], item_size)) {
                    search_blocks[search_blocks.size()-1].push_back(search_cells[i], item_size);
                } else {
                    search_blocks.push_back(SearchingBlock());
                    search_blocks[search_blocks.size()-1].push_back(search_cells[i], item_size);
                }
            }
        }
        std::sort(search_blocks.begin(), search_blocks.end());
    }

    Int HierachicalCluster::search(const Eigen::Ref<const Eigen::RowVectorXf> & feature, Int topk,
        std::vector<std::pair<FeatureId, float>> & result) {
        Eigen::setNbThreads(1);
        //std::cout << feature << std::endl;
        TimeStat ts("search ", false);
        Eigen::RowVectorXf query2first_distance = feature * m_first_centers.transpose() * (-2);
        query2first_distance = query2first_distance.array() + feature.squaredNorm();
        query2first_distance += m_first_centers_squa_norm.transpose();
        std::vector<std::vector<std::pair<float, Int>>> topkfirst_center = 
                findTopKNeighbors(query2first_distance, m_conf.m_search_first_center_num);
        m_time_stat[0] += ts.TimeCost();
        std::vector<LimitedMaxHeap<FeatureAssign>>
                heap_vecs(1, LimitedMaxHeap<FeatureAssign>(m_conf.m_search_second_center_num));
        Int ret = findTopkSecondCenters(feature, topkfirst_center, heap_vecs, true);
        m_time_stat[1] += ts.TimeCost();
        if (ret != 0) {
            std::cerr << "HierachicalCluster::search findTopkSecondCenters fail" << std::endl;
        }
        Int item_size;
        if (m_conf.m_hs_mode) {
            item_size = m_conf.m_dim * sizeof(float);
        } else {
            item_size = sizeof(FeatureId) + m_conf.m_dim * sizeof(float);
        }
        std::vector<SearchingCell> search_cells;
        while (!heap_vecs[0].empty()) {
            const FeatureAssign & tmp = heap_vecs[0].top();
            Int file_id = tmp.m_first_center_id % m_conf.m_index_file_num;
            Int cell_id = 
                tmp.m_first_center_id * m_conf.m_second_cluster_num + tmp.m_second_center_id;
            float dist = tmp.m_distance;
            Int offset = m_first2second_cells[cell_id].m_offset;
            Int len = m_first2second_cells[cell_id].m_len;
            double radius = m_first2second_cells[cell_id].m_radius;
            search_cells.push_back(SearchingCell(file_id, cell_id, dist, offset, len, radius));
            heap_vecs[0].pop();
        }
        m_time_stat[2] += ts.TimeCost();
        std::vector<SearchingBlock> search_blocks;
        make_search_block(search_cells, search_blocks, item_size);
        m_time_stat[3] += ts.TimeCost();
        LimitedMaxHeap<Result> result_heap(topk);
        //std::vector<char> tmp_data;
        
        //RMatrixXf block_features;
        std::vector<float> block_features_data;
        std::vector<FeatureId> block_item_ids;
        Int cut = 0;
        Int searched_num = 0;
        Int searched_block_num = 0;
        for (auto & block: search_blocks) {
            TimeStat ts2("block search", false);
            if (result_heap.is_full()) {
                if (searched_num >= m_conf.m_search_neighbors) {
                    break;
                }
                if (searched_block_num >= m_conf.m_search_block_num) {
                    break;
                }
                float cut_dist = result_heap.top().m_distance;
                while (!block.m_search_cells.empty()) {
                    float radius = block.m_search_cells.front().m_radius;
                    float distance = block.m_search_cells.front().m_distance;
                    /*
                    std::cout << "distance=" << std::sqrt(distance) 
                        << " radius=" << std::sqrt(radius) << " cut_dist=" 
                        << std::sqrt(cut_dist) << " searched_num=" 
                        << searched_num << std::endl;
                    */
                    if (std::sqrt(distance) - std::sqrt(radius) > std::sqrt(cut_dist)) {
                        cut++;
                        block.pop_front();
                    } else if (distance / cut_dist > m_conf.m_search_top_cut) {
                        cut++;
                        block.pop_front();
                    } else {
                        break;
                    }
                }
                while (!block.m_search_cells.empty()) {
                    float radius = block.m_search_cells.back().m_radius;
                    float distance = block.m_search_cells.back().m_distance;
                    /*
                    std::cout << "distance=" << std::sqrt(distance) 
                        << " radius=" << std::sqrt(radius) << " cut_dist=" 
                        << std::sqrt(cut_dist) << " searched_num=" 
                        << searched_num << std::endl;
                    */
                    if (std::sqrt(distance) - std::sqrt(radius) > std::sqrt(cut_dist)) {
                        cut++;
                        block.pop_back(item_size);
                    } else if (distance / cut_dist > m_conf.m_search_top_cut) {
                        cut++;
                        block.pop_back(item_size);
                    } else {
                        break;
                    }
                }
            } else {
            }
            m_time_stat[6] += ts2.TimeCost();
            /*
            for (auto & cell: block.m_search_cells) {
                cell.print();
            }
            */
            if (block.m_search_cells.empty()) {
                continue;
            }
            Int read_len = block.m_max_offset - block.m_offset;
            if (read_len <= 0 || read_len % item_size != 0) {
                std::cerr << "read_len err read_len=" << read_len << std::endl;
                continue;
            }
            //std::cout << "read_len " << read_len << std::endl;
            Int item_num = read_len / item_size;
            float * block_features_data_ptr = NULL;
            if (m_conf.m_hs_mode) {
                if (m_conf.m_is_disk) {
                    ret = m_file_read_writer.read(block.m_file_id, block.m_offset, read_len, block_features_data);
                    block_features_data_ptr = block_features_data.data();
                } else {
                    block_features_data_ptr = reinterpret_cast<float *>(
                        m_file_read_writer.get_mem_ptr(block.m_file_id, block.m_offset));
                }
                Int tot_offset = get_tot_offset(block.m_file_id, block.m_offset, item_size);
                block_item_ids.resize(item_num);
                memcpy(block_item_ids.data(),
                    m_feature_ids.data() + tot_offset,
                    item_num * sizeof(FeatureId));
            } else {
                ret = m_file_read_writer.read_matrix(block.m_file_id, block.m_offset, read_len, 
                    item_size, item_num, m_conf.m_dim, block_features_data, block_item_ids, m_time_stat);
                block_features_data_ptr = block_features_data.data();
            }
            Eigen::Map<CMatrixDf> block_features(block_features_data_ptr, m_conf.m_dim, item_num);
            m_time_stat[7] += ts2.TimeCost();

            Eigen::RowVectorXf query2block_features_dist = computeDistanceMatrix(feature, block_features, true);
            m_time_stat[8] += ts2.TimeCost();
            //ts.TimeMark("computeDistanceMatrix end");
            /*
            std::cout << block_features << std::endl;
            for (auto a: block_item_ids) {
                std::cout << a << " ";
            }
            std::cout << std::endl;
            std::cout << "query2block_features_dist=" << query2block_features_dist << std::endl;
            */
            for (Int i = 0; i < item_num; i++) {
                result_heap.push(Result(block_item_ids[i], query2block_features_dist[i]));
            }
            searched_num += item_num;
            searched_block_num++;
            //ts.TimeMark("search block item_num=" + num2str(item_num));
            /*
            printf ("top[%f] block_distance[%f] block_min_distance[%f] cut[%ld] searched_num[%ld] searched_block_num[%ld]\n",
                result_heap.top().m_distance,
                block.m_search_cells.front().m_distance,
                block.m_min_distance,
                cut,
                searched_num, searched_block_num);
            */
            m_time_stat[9] += ts2.TimeCost();
        }
        m_time_stat[4] += ts.TimeCost();
        //std::cout << "CUT\t" << cut << "\tsearched_num\t" << searched_num << std::endl;
        if (result_heap.size() <= 0) {
            std::cerr << "result_heap err result_heap.size() <= 0" << std::endl;
            return 0;
        }
        result.resize(result_heap.size());
        for (Int i = result.size() - 1; i >= 0; i--) {
            const auto & res = result_heap.top();
            result[i] = std::make_pair(res.m_vec_id, res.m_distance);
            result_heap.pop();
        }
        m_time_stat[5] += ts.TimeCost();

        m_time_stat[14] += searched_num;
        m_time_stat[15] += searched_block_num;
        return 0;
    }

    Int HierachicalCluster::search(std::vector<float> & feature_data, Int topk,
        std::vector<std::pair<FeatureId, float>> & result) {
        if (feature_data.size() != (size_t)m_conf.m_dim) {
            std::cerr << "HierachicalCluster::search fail "
                "feature_data.size() != m_conf.m_dim size = "
                << feature_data.size();
            return -1;
        }
        Eigen::Map<Eigen::RowVectorXf> feature(feature_data.data(), m_conf.m_dim);
        return search(feature, topk, result);
    }

    Int HierachicalCluster::search(const Eigen::Ref<const Eigen::RowVectorXf> & feature, Int topk,
                std::vector<FeatureId> & result) {
        std::vector<std::pair<FeatureId, float>> tmp_result;
        Int ret = search(feature, topk, tmp_result);
        if (ret != 0) {
            return -1;
        }
        result.clear();
        for (auto & res: tmp_result) {
            result.push_back(res.first);
        }
        return 0;
    }

    Int HierachicalCluster::search(std::vector<float> & feature_data, Int topk,
                std::vector<FeatureId> & result) {
                std::vector<std::pair<FeatureId, float>> tmp_result;
        Int ret = search(feature_data, topk, tmp_result);
        if (ret != 0) {
            return -1;
        }
        result.clear();
        for (auto & res: tmp_result) {
            result.push_back(res.first);
        }
        return 0;
    }
}