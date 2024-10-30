#pragma once
#include "unity.h"
#include <string>
#include "common.h"

namespace disk_hivf {
    class Conf: public Config {
        public:
            Conf();
            int Init(const char * configFile);
        public:
            std::string m_train_data_file; //训练数据
            std::string m_index_data_file; //需要索引的数据
            std::string m_model_file; //索引模型文件
            std::string m_index_dir; //对数据建索引后的输出目录
            Int m_index_file_num;
            Int m_dim;
            Int m_kmeans_epoch;
            Int m_batch_size;
            Int m_kmeans_centers_select_type;
            Int m_first_cluster_num;
            Int m_second_cluster_num;
            Int m_hierachical_cluster_epoch;
            Int m_read_file_batch_size;
            Int m_build_index_search_first_center_num;
            Int m_search_first_center_num;
            Int m_search_second_center_num;
            Int m_is_disk;
            Int m_search_neighbors;
            Int m_search_block_num;
            float m_search_top_cut;
            Int m_hs_mode;
    };
}
