#include "conf.h"
#include "unity.h"
#include "common.h"

namespace disk_hivf {
    Conf::Conf() {
        m_train_data_file = "../data/sift/sift_learn.dim.fvecs";
        m_index_data_file = "../data/sift/sift_base.dim.fvecs";
        m_model_file = "./model";
        m_index_dir = "./index";
        m_index_file_num = 1;
        m_dim = 128;
        m_kmeans_epoch = 40;
        m_batch_size = 256;
        m_kmeans_centers_select_type = 3;
        m_first_cluster_num = 300;
        m_second_cluster_num = 300;
        m_hierachical_cluster_epoch = 40;
        m_read_file_batch_size = 10000;
        m_search_first_center_num = 20;
        m_search_second_center_num = 20;
        m_is_disk = 0;
        m_search_neighbors = 1000;
        m_search_block_num = 20;
        m_search_top_cut = 1.5;
    }
    int Conf::Init(const char * configFile) {
        Int ret = makePool(configFile);
        if (ret != 0) {
            //LOG
            return -1;
        } 
        try {
            m_train_data_file = pool["train_data_file"];
            m_index_data_file = pool["index_data_file"];
            m_model_file = pool["model_file"];
            m_index_dir = pool["index_dir"];
            m_index_file_num = str2num<Int>(pool["index_file_num"]);
            m_dim = str2num<Int>(pool["dim"]);
            m_kmeans_epoch = str2num<Int>(pool["kmeans_epoch"]);
            m_batch_size = str2num<Int>(pool["batch_size"]);
            m_kmeans_centers_select_type = str2num<Int>(pool["kmeans_centers_select_type"]);
            m_first_cluster_num = str2num<Int>(pool["first_cluster_num"]);
            m_second_cluster_num = str2num<Int>(pool["second_cluster_num"]);
            m_hierachical_cluster_epoch = str2num<Int>(pool["hierachical_cluster_epoch"]);
            m_read_file_batch_size = str2num<Int>(pool["read_file_batch_size"]);
            m_search_first_center_num = str2num<Int>(pool["search_first_center_num"]);
            m_search_second_center_num = str2num<Int>(pool["search_second_center_num"]);
            m_is_disk = str2num<Int>(pool["is_disk"]);
            m_search_neighbors = str2num<Int>(pool["search_neighbors"]);
            m_search_block_num = str2num<Int>(pool["search_block_num"]);
            m_search_top_cut = str2num<float>(pool["search_top_cut"]);
        } catch (...) {
            fprintf(stderr, "Init conf fail!!!!!!!!");
            return -1;
        }
        std::string conf_info = ToString();
        fprintf(stderr, "Conf:\n%s", conf_info.c_str());
        return 0;
    }
}