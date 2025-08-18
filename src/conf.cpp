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
        m_kmeans_sample_rete = 1;
        m_batch_size = 256;
        m_kmeans_centers_select_type = 3;
        m_first_cluster_num = 300;
        m_second_cluster_num = 300;
        m_hierachical_cluster_epoch = 40;
        m_read_file_batch_size = 10000;
        m_build_index_search_first_center_num = 60;
        m_search_first_center_num = 20;
        m_search_second_center_num = 20;
        m_is_disk = 0;
        m_search_neighbors = 1000;
        m_search_block_num = 20;
        m_search_top_cut = 1.5;
        m_hs_mode = 0;
        m_thread_num = 1;
        m_read_index_file_thread_num = 5;
        m_is_async_read = 1;
        m_build_index_num = -1;
        m_train_data_num = -1;
        m_use_uint8_data = 0;
        m_io_thread_num = 0;
        m_debug_log = 0;
        m_dynamic_prune_switch = 0;
        m_dynamic_prune_a = 0;
        m_dynamic_prune_b = 0;
        m_dynamic_prune_c = 1000000;
        m_use_cache = 0;
        m_cache_capacity = 0;
        m_cache_segment = 0;
        m_build_search_topk = 1;
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
            m_kmeans_sample_rete = str2num<float>(pool["kmeans_sample_rete"]);
            m_batch_size = str2num<Int>(pool["batch_size"]);
            m_kmeans_centers_select_type = str2num<Int>(pool["kmeans_centers_select_type"]);
            m_first_cluster_num = str2num<Int>(pool["first_cluster_num"]);
            m_second_cluster_num = str2num<Int>(pool["second_cluster_num"]);
            m_hierachical_cluster_epoch = str2num<Int>(pool["hierachical_cluster_epoch"]);
            m_read_file_batch_size = str2num<Int>(pool["read_file_batch_size"]);
            m_build_index_search_first_center_num = str2num<Int>(pool["build_index_search_first_center_num"]);
            m_search_first_center_num = str2num<Int>(pool["search_first_center_num"]);
            m_search_second_center_num = str2num<Int>(pool["search_second_center_num"]);
            m_is_disk = str2num<Int>(pool["is_disk"]);
            m_search_neighbors = str2num<Int>(pool["search_neighbors"]);
            m_search_block_num = str2num<Int>(pool["search_block_num"]);
            m_search_top_cut = str2num<float>(pool["search_top_cut"]);
            m_hs_mode = str2num<float>(pool["hs_mode"]);
            m_thread_num = str2num<float>(pool["thread_num"]);
            m_read_index_file_thread_num = str2num<float>(pool["read_index_file_thread_num"]);
            m_is_async_read = str2num<float>(pool["is_async_read"]);
            m_build_index_num = str2num<Int>(pool["build_index_num"]);
            m_train_data_num = str2num<Int>(pool["train_data_num"]);
            m_use_uint8_data = str2num<Int>(pool["use_uint8_data"]);
            m_io_thread_num = str2num<Int>(pool["io_thread_num"]);
            if (pool.find("debug_log") != pool.end()) {
                m_debug_log = str2num<Int>(pool["debug_log"]);
            }
            if (pool.find("dynamic_prune_switch") != pool.end()) {
                m_dynamic_prune_switch = str2num<Int>(pool["dynamic_prune_switch"]);
            }
            if (pool.find("dynamic_prune_a") != pool.end()) {
                m_dynamic_prune_a = str2num<float>(pool["dynamic_prune_a"]);
            }
            if (pool.find("dynamic_prune_b") != pool.end()) {
                m_dynamic_prune_b = str2num<float>(pool["dynamic_prune_b"]);
            }
            if (pool.find("dynamic_prune_c") != pool.end()) {
                m_dynamic_prune_c = str2num<float>(pool["dynamic_prune_c"]);
            }
            if (pool.find("use_cache") != pool.end()) {
                m_use_cache = str2num<Int>(pool["use_cache"]);
            }
            if (pool.find("cache_capacity") != pool.end()) {
                m_cache_capacity = str2num<Int>(pool["cache_capacity"]);
            }
            if (pool.find("cache_segment") != pool.end()) {
                m_cache_segment = str2num<Int>(pool["cache_segment"]);
            }
            if (pool.find("build_search_topk") != pool.end()) {
                m_build_search_topk = str2num<Int>(pool["build_search_topk"]);
            }
        } catch (...) {
            fprintf(stderr, "Init conf fail!!!!!!!!");
            return -1;
        }
        std::string conf_info = ToString();
        fprintf(stderr, "Conf:\n%s", conf_info.c_str());
        return 0;
    }
}
