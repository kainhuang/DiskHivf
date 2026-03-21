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
            float m_kmeans_sample_rete;
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
            Int m_thread_num;
            Int m_read_index_file_thread_num;
            Int m_is_async_read;
            Int m_build_index_num;
            Int m_train_data_num;
            Int m_use_uint8_data;
            Int m_io_thread_num;
            Int m_debug_log;
            Int m_dynamic_prune_switch;
            float m_dynamic_prune_a;
            float m_dynamic_prune_b;
            float m_dynamic_prune_c;
            Int m_use_cache;
            Int m_cache_capacity;
            Int m_cache_segment;
            Int m_build_search_topk;
            Int m_build_search_first_center_id_diff;
            Int m_use_pread;               // 是否使用pread替代fstream读取，默认1
            Int m_use_direct_io;           // 是否使用O_DIRECT模式，默认0
            Int m_prefetch_bytes_limit;    // 预读窗口字节数上限，默认512KB
            Int m_block_split_threshold;   // 大block拆分阈值，默认256KB
            Int m_min_sub_task_size;       // 拆分后子任务最小大小，默认64KB
    };
}
