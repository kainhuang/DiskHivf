#define WARMUP
#include "common.h"
#include "hierachical_cluster.h"
#include "file_read_write.h"
#include "conf.h"
#include "unity.h"

using namespace disk_hivf;

bool isInVector(const std::vector<std::pair<FeatureId, float>>& vec, Int value, float dist = 0, int use_dist = 0) {
    if (use_dist) {
        for (const auto & item: vec) {
            if (item.first == value) {
                return true;
            }
            if (std::fabs(item.second - dist) < 1e-6) {
                return true;
            }
        }
        // if (std::fabs(vec[col].second - dist) < 1e-8) {
        //    return true;
        // }
    } else {
        for (const auto & item: vec) {
            if (item.first == value) {
                return true;
            }
        }
    }
    return false;
    //return std::find(vec.begin(), vec.end(), value) != vec.end();
}

void run_test_set(HierachicalCluster & hc, Eigen::Map<RMatrixDf> & querys,
    Eigen::Map<RMatrixDi> & groundtruth, Eigen::Map<RMatrixDf> & dist,
    Int recall_topk, Int at_num, Int thread_num, Int use_cache, int use_dist = 0) {
    TimeStat ts("run_test_set ");
    std::vector<std::vector<std::pair<FeatureId, float>>> results(querys.rows(), std::vector<std::pair<FeatureId, float>>());
    
    for (Int _ = 0; _ < 1; _++) {
        long long st = ts.TimeMark("search begin");
    #pragma omp parallel for num_threads(thread_num) schedule(dynamic)
        for (Int i = 0; i < querys.rows(); i++) {
            //TimeStat ts("searching " + num2str<Int>(i));
            std::vector<std::pair<FeatureId, float>> result;;
            hc.search(querys.row(i), at_num, result, use_cache);
            results[i] = std::move(result);
        }
        long long tim = ts.TimeMark("search end");
        std::cout << "avg_time " << ((tim - st) * 1.0 / querys.rows()) << " us" << std::endl;
    }
    for (Int i = 0; i < 16; i++) {
        std::cout << i << " " << hc.m_time_stat[i] * 1.0 / querys.rows() << std::endl;
    }
    /*
    for (auto & result: results) {
        for (auto & a: result) {
            std::cout << a << " ";
        }
        std::cout << std::endl;
    }
    */

    Int tot = 0;
    Int recall = 0;

    for (Int row = 0; row < groundtruth.rows(); row++) {
        for (Int col = 0; col < groundtruth.cols() && col < recall_topk; col++) {
            tot++;
            if (use_dist) {
                if (isInVector(results[row], groundtruth(row, col), dist(row, col), use_dist)) {
                    recall++;
                }
            } else {
                if (isInVector(results[row], groundtruth(row, col), 0, use_dist)) {
                    recall++;
                }
            }

        }
    }
    float recall_rate = recall * 1.0 / tot * 100;
    printf("%ld-recall@%ld = %f\n", recall_topk, at_num, recall_rate);
}


int main(int argc, char* argv[]) {
    //Eigen::initParallel();
    if (argc <= 8) {
        std::cerr << "Usage: " << argv[0] 
        << " <conf_file> <query_file> <groundtruth_file> <topk> <at_num> <thread_num>"
        " <first_centers_num> <second_centers_num> <debug_log> <use_cache> <is_query_uint8> "
        " <use_dist> <search_neighbors> <search_blocks>" << std::endl;
        return 1;
    }
    std::string conf_file = argv[1];
    std::string query_file = argv[2];
    std::string groundtruth_file = argv[3];
    Int topk = str2num<Int>(argv[4]);
    Int at_num = str2num<Int>(argv[5]);
    Int thread_num = str2num<Int>(argv[6]);
    Int first_centers_num = str2num<Int>(argv[7]);
    Conf conf;
    conf.Init(conf_file.c_str());
    if (first_centers_num > 0) {
        conf.m_search_first_center_num = first_centers_num;       
        if (argc > 8) {
            conf.m_search_second_center_num = str2num<Int>(argv[8]);
        } else {
            conf.m_search_second_center_num = first_centers_num * first_centers_num;
        }
    }
    if (argc > 9) {
        conf.m_debug_log = str2num<Int>(argv[9]);
    }
    if (argc > 10) {
        conf.m_use_cache = str2num<Int>(argv[10]);
    }
    int is_query_uint8 = 0;
    if (argc > 11) {
        is_query_uint8 = str2num<int>(argv[11]);
    }
    int use_dist = 0;
    if (argc > 12) {
        use_dist = str2num<int>(argv[12]);
    }
    if (argc > 13) {
        conf.m_search_neighbors = str2num<Int>(argv[13]);
    }
    if (argc > 14) {
        conf.m_search_block_num = str2num<Int>(argv[14]);
    }
    Int ret = 0;
    HierachicalCluster hc(conf);
    ret = hc.init();
    if (ret != 0) {
        std::cerr << "hc.init() fail" << std::endl;
        return -1;
    }
    ret = hc.load_model();
    if (ret != 0) {
        std::cerr << "hc.load_model() fail" << std::endl;
        return -1;
    }
    ret = hc.load_index();
    if (ret != 0) {
        std::cerr << "hc.load_index() fail" << std::endl;
        return -1;
    }

    std::vector<float> query_data;
    Eigen::Map<RMatrixDf> querys = readMatrixFromDimVecs(query_file, query_data, is_query_uint8);
    // std::cout << querys << std::endl;
    std::cout << "querys num = " << querys.rows() << " dim=" << querys.cols() << std::endl;
    std::vector<int> groundtruth_data;
    std::vector<float> gt_dist;
    int dim;
    Int numVecs;
    if (use_dist) {
        readGTData(groundtruth_file, groundtruth_data, gt_dist, dim, numVecs);
    } else {
        readDimVecs(groundtruth_file, groundtruth_data, dim, numVecs);
    }
    Eigen::Map<RMatrixDi> groundtruth(groundtruth_data.data(), numVecs, dim);
    //std::cout << groundtruth << std::endl;
    std::cout << "groundtruth numVecs=" << numVecs << " dim=" << dim << std::endl;

    Eigen::Map<RMatrixDf> dist(gt_dist.data(), numVecs, dim);

    // warmup
    #ifdef WARMUP
    #pragma omp parallel for num_threads(32) schedule(dynamic)
    for (Int i = 0; i < querys.rows(); i++) {
        std::vector<std::pair<FeatureId, float>> result;
        hc.search(querys.row(i), at_num, result, 0);
    }
    #endif
    

    hc.m_time_stat.resize(16, 0);

    run_test_set(hc, querys, groundtruth, dist, topk, at_num, thread_num, conf.m_use_cache, use_dist);
    return 0;
}