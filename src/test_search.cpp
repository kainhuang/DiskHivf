#include "common.h"
#include "hierachical_cluster.h"
#include "file_read_write.h"
#include "conf.h"
#include "unity.h"

using namespace disk_hivf;

bool isInVector(const std::vector<FeatureId>& vec, Int value) {
    return std::find(vec.begin(), vec.end(), value) != vec.end();
}

void run_test_set(HierachicalCluster &hc, Eigen::Map<RMatrixDf> & querys,
    Eigen::Map<RMatrixDi> & groundtruth,
    Int recall_topk, Int at_num, Int thread_num) {
    TimeStat ts("run_test_set ");
    std::vector<std::vector<FeatureId>> results(querys.rows(), std::vector<FeatureId>());
    
    for (Int _ = 0; _ < 1; _++) {
        long long st = ts.TimeMark("search begin");
    #pragma omp parallel for num_threads(thread_num)
        for (Int i = 0; i < querys.rows(); i++) {
            //TimeStat ts("searching " + num2str<Int>(i));
            std::vector<FeatureId> result;
            hc.search(querys.row(i), at_num, result);
            results[i] = result;
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
            if (isInVector(results[row], groundtruth(row, col))) {
                recall++;
            }
        }
    }
    float recall_rate = recall * 1.0 / tot;
    printf("%ld-recall@%ld = %f\n", recall_topk, at_num, recall_rate);
}

Int isInVector2(const std::vector<std::pair<FeatureId, float>>& vec, Int value) {
    for (size_t i = 0; i < vec.size(); i++) {
        auto & pair = vec[i];
        if (value == pair.first) {
            return i;
        }
    }
    return -1;
}

int main(int argc, char* argv[]) {
    //Eigen::initParallel();
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] 
        << " <conf_file> <query_file> <at_num> <first_centers_num> <second_centers_num>" << std::endl;
        return 1;
    }
    std::string conf_file = argv[1];
    std::string query_file = argv[2];
    Int at_num = str2num<Int>(argv[3]);
    Int first_centers_num = str2num<Int>(argv[4]);
    Conf conf;
    conf.Init(conf_file.c_str());
    conf.m_search_first_center_num = first_centers_num; 
    conf.m_search_second_center_num = str2num<Int>(argv[5]);
    conf.m_debug_log = 1;
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
    Eigen::Map<RMatrixDf> querys = readMatrixFromDimVecs(query_file, query_data);
    for (Int i = 0; i < querys.rows(); i++) {
        //TimeStat ts("searching " + num2str<Int>(i));
        std::vector<FeatureId> result;
        hc.search(querys.row(i), at_num, result);
    }
    return 0;
}