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
    Int recall_topk, Int at_num) {
    TimeStat ts("run_test_set ");
    std::vector<std::vector<FeatureId>> results(querys.rows(), std::vector<FeatureId>());
    ts.TimeMark("search begin");
    //#pragma omp parallel for
    for (Int i = 0; i < querys.rows(); i++) {
        //TimeStat ts("searching " + num2str<Int>(i));
        std::vector<FeatureId> result;
        hc.search(querys.row(i), at_num, result);
        results[i] = result;
    }
    for (Int i = 0; i < 14; i++) {
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
    long long tim = ts.TimeMark("search end");
    std::cout << "avg_time " << (tim * 1.0 / querys.rows()) << " us" << std::endl;
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


void run_test_set2(HierachicalCluster &hc, Eigen::Map<RMatrixDf> & querys,
    Eigen::Map<RMatrixDi> & groundtruth,
    Int recall_topk, Int at_num) {
    TimeStat ts("run_test_set ");
    std::vector<std::vector<std::pair<FeatureId, float>>> results(querys.rows(), std::vector<std::pair<FeatureId, float>>());
    ts.TimeMark("search begin");
    #pragma omp parallel for
    for (Int i = 0; i < querys.rows(); i++) {
        //TimeStat ts("searching " + num2str<Int>(i));
        std::vector<std::pair<FeatureId, float>> result;
        hc.search(querys.row(i), at_num, result);
        results[i] = result;
    }
    for (Int i = 0; i < 14; i++) {
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
    long long tim = ts.TimeMark("search end");
    std::cout << "avg_time " << (tim * 1.0 / querys.rows()) << " us" << std::endl;
    Int tot = 0;
    Int recall = 0;

    for (Int row = 0; row < groundtruth.rows(); row++) {
        for (Int col = 0; col < groundtruth.cols() && col < recall_topk; col++) {
            tot++;
            Int rank = isInVector2(results[row], groundtruth(row, col));
            if (rank >= 0) {
                recall++;
            }
            std::cout << "rank=" << rank << std::endl;
            for (auto res: results[row]) {
                std::cout << res.first << " " << res.second << std::endl;
            }
            std::cout << groundtruth(row, col) << std::endl;
        
        }
    }
    float recall_rate = recall * 1.0 / tot;
    printf("%ld-recall@%ld = %f\n", recall_topk, at_num, recall_rate);
}


int main(int argc, char* argv[]) {
    //Eigen::initParallel();
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] 
        << " <conf_file> <query_file> <groundtruth_file> <topk> <at_num>" << std::endl;
        return 1;
    }
    std::string conf_file = argv[1];
    std::string query_file = argv[2];
    std::string groundtruth_file = argv[3];
    Int topk = str2num<Int>(argv[4]);
    Int at_num = str2num<Int>(argv[5]);
    Conf conf;
    conf.Init(conf_file.c_str());
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

    std::vector<int> groundtruth_data;
    int dim;
    Int numVecs;
    readDimVecs(groundtruth_file, groundtruth_data, dim, numVecs);
    Eigen::Map<RMatrixDi> groundtruth(groundtruth_data.data(), numVecs, dim);

    run_test_set(hc, querys, groundtruth, topk, at_num);
    return 0;
}