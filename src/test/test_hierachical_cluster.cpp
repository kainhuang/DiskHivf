#include "hierachical_cluster.h"
#include "file_read_write.h"
#include "conf.h"

using namespace disk_hivf;

int main(int argc, char* argv[]) {
    Eigen::initParallel();
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <conf_file> <query_file> <groundtruth_file>" << std::endl;
        return 1;
    }
    
    //test_kmeans_core();
    std::string conf_file = argv[1];
    std::string query_file = argv[2];
    std::string groundtruth_file = argv[3];
    Conf conf;
    conf.Init(conf_file.c_str());
    
    /*
    HierachicalCluster hc(conf);
    hc.init();
    hc.train_model();
    hc.save_model();
    */
    HierachicalCluster hc2(conf);
    hc2.init();
    hc2.load_model();
    hc2.build_index();
    hc2.save_index();
    
    HierachicalCluster hc3(conf);
    hc3.init();
    hc3.load_model();
    hc3.load_index();

    std::vector<float> query_data;
    Eigen::Map<RMatrixDf> querys = readMatrixFromDimVecs(query_file, query_data);
    //std::cout << querys << std::endl << std::endl;

    for (int i = 0; i < 100; i++) {
        
        std::vector<std::pair<Int, float>> result;
        hc3.search(querys.row(i), 1, result);
        
        for (auto & a: result) {
            std::cout << a.first << " ";
        }
        std::cout << std::endl;
        
    }

    std::vector<int> groundtruth_data;
    int dim;
    Int numVecs;
    readDimVecs(groundtruth_file, groundtruth_data, dim, numVecs);
    Eigen::Map<RMatrixDi> groundtruth(groundtruth_data.data(), numVecs, dim);

    //std::cout << groundtruth << std::endl;
    return 0;
}