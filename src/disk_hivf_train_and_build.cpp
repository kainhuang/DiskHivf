#include "hierachical_cluster.h"
#include "conf.h"

using namespace disk_hivf;

int main(int argc, char* argv[]) {
    Eigen::initParallel();
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <conf_file>" << std::endl;
        return 1;
    }
    std::string conf_file = argv[1];
    Conf conf;
    conf.Init(conf_file.c_str());
    
    Int ret;
    HierachicalCluster hc(conf);
    ret = hc.init();
    if (ret != 0) {
        std::cerr << "hc.init() fail" << std::endl;
        return -1;
    }
    ret = hc.train_model();
    if (ret != 0) {
        std::cerr << "hc.train_model() fail" << std::endl;
        return -1;
    }
    ret = hc.save_model();
    if (ret != 0) {
        std::cerr << "hc.save_model() fail" << std::endl;
        return -1;
    }
    ret = hc.build_index();
    if (ret != 0) {
        std::cerr << "hc.build_index() fail" << std::endl;
        return -1;
    }
    ret = hc.save_index();
    if (ret != 0) {
        std::cerr << "hc.save_index() fail" << std::endl;
        return -1;
    }
    return 0;
}