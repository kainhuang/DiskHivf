#include <iostream>
#include "Eigen/Dense"
#include <chrono>
#include "file_read_write.h"
#include "matrix.h"
#include "unity.h"
using namespace disk_hivf;

int main(int argc, char* argv[]) {
    // 使用 Eigen 库
    using namespace Eigen;
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0] << " <index_file> <query_file> <gt_file>"
            " <ivecs|dim.ivecs|bin> <topk> <batch_size> <is_uint8>" 
            << std::endl;
        return 1;
    }

    std::string index_file = argv[1];
    std::string query_file = argv[2];
    std::string gt_file = argv[3];
    std::string type = argv[4];
    int topk = atoi(argv[5]);
    int batch_size = atoi(argv[6]);
    int is_uint8 = atoi(argv[7]);

    std::vector<float> query_data;
    Eigen::Map<RMatrixXf> query = readMatrixFromDimVecs(query_file, query_data, is_uint8);
    std::cout << "RMatrixXf query rows and cols " << query.rows() << " " << query.cols() << std::endl;
    Eigen::Map<RMatrixXf> batch_query(query_data.data(), batch_size, query.cols());
    std::cout << "batch_query " << batch_query << std::endl;
    std::cout << std::endl;

    std::vector<float> index_data;
    Eigen::Map<RMatrixXf> index = readMatrixFromDimVecs(index_file, index_data, is_uint8);
    std::cout << "RMatrixXf index rows and cols " << index.rows() << " " << index.cols() << std::endl;
    Eigen::Map<RMatrixXf> print_batch_index(index_data.data(), batch_size, index.cols());
    std::cout << "print_batch_index " << print_batch_index << std::endl;
    std::cout << std::endl;

    std::ofstream outputFile(gt_file, std::ios::binary);
    if (!outputFile) {
        std::cerr << "open write file fail filename=" << gt_file << std::endl;
        return -1;
    }

    Int batchSize = batch_size;
    int num = query.rows();
    int dim = query.cols();

    if (type == "dim.ivecs") {
        Int vec_num = num;
        outputFile.write(reinterpret_cast<const char*>(&topk), sizeof(int));
        outputFile.write(reinterpret_cast<const char*>(&vec_num), sizeof(Int));
        if (!outputFile) {
            std::cerr << "write file fail filename=" << gt_file << std::endl;
            return -1;
        }
    } else if (type == "bin") {
        outputFile.write(reinterpret_cast<const char*>(&num), sizeof(int));
        outputFile.write(reinterpret_cast<const char*>(&topk), sizeof(int));
        if (!outputFile) {
            std::cerr << "write file fail filename=" << gt_file << std::endl;
            return -1;
        }
    } else {
    }
    std::vector<int> buff((size_t)num * topk);
    #pragma omp parallel for num_threads(8)
    for (Int i = 0; i < num; i += batchSize) {
        TimeStat ts("cal batch knn");
        Int currentBatchSize = std::min(batchSize, num - i);
        Eigen::Map<RMatrixXf> batch_query(query_data.data() + i * dim, currentBatchSize, dim);
        std::vector<std::vector<std::pair<float, Int>>> results = 
            findTopKNeighbors(batch_query, index, topk);

        for (size_t item_id = 0; item_id < results.size(); item_id++) {
            auto & vec = results[item_id];
            Int idx = i + item_id;
            for (size_t dim_id = 0; dim_id < vec.size(); dim_id++) {
                Int id = idx * topk + dim_id;
                buff[id] = vec[dim_id].second;
                std::cout << vec[dim_id].second << " " << vec[dim_id].first << " ";
            }
            std::cout << std::endl;
        }
        //for (int a: buff) {
        //    std::cout << a << " ";
        //}
        //std::cout << "running num = " << i << std::endl;
    }
    std::cout << " write buff.size()=" << buff.size() <<std::endl;
    if (type != "ivecs") {
        outputFile.write(reinterpret_cast<const char*>(buff.data()), buff.size() * sizeof(int));
        if (!outputFile) {
            std::cerr << "batch write file fail filename=" << gt_file << std::endl;
            return -1;
        }
    } else {
        for (Int i = 0; i < num; i++) {
            outputFile.write(reinterpret_cast<const char*>(&topk), sizeof(int));
            outputFile.write(reinterpret_cast<const char*>(buff.data() + i * topk), topk * sizeof(int));
            if (!outputFile) {
                std::cerr << "batch write file fail filename=" << gt_file << std::endl;
                return -1;
            }
        }
    }
    outputFile.close();
    return 0;
}