#include "file_read_write.h"
using namespace disk_hivf;

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <input_filename> <input_query> <output_filename> <output_query>" 
        << std::endl;
        return 1;
    }
    std::string inputFilename = argv[1];
    std::string input_query_file = argv[2];
    std::string outputFilename = argv[3];
    std::string output_query_file = argv[4];

    std::ofstream outputFile(outputFilename, std::ios::binary);
    if (!outputFile) {
        std::cerr << "open write file fail filename=" << outputFilename << std::endl;
        return -1;
    }

    std::ofstream output_query(output_query_file, std::ios::binary);
    if (!output_query) {
        std::cerr << "open write file fail output_query_file=" << output_query_file << std::endl;
        return -1;
    }

    std::vector<float> query_data;
    Eigen::Map<RMatrixXf> query = readMatrixFromDimVecs(input_query_file, query_data);
    std::cout << "query numVecs=" << query.rows() << " dim=" << query.cols() << std::endl;
    int query_dim = query.cols();
    std::vector<int> groundtruth_data;
    std::vector<float> gt_dist;
    int dim;
    Int numVecs;
    readGTData(inputFilename, groundtruth_data, gt_dist, dim, numVecs);
    Eigen::Map<RMatrixDi> groundtruth(groundtruth_data.data(), numVecs, dim);
    //std::cout << groundtruth << std::endl;
    std::cout << "groundtruth numVecs=" << groundtruth.rows() << " dim=" << groundtruth.cols() << std::endl;
    Eigen::Map<RMatrixDf> dist(gt_dist.data(), numVecs, dim);
    std::cout << "dist numVecs=" << dist.rows() << " dim=" << dist.cols() << std::endl;
    int out_num = 0;
    std::vector<int> buff;
    buff.resize(numVecs * dim);
    Eigen::Map<RMatrixDi> out_gt(buff.data(), numVecs, dim);

    std::vector<float> dist_buff;
    dist_buff.resize(numVecs * dim);
    Eigen::Map<RMatrixDf> out_dist(dist_buff.data(), numVecs, dim);

    std::vector<float> query_buff;
    query_buff.resize(numVecs * query_dim);
    Eigen::Map<RMatrixDf> out_query(query_buff.data(), numVecs, query_dim);


    for (Int i = 0; i < numVecs; i++) {
        if (dist(i, 0) > 1e-6) {
            out_gt.row(out_num) = groundtruth.row(i);
            out_dist.row(out_num) = dist.row(i);
            out_query.row(out_num) = query.row(i);
            out_num++;
        }
    }
    std::cout << "out_num = " << out_num << std::endl;
    output_query.write(reinterpret_cast<char*>(&out_num), sizeof(int));
    output_query.write(reinterpret_cast<char*>(&query_dim), sizeof(int));
    output_query.write(reinterpret_cast<char *>(query_buff.data()), out_num * query_dim * sizeof(float));
    if (!output_query) {
        std::cerr << "write file fail output_query_file=" << output_query_file << std::endl;
        return -1;
    }
    std::cout << "out_num = " << out_num << std::endl;
    outputFile.write(reinterpret_cast<char*>(&out_num), sizeof(int));
    outputFile.write(reinterpret_cast<char*>(&dim), sizeof(int));
    outputFile.write(reinterpret_cast<char *>(buff.data()), out_num * dim * sizeof(int));
    outputFile.write(reinterpret_cast<char *>(dist_buff.data()), out_num * dim * sizeof(float));
    if (!outputFile) {
        std::cerr << "write file fail outputFile=" << outputFilename << std::endl;
        return -1;
    }
    std::cout << "out_num = " << out_num << std::endl;
    outputFile.close();
    return 0;
}