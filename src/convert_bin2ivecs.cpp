#include "file_read_write.h"
using namespace disk_hivf;

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_filename> <output_filename> <batch_size>" 
        << std::endl;
        return 1;
    }
    std::string inputFilename = argv[1];
    std::string outputFilename = argv[2];
    Int batchSize = std::stoi(argv[3]);
    std::ifstream file(inputFilename, std::ios::binary);
    if (!file) {
        std::cerr << "open read file fail filename=" << inputFilename << std::endl;
        return -1;
    }
    std::ofstream outputFile(outputFilename, std::ios::binary);
    if (!outputFile) {
        std::cerr << "open write file fail filename=" << outputFilename << std::endl;
        return -1;
    }
    int num;
    file.read(reinterpret_cast<char*>(&num), sizeof(int));
    int dim;
    file.read(reinterpret_cast<char*>(&dim), sizeof(int));
    if (!file) {
        std::cerr << "read file fail filename=" << inputFilename << std::endl;
        return -1;
    }
    std::cout << "dim=" <<dim << " num=" << num << std::endl;
    std::vector<char> buff(batchSize * dim * sizeof(int));
    for (Int i = 0; i < num; i += batchSize) {
        Int currentBatchSize = std::min(batchSize, num - i);
        size_t read_size = currentBatchSize * (dim * sizeof(int));
        file.read(buff.data(), read_size);
        if (!file) {
            std::cerr << "batch read file fail filename=" << inputFilename << std::endl;
            return -1;
        }
        for (Int j = 0; j < currentBatchSize; j++) {
            outputFile.write(reinterpret_cast<char *>(&dim), sizeof(dim));
            
            int * ptr = reinterpret_cast<int *>(buff.data() + j * (dim * sizeof(int))); 
            for (Int j = 0; j < dim; j++) {
                std::cout << ptr[j] << " ";
            }
            std::cout << std::endl;
            
            outputFile.write(buff.data() + j * (dim * sizeof(int)), dim * sizeof(int));
            if (!outputFile) {
                std::cerr << "batch write file fail filename=" << outputFilename << std::endl;
                return -1;
            }
        }
    }
    file.close();
    outputFile.close();
    return 0;
}