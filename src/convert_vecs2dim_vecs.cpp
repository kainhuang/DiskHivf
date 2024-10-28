#include "file_read_write.h"
using namespace disk_hivf;

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_filename> <output_filename> <batch_size>" << std::endl;
        return 1;
    }

    std::string inputFilename = argv[1];
    std::string outputFilename = argv[2];
    Int batchSize = std::stoi(argv[3]);

    FileType fileType = getFileType(inputFilename);
    Int ret = 0;

    int dimension = 0;
    Int numVectors = 0;
    if (fileType == FileType::BVEC) {
        ret = readVectorFileMetadata<char>(inputFilename, dimension, numVectors);
        std::cout << "Read " << numVectors << " vectors of dimension " << dimension << " from " << inputFilename << std::endl;
        ret = convertVecs2DimVecs<char>(inputFilename, outputFilename, dimension, numVectors, batchSize);
    } else if (fileType == FileType::FVEC) {
        ret = readVectorFileMetadata<float>(inputFilename, dimension, numVectors);
        std::cout << "Read " << numVectors << " vectors of dimension " << dimension << " from " << inputFilename << std::endl;
        ret = convertVecs2DimVecs<float>(inputFilename, outputFilename, dimension, numVectors, batchSize);
    } else if (fileType == FileType::IVEC) {
        ret = readVectorFileMetadata<int>(inputFilename, dimension, numVectors);
        std::cout << "Read " << numVectors << " vectors of dimension " << dimension << " from " << inputFilename << std::endl;
        ret = convertVecs2DimVecs<int>(inputFilename, outputFilename, dimension, numVectors, batchSize);
    } else {
        std::cerr << "Unknown file type: " << inputFilename << std::endl;
        return 1;
    }
    if (ret != 0) {
        std::cerr << "convert err" << std::endl;
        return 1;
    }

    return 0;
}