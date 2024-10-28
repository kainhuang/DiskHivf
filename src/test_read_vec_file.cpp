#include "file_read_write.h"
using namespace disk_hivf;

int main(int argc, char* argv[]) {

    FileReadWriter file = FileReadWriter("index/", 1, 1);
    file.Init();
    std::vector<char> data;
    file.read(0, data);
    Int id = *(reinterpret_cast<Int *>(data.data()));
    std::cout << "id = " << id << std::endl;
    /*
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_filename> <output_filename> <batch_size>" << std::endl;
        return 1;
    }

    std::string inputFilename = argv[1];
    std::string outputFilename = argv[2];
    Int batchSize = std::stoi(argv[3]);

    FileType fileType = getFileType(inputFilename);

    try {
        int dimension = 0;
        Int numVectors = 0;
        if (fileType == FileType::BVEC) {
            readVectorFileMetadata<char>(inputFilename, dimension, numVectors);
            std::cout << "Read " << numVectors << " vectors of dimension " << dimension << " from " << inputFilename << std::endl;
            convertVecs2DimVecs<char>(inputFilename, outputFilename, dimension, numVectors, batchSize);
            std::vector<char> data;
            //readVectors<char>(inputFilename, data, dimension, numVectors);
            //printVectors<char>(data, dimension);
            //data.clear();
            readDimVecs<char>(outputFilename, data, dimension, numVectors);
            printVectors<char>(data, dimension);
        } else if (fileType == FileType::FVEC) {
            readVectorFileMetadata<float>(inputFilename, dimension, numVectors);
            std::cout << "Read " << numVectors << " vectors of dimension " << dimension << " from " << inputFilename << std::endl;
            convertVecs2DimVecs<float>(inputFilename, outputFilename, dimension, numVectors, batchSize);
            std::vector<float> data;
            //readVectors<float>(inputFilename, data, dimension, numVectors);
            //printVectors<float>(data, dimension);
            //data.clear();
            readDimVecs<float>(outputFilename, data, dimension, numVectors);
            printVectors<float>(data, dimension);
        } else if (fileType == FileType::IVEC) {
            readVectorFileMetadata<int>(inputFilename, dimension, numVectors);
            std::cout << "Read " << numVectors << " vectors of dimension " << dimension << " from " << inputFilename << std::endl;
            convertVecs2DimVecs<int>(inputFilename, outputFilename, dimension, numVectors, batchSize);
            std::vector<int> data;
            //readVectors<int>(inputFilename, data, dimension, numVectors);
            //printVectors<int>(data, dimension);
            //data.clear();
            readDimVecs<int>(outputFilename, data, dimension, numVectors);
            printVectors<int>(data, dimension);
        } else {
            std::cerr << "Unknown file type: " << inputFilename << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    */

    return 0;
}