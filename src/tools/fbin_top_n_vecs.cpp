#include "file_read_write.h"
using namespace disk_hivf;

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <input_filename> <output_filename> <top> <is2fvecs>" 
        << std::endl;
        return 1;
    }
    std::string inputFilename = argv[1];
    std::string outputFilename = argv[2];
    int top = std::stoi(argv[3]);
    int is2fvecs = std::stoi(argv[4]);
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
    std::cout << "dim=" <<dim << " num=" << num << " top="<< top << std::endl;

    size_t read_size = ((size_t)top) * (dim * sizeof(float));
    std::cout << " read_size=" << read_size << std::endl; 
    std::vector<char> buff(read_size);

    file.read(buff.data(), read_size);
    if (!file) {
        std::cerr << "batch read file fail filename=" << inputFilename << std::endl;
        return -1;
    }
    if (is2fvecs) {
        for (size_t i = 0; i < (size_t)top; i++) {
            outputFile.write(reinterpret_cast<const char*>(&dim), sizeof(int));
            outputFile.write(buff.data() + i * (dim * sizeof(float)), dim * sizeof(float));
            if (!outputFile) {
            std::cerr << "line write file fail filename=" << outputFilename << std::endl;
            return -1;
            }
        }

    } else {
        outputFile.write(reinterpret_cast<const char*>(&top), sizeof(int));
        outputFile.write(reinterpret_cast<const char*>(&dim), sizeof(int));

        if (!outputFile) {
            std::cerr << "write file fail filename=" << outputFilename << std::endl;
            return -1;
        }
        outputFile.write(buff.data(), read_size);
        if (!outputFile) {
            std::cerr << "batch write file fail filename=" << outputFilename << std::endl;
            return -1;
        }
    }
    file.close();
    outputFile.close();
    return 0;
}