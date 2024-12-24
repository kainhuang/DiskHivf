#include "file_read_write.h"
#include "random.h"
using namespace disk_hivf;

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_filename> <output_filename> <rand_num>" 
        << std::endl;
        return 1;
    }
    std::string inputFilename = argv[1];
    std::string outputFilename = argv[2];
    Int rand_num = std::stoi(argv[3]);
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
    int dim;
    file.read(reinterpret_cast<char*>(&dim), sizeof(int));
    Int num;
    file.read(reinterpret_cast<char*>(&num), sizeof(Int));

    if (!file) {
        std::cerr << "read file fail filename=" << inputFilename << std::endl;
        return -1;
    }
    std::cout << "dim=" <<dim << " num=" << num << std::endl;
    outputFile.write(reinterpret_cast<const char*>(&dim), sizeof(int));
    outputFile.write(reinterpret_cast<const char*>(&rand_num), sizeof(Int));
    if (!outputFile) {
        std::cerr << "write file fail filename=" << outputFilename << std::endl;
        return -1;
    }
    Int vec_size = dim * sizeof(float);
    std::vector<char> out_buff(rand_num * vec_size);
    file.read(out_buff.data(), rand_num * vec_size);
    Kiss32Random ks;

    std::vector<char> read_buff(vec_size);
    for (Int i = rand_num; i < num; i++) {
        file.read(read_buff.data(), vec_size);
        uint32_t rd = ks.kiss() % i;
        if (rd < rand_num) {
            std::memcpy(out_buff.data() + rd * vec_size, read_buff.data(), vec_size);
        }
    }
    outputFile.write(out_buff.data(), rand_num * vec_size);
    file.close();
    outputFile.close();
    return 0;
}