#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>
#include <cstring>
#include "Eigen/Dense"
#include "common.h"
#include "unity.h"
#include <mutex>
#include <omp.h>
#include <thread>
#include <unordered_map>

namespace disk_hivf {
    enum class FileType {
        BVEC,
        FVEC,
        IVEC,
        UNKNOWN
    };

    FileType getFileType(const std::string& filename);


    // Template function to read vector file metadata
    template <typename T>
    Int readVectorFileMetadata(const std::string& filename, int& dim, Int& numVectors) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Cannot open file: " << filename << std::endl;
            return -1;
        }

        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (file.gcount() != sizeof(int)) {
            std::cerr << "Error reading dimension from file: " << filename << std::endl;
            return -1;
        }

        file.seekg(0, std::ios::end);
        std::streampos fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        numVectors = fileSize / (4 + dim * sizeof(T));
        file.close();
        return 0;
    }


    template<typename T>
    Int readDimVecs(const std::string& filename, std::vector<T>& data, int& dimension, Int& numVecs) {
        TimeStat ts("readDimVecs " + filename);
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Cannot open file: " << filename << std::endl;
            return -1;
        }
        
        int d;
        file.read(reinterpret_cast<char*>(&d), sizeof(int));
        if (!file) {
            std::cerr << "Error reading dimension from file: " << filename << std::endl;
            return -1;
        }
        Int num;
        file.read(reinterpret_cast<char*>(&num), sizeof(Int));
        if (!file) {
            std::cerr << "Error reading numVecs from file: "  << filename << std::endl;
            return -1;
        }
        dimension = d;
        numVecs = num;
        data.resize(num * d);
        file.read(reinterpret_cast<char*>(data.data()), num * d * sizeof(T));
        if (!file) {
            std::cerr << "Error reading Vecs from file: "  << filename << std::endl;
            return -1;
        }
        return 0;
    }


    Eigen::Map<RMatrixXf> readMatrixFromDimVecs(const std::string& filename, std::vector<float>& data);


    template<typename T>
    Int readVectors(const std::string& filename, std::vector<T>& data, int& dimension, int& numVecs) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Cannot open file: " << filename << std::endl;
            return -1;
        }

        int d;
        file.read(reinterpret_cast<char*>(&d), sizeof(int));
        if (!file) {
            std::cerr << "Error reading dimension from file: " << filename << std::endl;
            return -1;
        }

        dimension = d;
        std::vector<T> buffer(d);
        while (file.read(reinterpret_cast<char*>(buffer.data()), d * sizeof(T))) {
            data.insert(data.end(), buffer.begin(), buffer.end());
            int next_d;
            file.read(reinterpret_cast<char*>(&next_d), sizeof(int));
            if (file && next_d != d) {
                std::cerr << "Inconsistent vector dimensions in file: " << filename << std::endl;
                return -1;
            }
        }
        numVecs = data.size() / d;
        return 0;
    }


    template<typename T>
    void printVectors(const std::vector<T>& data, int dimension) {
        std::cout << "Dimension: " << dimension << std::endl;
        Int numVectors = data.size() / dimension;
        for (Int i = 0; i < numVectors; ++i) {
            std::cout << "Vector " << i + 1 << ": ";
            for (Int j = 0; j < dimension; ++j) {
                std::cout << static_cast<Int>(data[i * dimension + j]) << " ";
            }
            std::cout << std::endl;
        }
    }

    template<typename T>
    Int convertVecs2DimVecs(const std::string& inputFilename, const std::string& outputFilename,
        int dim, Int numVectors, Int batchSize) {
        std::ifstream inputFile(inputFilename, std::ios::binary);
        if (!inputFile) {
            std::cerr << "Cannot open input file: " << inputFilename << std::endl;
            return -1;
        }

        std::ofstream outputFile(outputFilename, std::ios::binary);
        if (!outputFile) {
            std::cerr << "Cannot open output file: " << outputFilename << std::endl;
            return -1;
        }

        outputFile.write(reinterpret_cast<const char*>(&dim), 4);
        outputFile.write(reinterpret_cast<const char*>(&numVectors), sizeof(Int));

        size_t buffer_size = batchSize * (4 + dim * sizeof(T));
        std::unique_ptr<char> buffer(new char[buffer_size]);
        
        size_t out_buffer_size = batchSize * (dim * sizeof(T));
        std::unique_ptr<char> out_buffer(new char[out_buffer_size]);

        std::vector<float> convert_buffer(out_buffer_size, 0);

        for (Int i = 0; i < numVectors; i += batchSize) {
            Int currentBatchSize = std::min(batchSize, numVectors - i);
            size_t read_size = currentBatchSize * (4 + dim * sizeof(T));
            // Read a batch of vectors
            inputFile.read(buffer.get(), read_size);
            if (inputFile.gcount() != read_size) {
                std::cerr << "Error reading vector data from input file: " << inputFilename << std::endl;
                return -1;
            }
            char * dest = out_buffer.get();
            char * src = buffer.get();
            for (Int j = 0; j < currentBatchSize; j++) {
                memcpy(dest, src + 4, dim * sizeof(T));
                dest += dim * sizeof(T);
                src += (4 + dim * sizeof(T));
            }
            if (sizeof(T) < 4) {
                T * ptr = reinterpret_cast<T *>(out_buffer.get());
                for (Int j = 0; j < currentBatchSize * dim; j++) {
                    convert_buffer[j] = static_cast<float>(ptr[j]);
                }
                outputFile.write(reinterpret_cast<const char *>(convert_buffer.data()), currentBatchSize * dim * sizeof(float));
            }
            else {
                // Write the batch of vectors to the output file
                outputFile.write(out_buffer.get(), currentBatchSize * dim * sizeof(T));
                if (!outputFile) {
                    std::cerr << "Error writing vector data to output file: " << outputFilename << std::endl;
                    return -1;
                }
            }
        }

        inputFile.close();
        outputFile.close();
        return 0;
    }

    class FileReadWriter {
        public:
            /*
            FileReadWriter是一个文件读写类，能按照输入的文件id，实现将数据分散到多个文件读写的能力
            */

            /*构造函数
            输入:
            file_dir:文件目录,如果不存在则创建目录
            file_num:文件个数
            */
            FileReadWriter(const std::string & file_dir, Int file_num, Int is_disk = 1);

            ~FileReadWriter();

            /*初始化函数，创建文件，
            */
            Int Init();

            /*写函数
            输入:
            file_id:文件ID，将数据写到该id的文件
            data:数据的起始指针
            len:数据的长度
            */
            Int write(Int file_id, const char * data, Uint len);

            /*读函数
            输入:
            file_id:文件ID，读该文件所有的数据
            data:数据指针，用来返回数据
            */
            template<typename T>
            Int read(Int file_id, std::vector<T>& data){
                if (file_id < 0 || file_id >= file_num_) {
                    std::cerr << "FileReadWriter::read fail file id=" << file_id << std::endl;  
                    return -1; // Invalid file_id
                }

                auto & ifs = file_streams_[file_id];
                
                if ((!ifs) || (!ifs->is_open())) {
                    std::cerr << "FileReadWriter::read fail file is not open file id=" 
                        << file_id << std::endl; 
                    return -1;
                }
                
                ifs->seekg(0, std::ios::end);
                std::streamsize size = ifs->tellg();
                ifs->seekg(0, std::ios::beg);
                data.resize(size / sizeof(T));
                ifs->read(reinterpret_cast<char*>(data.data()), size);
                if (!ifs) {
                    std::cerr << "FileReadWriter::read data fail file id=" << file_id << std::endl; 
                    return -1; // Read failed
                }

                return size; // Return the number of T read
            }
            /*读函数
            输入:
            file_id:文件ID，读该文件的数据
            offset:读文件的offset
            len:读取数据的长度
            data:数据指针，用来返回数据
            */
            template<typename T>
            Int read(Int file_id, Int offset, Uint len, std::vector<T>& data) {
                //TimeStat ts("read");
                //std::vector<Int> time_stat(4, 0);
                if (file_id < 0 || file_id >= file_num_) {
                    std::cerr << "read err file_id=" << file_id 
                        << " offset=" << offset << " len=" << len << std::endl;
                    return -1; // Invalid file_id
                }
                std::thread::id main_id = std::this_thread::get_id();
                //std::cout << "Main Thread ID: " << main_id << std::endl;
                std::string key = num2str(main_id) + "_" + num2str(file_id);
                std::shared_ptr<std::fstream> ifs;
                {
                    std::lock_guard<std::mutex> lock(fs_cache_mutex_);
                    if (fs_cache_.find(key) != fs_cache_.end()) {
                        ifs = fs_cache_[key];
                    } else {
                        //std::cout << "open" << std::endl;
                        std::string file_path = file_dir_ + "/file_" + std::to_string(file_id) + ".dat";
                        ifs = std::make_shared<std::fstream>(file_path,
                                std::ios::in | std::ios::out | std::ios::binary);
                        fs_cache_[key] = ifs;
                    }
                }
                //time_stat[0] += ts.TimeCost();
                //auto & ifs = file_streams_[file_id];
                if (!ifs || !ifs->is_open()) {
                    std::cerr << "read error: file stream is not open for file_id=" << file_id << std::endl;
                    return -1; // File stream is not open
                }

                ifs->clear(); // Clear any existing errors
                ifs->seekg(offset, std::ios::beg);
                if (!ifs) {
                    std::cerr << "read error: seekg failed for file_id=" << file_id 
                            << " offset=" << offset << std::endl;
                    return -1; // Seek failed
                }
                //time_stat[1] += ts.TimeCost();
                data.resize(len);
                //time_stat[2] += ts.TimeCost();
                ifs->read(reinterpret_cast<char*>(data.data()), len * sizeof(T));
                if (!ifs) {
                    std::cerr << "read fail file_id=" << file_id 
                        << " offset=" << offset << " len=" << len << std::endl;
                    return -1; // Read failed
                }
                /*
                time_stat[3] += ts.TimeCost();
                for (auto & a: time_stat) {
                    std::cout << a << " ";
                }
                std::cout << len << std::endl;
                */
                return ifs->gcount(); // Return the number of bytes read
            }

            Int clear(Int file_id);

            Int clear();

            inline Int get_file_num() const {
                return file_num_;
            }
            
            inline char * get_mem_ptr(Int file_id, Int offset) {
                return mem_datas_[file_id].data() + offset;
            }
            
            Int read_matrix(Int file_id, Int offset, Uint len, Int item_size,
                Int item_num, Int dim,
                std::vector<float> & matrix_data, std::vector<FeatureId> & block_item_ids, 
                std::vector<Int> & time_stat);

        private:
            std::string file_dir_;
            Int file_num_;
            Int is_disk_;
            std::vector<std::shared_ptr<std::fstream>> file_streams_;
            std::vector<std::vector<char>> mem_datas_;
            std::unordered_map<std::string, std::shared_ptr<std::fstream>> fs_cache_;
            std::mutex fs_cache_mutex_;
    };

}