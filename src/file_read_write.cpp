#include "file_read_write.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace disk_hivf {
    FileReadWriter::FileReadWriter(const std::string & file_dir, Int file_num, Int is_disk):
        file_dir_(file_dir), file_num_(file_num), is_disk_(is_disk) {
    }

    FileReadWriter::~FileReadWriter() {
        for (auto & file_stream : file_streams_) {
            if (file_stream && file_stream->is_open()) {
                file_stream->close();
            }
        }
    }

    Int FileReadWriter::Init() {
        try {
            struct stat info;
            if (stat(file_dir_.c_str(), &info) != 0) {
                if (mkdir(file_dir_.c_str(), 0777) != 0) {
                    std::cerr << "Failed to create directory" << std::endl;
                    return -1; // Failed to create directory
                }
            }
            for (Int i = 0; i < file_num_; ++i) {
                std::string file_path = file_dir_ + "/file_" + std::to_string(i) + ".dat";
                // Check if file exists
            std::ifstream ifs(file_path);
            if (!ifs) {
                // File does not exist, create it
                std::ofstream ofs(file_path);
                if (!ofs) {
                    std::cerr << "Failed to create file" << std::endl;
                    return -1; // Failed to create file
                }
                ofs.close();
            } else {
                ifs.close(); // Explicitly close the ifstream
            }
                // Open file streams and store them
                std::shared_ptr<std::fstream> file_stream(
                    new std::fstream(file_path,
                        std::ios::in | std::ios::out | std::ios::binary));
                if (!file_stream->is_open()) {
                    std::cerr << "Failed to open file" << std::endl;
                    return -1; // Failed to open file
                }
                file_streams_.push_back(file_stream);
            }
            if (!is_disk_) {
                mem_datas_.resize(file_num_);
                for (Int i = 0; i < file_num_; i++) {
                    Int ret = read(i, mem_datas_[i]);
                    if (ret < 0) {
                        std::cerr << "all mem mode fail" << std::endl;
                        return -1;
                    }
                }
            }
            return 0; // Success
        } catch (const std::exception & e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return -1;
        }
    }

    Int FileReadWriter::write(Int file_id, const char * data, Uint len) {
        //std::cout << "ptr2=" << data << std::endl;
        /*
        if (flag == 1) {
            Int features_id = *(reinterpret_cast<const Int *>(data));
            const float * features_ptr = reinterpret_cast<const float *>(data + sizeof(Int));
            Eigen::Map<const Eigen::RowVectorXf> features_mp(features_ptr, 128);
            std::cerr << features_id << std::endl;
            std::cerr << features_mp << std::endl << std::endl;
        }
        */
        if (file_id < 0 || file_id >= file_num_) {
            //LOG
            std::cerr << "ERR FileReadWriter.write fail file_id=" << file_id << std::endl; 
            return -1; // Invalid file_id
        }
        auto & ofs = file_streams_[file_id];
        ofs->seekp(0, std::ios::end);
        Int ret_offset = ofs->tellp();
        if (-1 == ret_offset) {
            std::cerr << "Failed to get the initial offset." << std::endl;
            return -1;
        }
        ofs->write(data, len);
        if (!ofs) {
            std::cerr << "ERR FileReadWriter.write fail err file_id=" << file_id << std::endl; 
            return -1; // Write failed
        }
        
        return ret_offset; // Success
    }

    Int FileReadWriter::clear(Int file_id) {
        std::string file_path = file_dir_ + "/file_" + std::to_string(file_id) + ".dat";
        if (file_id < 0 || file_id >= file_num_) {
            std::cerr << "Invalid file ID: " << file_id << std::endl;
            return -1;
        }

        auto &ofs = file_streams_[file_id];
        ofs->close();

        ofs->open(file_path,
            std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
        if (!ofs->is_open()) {
            std::cerr << "Failed to clear file: " << file_path << std::endl;
            return -1;
        }

        return 0;
    }

    Int FileReadWriter::clear() {
        for (Int i = 0; i < file_num_; i++) {
            Int ret = clear(i);
            if (ret < 0) {
                return -1;
            }
        }
        return 0;
    }

    Int FileReadWriter::read_matrix(Int file_id, Int offset, Uint len, 
                Int item_size, Int item_num, Int dim,
                std::vector<float> & matrix_data, std::vector<Int> & block_item_ids) {
        /*
        std::cout << "file_id=" << file_id
            << " offset=" << offset
            << " len=" << len << " item_size=" << item_size << " item_num=" << item_num
            << " dim=" << dim << std::endl;
        */
        //Int ex_num = sizeof(Int) / sizeof(float);
        block_item_ids.resize(item_num);
        std::vector<char> tmp_data;
        char * ptr = NULL;
        if (is_disk_) {
            Int ret = read(file_id, offset, len, tmp_data);
            if (ret < 0) {
                //LOG
                return -1;
            }
            ptr = tmp_data.data();
        } else {
            ptr = mem_datas_[file_id].data() + offset;
        }
        matrix_data.resize(dim * item_num);
        float * dest = matrix_data.data();
        for (Int i = 0; i < item_num; i++) {
                Int item_id = *(reinterpret_cast<Int*>(ptr + (i * item_size)));
                block_item_ids[i] = item_id;
                memcpy(dest + i * dim, reinterpret_cast<float *>(ptr + (i * item_size) + sizeof(Int)), dim * sizeof(float));
        }

        //Eigen::Map<RMatrixXf> block_map(reinterpret_cast<float*>(ptr),
        //        item_num, ex_num + dim);
        
        //matrix = block_map.block(0, ex_num, item_num, dim).transpose();
        //std::cout << matrix << std::endl;
        return 0;
    }

    FileType getFileType(const std::string& filename) {
        if (filename.size() >= 6 && filename.substr(filename.size() - 6) == ".bvecs") {
            return FileType::BVEC;
        } else if (filename.size() >= 6 && filename.substr(filename.size() - 6) == ".fvecs") {
            return FileType::FVEC;
        } else if (filename.size() >= 6 && filename.substr(filename.size() - 6) == ".ivecs") {
            return FileType::IVEC;
        } else {
            return FileType::UNKNOWN;
        }
    }

    Eigen::Map<RMatrixXf> readMatrixFromDimVecs(const std::string& filename, std::vector<float>& data) {
        Int ret = 0;
        int dimension;
        Int numVecs;
        ret = readDimVecs<float>(filename, data, dimension, numVecs);
        if (ret != 0) {
            std::cerr << "readMatrixFromDimVecs readDimVecs err" << std::endl;
            throw std::runtime_error("readMatrixFromDimVecs readDimVecs err");
        }
        return Eigen::Map<RMatrixXf>(data.data(), numVecs, dimension);
    }

    
}