#pragma once

#include <cstdlib>
#include <cstring>
#include <cstddef>

namespace disk_hivf {

// O_DIRECT模式需要512字节对齐的内存
// 使用posix_memalign分配，C++11兼容实现
class AlignedBuffer {
public:
    static const size_t ALIGNMENT = 512;

    AlignedBuffer()
        : data_(NULL), size_(0), capacity_(0) {
    }

    explicit AlignedBuffer(size_t size)
        : data_(NULL), size_(0), capacity_(0) {
        resize(size);
    }

    ~AlignedBuffer() {
        if (data_) {
            free(data_);
            data_ = NULL;
        }
        size_ = 0;
        capacity_ = 0;
    }

    // 移动构造函数
    AlignedBuffer(AlignedBuffer&& other)
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
        other.data_ = NULL;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    // 移动赋值运算符
    AlignedBuffer& operator=(AlignedBuffer&& other) {
        if (this != &other) {
            if (data_) {
                free(data_);
            }
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = NULL;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    // 调整大小：仅当请求大小超过当前capacity时才重新分配
    void resize(size_t new_size) {
        if (new_size > capacity_) {
            // 对齐到512字节的整数倍
            size_t aligned_size = (new_size + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
            void* new_data = NULL;
            int ret = posix_memalign(&new_data, ALIGNMENT, aligned_size);
            if (ret != 0 || new_data == NULL) {
                return; // 分配失败
            }
            // 拷贝旧数据
            if (data_ && size_ > 0) {
                size_t copy_size = size_ < new_size ? size_ : new_size;
                memcpy(new_data, data_, copy_size);
            }
            if (data_) {
                free(data_);
            }
            data_ = static_cast<char*>(new_data);
            capacity_ = aligned_size;
        }
        size_ = new_size;
    }

    char* data() { return data_; }
    const char* data() const { return data_; }
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }

private:
    // 禁用拷贝构造和拷贝赋值
    AlignedBuffer(const AlignedBuffer&);
    AlignedBuffer& operator=(const AlignedBuffer&);

    char* data_;
    size_t size_;
    size_t capacity_;
};

} // namespace disk_hivf
