#pragma once
#include <iostream>
#include <queue>
#include <vector>
#include <functional>

namespace disk_hivf {
template <typename T>
    class LimitedMaxHeap {
    public:
        LimitedMaxHeap(size_t capacity): capacity_(capacity) {}

        void push(const T& value) {
            if (heap_.size() < capacity_) {
                heap_.push(value);
            } else if (value < heap_.top()) {
                pre = heap_.top();
                heap_.pop();
                heap_.push(value);
            }
            //std::cout << "heap size() " << size() << std::endl;
        }

        T top() const {
            return heap_.top();
        }

        void pop() {
            heap_.pop();
        }

        bool empty() const {
            return heap_.empty();
        }

        size_t size() const {
            return heap_.size();
        }

        bool is_full() const {
            return heap_.size() >= capacity_;
        }

        T get_pre() const {
            return pre;
        }

        size_t capacity() const {
            return capacity_;
        }
    private:
        size_t capacity_;
        std::priority_queue<T> heap_;
        T pre;
    };
}