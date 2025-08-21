#pragma once
#include <iostream>
#include <unordered_map>
#include <list>
#include <vector>
//#include <boost/thread/shared_mutex.hpp>
//#include <boost/thread/locks.hpp>
#include <stdexcept>
#include <thread>
#include "common.h"
#include "Eigen/Dense"
#include "heap.h"


namespace disk_hivf {

    struct Result {
        Result(Int vec_id, float distance, int rank_id, int searched_num):
            m_vec_id(vec_id), m_distance(distance),
            m_rank_id(rank_id), m_searched_num(searched_num) {}
        Result() {
            m_vec_id = -1;
            m_distance = -1;
            m_rank_id = 0;
            m_searched_num = 0;
        }
        inline bool operator < (const Result & other) const {
            if (m_distance != other.m_distance) {
                return m_distance < other.m_distance;
            } else {
                return m_vec_id < other.m_vec_id;
            }
        }
        Int m_vec_id;
        float m_distance;
        int m_rank_id;
        int m_searched_num;
    };
    /*
    struct CacheData {
        CacheData(const Int key, const FeatureId * ids,
            const float * data, Int len, int dim): m_key(key) {
            m_ids.resize(len);
            m_data.resize(len * dim);
            std::memcpy(m_ids.data(), ids, len * sizeof(FeatureId));
            std::memcpy(m_data.data(), data, len * dim * sizeof(float));
        }
        Int m_key;
        std::vector<FeatureId> m_ids;
        std::vector<float>m_data;
    };
    

    class Segment {
    public:
        Segment(size_t capacity) : size_(0), capacity_(capacity) {}
        Segment() : size_(0), capacity_(0) {}
        Int get(const Int key,
            const Eigen::Ref<const Eigen::RowVectorXf> & feature,
            LimitedMaxHeap<Result> & result_heap
            );

        void put(const Int& key, const FeatureId * ids, const float * data, Int len, int dim);

    private:
        size_t size_;
        size_t capacity_;
        std::list<CacheData> cache_items_;
        std::unordered_map<Int, typename std::list<CacheData>::iterator> cache_map_;
        mutable boost::shared_mutex mutex_;
    };

   
    class ThreadSafeLRUCache {
    public:
        ThreadSafeLRUCache(size_t capacity, size_t num_segments);
        ~ThreadSafeLRUCache();

        Int get(const Int key, 
            const Eigen::Ref<const Eigen::RowVectorXf> & feature,
            LimitedMaxHeap<Result> & result_heap);

        void put(const Int key, const FeatureId * ids, const float * data, Int len, int dim);

    private:
        size_t capacity_;
        size_t num_segments_;
        std::vector<Segment *> segments_;
        std::hash<Int> hash;
    };
    */
}