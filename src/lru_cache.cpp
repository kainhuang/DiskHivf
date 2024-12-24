#include "lru_cache.h"

namespace disk_hivf {

    Int Segment::get(const Int key,
        const Eigen::Ref<const Eigen::RowVectorXf> & feature,
        LimitedMaxHeap<Result> & result_heap
        ) {
        {
            boost::shared_lock<boost::shared_mutex> lock(mutex_);
            auto it = cache_map_.find(key);
            if (it == cache_map_.end()) {
                return 0;
            }

            float * data_ptr = it->second->m_data.data();
            size_t data_size = it->second->m_data.size();
            FeatureId * ids_ptr = it->second->m_ids.data();
            size_t ids_size = it->second->m_ids.size();
            Eigen::Map<RMatrixDf> block_features(data_ptr, 
                ids_size, data_size / ids_size);
            Eigen::VectorXf query2block_features_dist = 
            (block_features.rowwise() - feature).rowwise().squaredNorm();
            for (size_t i = 0; i < ids_size; i++) {
                result_heap.push(
                    Result(ids_ptr[i], query2block_features_dist(i), 0, 0));
            }
            // Move the accessed item to the front of the list
            {
                boost::unique_lock<boost::shared_mutex> unique_lock(mutex_);
                cache_items_.splice(cache_items_.begin(), cache_items_, it->second);
            }
            return ids_size;
        }
    }

    void Segment::put(const Int& key, const FeatureId * ids, const float * data, Int len, int dim) {
        boost::unique_lock<boost::shared_mutex> lock(mutex_);
        auto it = cache_map_.find(key);
        if (it != cache_map_.end()) {
            // move the item to the front of the list
            cache_items_.splice(cache_items_.begin(), cache_items_, it->second);
        } else {
            // If the cache is full, remove the least recently used item
            if (size_ >= capacity_) {
                auto last = cache_items_.end();
                --last;
                size_ -= last->m_ids.size();
                cache_map_.erase(last->m_key);
                cache_items_.pop_back();
            }
            // Insert the new item at the front of the list
            cache_items_.emplace_front(key, ids, data, len, dim);
            size_ += len;
            cache_map_[key] = cache_items_.begin();
        }
    }

    ThreadSafeLRUCache::ThreadSafeLRUCache(size_t capacity, size_t num_segments)
        : capacity_(capacity), num_segments_(num_segments) {
        if (capacity > 0 && num_segments > 0) {
            size_t segment_capacity = (capacity + num_segments - 1) / num_segments;
            for (size_t i = 0; i < num_segments; ++i) {
                Segment * ptr = new Segment(segment_capacity);
                segments_.push_back(ptr);
            }
        }
    }

    ThreadSafeLRUCache::~ThreadSafeLRUCache() {
        for (size_t i = 0; i < num_segments_; i++) {
            delete segments_[i];
        }
    }

    Int ThreadSafeLRUCache::get(const Int key, 
        const Eigen::Ref<const Eigen::RowVectorXf> & feature,
        LimitedMaxHeap<Result> & result_heap) {
        size_t segment_index = hash(key) % num_segments_;
        return segments_[segment_index]->get(key, feature, result_heap);
    }

    void ThreadSafeLRUCache::put(const Int key, const FeatureId * ids, const float * data, Int len, int dim) {
        size_t segment_index = hash(key) % num_segments_;
        segments_[segment_index]->put(key, ids, data, len, dim);
    }
}