#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <assert.h>
#include <numeric>
#include "def.h"
#include "common.h"
#include <vector>
#include <algorithm>

namespace disk_hivf {
    Float ran_gaussian();
    Float ran_gaussian(Float mean, Float stdev);
    Float ran_left_tgaussian(Float left);
    Float ran_left_tgaussian(Float left, Float mean, Float stdev);
    Float ran_left_tgaussian_naive(Float left);
    Float ran_uniform();
    Float ran_exp();			
    Float ran_gamma(Float alpha, Float beta);
    Float ran_gamma(Float alpha);
    bool ran_bernoulli(Float p);

    Float erf(Float x);	
    Float cdf_gaussian(Float x, Float mean, Float stdev);
    Float cdf_gaussian(Float x);

    struct Kiss32Random {
        uint32_t x;
        uint32_t y;
        uint32_t z;
        uint32_t c;
        bool is_set_seed;

        // seed must be != 0
        Kiss32Random(uint32_t seed = 123456789) {
            x = seed;
            y = 362436000;
            z = 521288629;
            c = 7654321;
            is_set_seed = false;
        }

        void set_seed(uint32_t seed) {
            //fprintf(stderr, "set seed %u\n", seed);
            z = seed;
            is_set_seed = true;
        }

        uint32_t kiss() {
            // Linear congruence generator
            x = 69069 * x + 12345;

            // Xor shift
            y ^= y << 13;
            y ^= y >> 17;
            y ^= y << 5;

            // Multiply-with-carry
            uint64_t t = 698769069ULL * z + c;
            c = t >> 32;
            z = (uint32_t) t;

            return x + y + z;
        }
        inline int flip() {
            // Draw random 0 or 1
            return kiss() & 1;
        }
        inline size_t index(size_t n) {
            // Draw random integer between 0 and n-1 where n is at most the number of data points you have
            return kiss() % n;
        }

        inline Float ran_uniform() {
            return kiss() / ((Float)MAX_UINT + 1);
        }
    };

    int rand_m_nums(Kiss32Random& ks, uint32_t n, uint32_t m, std::vector<uint32_t> & m_nums);

    template <typename T> 
    class randDist {
        public:
            randDist(Kiss32Random & ks, const std::vector<T> & dist): _sum(0), _ks(ks) {
                _sum_dist.resize(dist.size(), 0);
                for (uint32_t i = 0; i < dist.size(); i++) {
                    _sum += dist[i];
                    _sum_dist[i] = _sum;
                }
            }
            int sample() {
                T r = _ks.ran_uniform() * _sum;
                auto it = std::upper_bound(_sum_dist.begin(), _sum_dist.end(), r);
                return it - _sum_dist.begin();
            }
        private:
            std::vector<T> _sum_dist;
            T _sum;
            Kiss32Random _ks;
    };
}
