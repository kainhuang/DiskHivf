#include "random.h"
#include "def.h"
#include <cstring>

namespace disk_hivf
{
	Float erf(Float x) {
		Float t;
		if (x >= 0) {
			t = 1.0 / (1.0 + 0.3275911 * x);
		} else {
			t = 1.0 / (1.0 - 0.3275911 * x);
		}

		Float result = 1.0 - (t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))))*exp(-x*x);
		if (x >= 0) {
			return result;
		} else {
			return -result;
		}
	}

	Float cdf_gaussian(Float x, Float mean, Float stdev) {
		return 0.5 + 0.5 * erf(0.707106781 * (x-mean) / stdev);
	}

	Float cdf_gaussian(Float x) {
		return 0.5 + 0.5 * erf(0.707106781 * x );
	}


	Float ran_left_tgaussian(Float left) {
		// draw a trunctated normal: acceptance region are values larger than <left>
		if (left <= 0.0) { // acceptance probability > 0.5
			return ran_left_tgaussian_naive(left);
		} else {
			// Robert: Simulation of truncated normal variables
			Float alpha_star = 0.5*(left + sqrt(left*left + 4.0));

			// draw from translated exponential distr:
			// f(alpha,left) = alpha * exp(-alpha*(z-left)) * I(z>=left)
			Float z,d,u;
			do {
				z = ran_exp() / alpha_star + left;
				d = z-alpha_star;
				d = exp(-(d*d)/2);
				u = ran_uniform();
				if (u < d) {
					return z;
				}
			} while (true);
		}
	}

	Float ran_left_tgaussian_naive(Float left) {
		// draw a trunctated normal: acceptance region are values larger than <left>
		Float result;
		do {
			result = ran_gaussian();
		} while (result < left);
		return result;
	}

	Float ran_left_tgaussian(Float left, Float mean, Float stdev) {
		return mean + stdev * ran_left_tgaussian((left-mean)/stdev); 
	}

	Float ran_right_tgaussian(Float right) {
		return -ran_left_tgaussian(-right);
	}

	Float ran_right_tgaussian(Float right, Float mean, Float stdev) {
		return mean + stdev * ran_right_tgaussian((right-mean)/stdev); 
	}



	Float ran_gamma(Float alpha) {
		assert(alpha > 0);
		if (alpha < 1.0) {
			Float u;
			do {
				u = ran_uniform();
			} while (u == 0.0);
			return ran_gamma(alpha + 1.0) * pow(u, 1.0 / alpha);
		} else {
			Float d,c,x,v,u;
			d = alpha - 1.0/3.0;
			c = 1.0 / std::sqrt(9.0 * d);
			do {
				do {
					x = ran_gaussian();
					v = 1.0 + c*x;
				} while (v <= 0.0);
				v = v * v * v;
				u = ran_uniform();
			} while ( 
				(u >= (1.0 - 0.0331 * (x*x) * (x*x)))
				&& (log(u) >= (0.5 * x * x + d * (1.0 - v + std::log(v))))
				);
			return d*v;
		}
	}

	Float ran_gamma(Float alpha, Float beta) {
		return ran_gamma(alpha) / beta;
	}

	Float ran_gaussian() {
		Float u,v, x, y, Q;
		do {
			do {
				u = ran_uniform();
			} while (u == 0.0); 
			v = 1.7156 * (ran_uniform() - 0.5);
			x = u - 0.449871;
			y = std::abs(v) + 0.386595;
			Q = x*x + y*(0.19600*y-0.25472*x);
			if (Q < 0.27597) { break; }
		} while ((Q > 0.27846) || ((v*v) > (-4.0*u*u*std::log(u)))); 
		return v / u;
	}

	Float ran_gaussian(Float mean, Float stdev) {
		if ((stdev == 0.0) || (std::isnan(stdev))) {
			return mean;
		} else {
			return mean + stdev*ran_gaussian();
		}
	}

	Float ran_uniform() {
		return rand()/((Float)RAND_MAX + 1);
	}

	Float ran_exp() {
		return -std::log(1-ran_uniform());
	}

	bool ran_bernoulli(Float p) {
		return (ran_uniform() < p);
	}

	int rand_m_nums(Kiss32Random & ks, uint32_t n, uint32_t m, std::vector<uint32_t> & m_nums) {
		if (m > n) {
			return -1;
		}
		std::vector<uint32_t> nums(n);
		std::iota(nums.begin(), nums.end(), 0);
		for (uint32_t i = 0; i < m; i++) {
			uint32_t rd = ks.kiss() % (n - i);
			std::swap(nums[i], nums[rd + i]);
		}
		m_nums.resize(m);
		memcpy(m_nums.data(), nums.data(), m * sizeof(uint32_t));
		return 0;
	}
}


