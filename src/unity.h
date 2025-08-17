#pragma once
#include<map>
#include<iostream>
#include<cstdlib>
#include<cstdio>
#include<cstring>
#include<string>
#include<vector>
#include<algorithm>
#include<sstream>
#include<cmath>
#include<assert.h>
#include<time.h>
#include<pthread.h>
#include <sys/time.h> 
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include "def.h"
#include "common.h"
namespace disk_hivf
{
    void UTF8ToCharSplit(const std::string &content, std::vector<std::string> &char_vector);
    void UTF8ToCharNoSplit(const std::string &content, std::vector<std::string> &char_vector);
    void GetNgrams(const std::vector<std::string> & words, int n, std::vector<std::string> &ngrams);
    std::string GetNgram(const std::vector<std::string> & words, int n, int i);
    inline Float sigmoid(Float d) { return (Float)1.0/(1.0+exp(-d)); } 
    inline void softmax(std::vector<Float> & lis) {
        Float m = lis[0];
        Float sum = 0.0;
        for (size_t i = 0; i < lis.size(); i++) {
            m = std::max(m, lis[i]);
        }
        for (size_t i = 0; i < lis.size(); i++) {
            lis[i] = exp(lis[i] - m);
            sum += lis[i];
        }
        for (size_t i = 0; i < lis.size(); i++) {
            lis[i] = lis[i] / sum;
        }
    }

	inline Int diff_in_vector(const std::vector<Int> & vec, Int tar) {
		Int ret = std::numeric_limits<Int>::max();
		for (Int a: vec) {
			if (a == -1) {
				continue;
			}
			ret = std::min(ret, std::abs(tar-a));
		}
		return ret;
	}

	class TimeStat
	{
		public:
			TimeStat(const std::string & prefix, bool is_print = true)
			{
				m_start = getCurrentTime();
				m_prefix = prefix;
				m_is_print = is_print;
				if (is_print) {
					fprintf(stderr, "%s begin\n",m_prefix.c_str());
				}
				//puts("---------------------------------------------------------------------------------------------------------\n");
			}

			long long TimeMark(const std::string & minstr)
			{
				long long tim = (long long)(getCurrentTime() - m_start);
				if (m_is_print) {
					fprintf(stderr, "%s %s TimeCost : %lldus\n",m_prefix.c_str(), minstr.c_str(), tim);
				}
				//puts("----------------------------------------------------------------------------------------------------------\n");
				return tim;
			}

			long long TimeCost()
			{
				long long tim = (long long)(getCurrentTime() - m_start);
				return tim;
			}

			~TimeStat()
			{
				if (m_is_print) {
					fprintf(stderr, "%s  TimeCost : %lldus\n", m_prefix.c_str(), (long long)(getCurrentTime() - m_start));
				}
				//puts("----------------------------------------------------------------------------------------------------------\n");
			}
			uint64_t getCurrentTime()  
			{  
				struct timeval tv;  
				gettimeofday(&tv,NULL);  
				return tv.tv_sec * 1000000 + tv.tv_usec;  
			}  
		private:
			uint64_t m_start;
			std::string m_prefix;
			bool m_is_print;
	};

	class ThreadLock
	{
		public:
			ThreadLock();
			~ThreadLock();
			int Lock();
			int UnLock();
		private:
			pthread_mutex_t m_mutex;
	};

	class ThreadLockGuard
	{
		public:
			ThreadLockGuard(ThreadLock * plock);
			~ThreadLockGuard();
			int Lock();
			int UnLock();
		private:
			ThreadLock * m_plock;
			bool m_islock;
		
	};


	template<typename ClassT> 
		ClassT* Delete(ClassT* p) {
			if ( NULL != p && p) {
				delete p; 
			}  
			p = NULL;
			return p; 
		}

	template<typename TypeT> 
		TypeT* Free(TypeT* p) {
			if ( NULL != p && p) {
				free(p);
			}  
			p = NULL; 
			return p; 
		}
	class stringHelper
	{
		public:
			static int split(const char *str,const char *spset,std::vector<std::string>&RetSet);
			static bool isInSpset(const char c,const char *spset);
	};
    template <typename T>
        std::string join(const std::vector<T> &vec, const std::string &sp) {
            std::ostringstream oss;
            oss.str("");
            for (size_t i = 0; i < vec.size(); i++) {
                if (i != 0) {
                    oss << sp;
                }
                oss << vec[i];
            }
            return oss.str();
        }

    template<class T> 
        T str2num(const std::string &str) {
            std::istringstream iss(str);
            T num;
            iss >> num;
            return num;
        }

    template<class T> 
        std::string num2str(const T num) {
            std::ostringstream oss;
            oss << num;
            return oss.str();
        }

	class Config
	{
		public:
			Config();
			virtual ~Config();
			virtual int Init(const char * configFile) = 0;
			int makePool( const char * configFile);
			std::string ToString();
			std::map<std::string,std::string> pool;
		private:
	};

	void convert_uint8_to_float(float* dst, const uint8_t* src, size_t count);
}

