#include <fstream>
#include "sstream"
#include "unity.h"
#include "Log.h"
namespace disk_hivf
{
    // UTF8ToCharSplit函数是将utf8的短语切分成字符，且对于英文、标点都会被切分开，比如robot会被切分成r、o、b、o、t
    void UTF8ToCharSplit(const std::string &content, std::vector<std::string> &char_vector) {
        int i = 0;
        int content_size = content.size();
        while (i < content_size) {
            int char_size = 1;
            unsigned char char_temp = (unsigned char) content[i];
            if (char_temp & 0x80) {
                char_temp <<= 1;
                do {
                    char_temp <<= 1;
                    char_size++;
                } while (char_temp & 0x80);
            }
            std::string single_char = content.substr(i, char_size);
            char_vector.push_back(single_char);
            i += char_size;
        }
    }

    // UTF8ToCharNoSplit函数是将utf8的短语切分成字符，但是对于英文、标点是不会被切分开，比如robot切分后还是robot
    void UTF8ToCharNoSplit(const std::string &content, std::vector<std::string> &char_vector) {
        int i = 0;
        int content_size = content.size();
        char_vector.clear();
        while (i < content_size) {
            unsigned char char_temp = (unsigned char)content[i];
            int beg = i;
            if (char_temp < 0x80) {
                if (char_temp == ' ') {
                    i++;
                } else {
                    while (i < content_size &&
                            char_temp < 0x80 && char_temp != ' ') {
                        char_temp = (unsigned char)content[++i];
                    }
                }
            } else {
                do{
                    char_temp <<= 1;
                    i++;
                } while (char_temp & 0x80);
            }
            std::string single_char = content.substr(beg, i - beg);
            char_vector.push_back(single_char);
        }
    } 

    void GetNgrams(const std::vector<std::string> & words, int n, std::vector<std::string> &ngrams) {
        for (int i = 0; i < (int)words.size(); i++) {
            ngrams.push_back(GetNgram(words, n, i));
        }
    }

    std::string GetNgram(const std::vector<std::string> &words, int n, int i) {
        int r = n / 2;
        std::vector<std::string> ngram;
        for (int j = i - r; j < i - r + n; j++) {
            if (j < 0) {
                ngram.push_back("<S>");
            } else if (j >= (int)words.size()) {
                ngram.push_back("<E>");
            } else {
                ngram.push_back(words[j]);
            }
        }
        return join(ngram, "_");
    }

    ThreadLock::ThreadLock()
    {
        assert(0 == pthread_mutex_init(&m_mutex,NULL));
    }

    ThreadLock::~ThreadLock()
	{
		assert(0 == pthread_mutex_destroy(&m_mutex));
	}

	int ThreadLock::Lock()
	{
		int ret = pthread_mutex_lock(&m_mutex);
		return ret;
	}

	int ThreadLock::UnLock()
	{
		return pthread_mutex_unlock(&m_mutex);
	}

	ThreadLockGuard::ThreadLockGuard(ThreadLock * plock)
	{
		m_plock = plock;
		m_islock = false;
	}

	ThreadLockGuard::~ThreadLockGuard()
	{
		UnLock();
	}

	int ThreadLockGuard::Lock()
	{
		int ret = m_plock->Lock();
		if(0 == ret)
		{
			m_islock = true;
		}
		return ret;
	}

	int ThreadLockGuard::UnLock()
	{
		int ret = 0;
		if(m_islock)
		{
			ret = m_plock->UnLock();
			if(0 == ret) m_islock = false;
		}
		return ret;
	}

	int stringHelper::split(const char *str,const char *spset,std::vector<std::string>&RetSet)
	{
		int len=strlen(str);
		char *tmp=new char[len+5];
		int i=0;
		int j=0;
		int cnt=0;
		while(i<len)
		{
			while(i<len&&isInSpset(str[i],spset))i++;
			while(i<len&&!isInSpset(str[i],spset))
			{
				tmp[j]=str[i];
				j++;
				i++;
			}
			if(j!=0)
			{
				tmp[j]='\0';
				std::string buf=tmp;
				RetSet.push_back(buf);
				cnt++;
				j=0;
			}
		}

		delete[] tmp;
		return cnt;
	}

	bool stringHelper::isInSpset(char c,const char *spset)
	{
		if(spset==NULL)return false;
		for(int i=0;spset[i];i++)
		{
			if(c==spset[i])return true;
		}
		return false;
	}

	Config::Config(){}
	
	Config::~Config(){}

	int Config::makePool(const char *configFile)
	{
		std::ifstream fpconfig(configFile);
		if( !fpconfig )
		{
			LogErr("Config::makePool %s not exit!",configFile);
			return -1;
		}
		pool.clear();
		std::string line;
		while(getline(fpconfig , line))
		{
			if('#'==line[0])continue;
			std::vector<std::string> col;
			stringHelper::split(line.c_str()," =\r\n",col);
			if(col.size() < 2)
			{
				continue;
			}
			pool[col[0]]=col[1];
		}
		fpconfig.close();
		return 0;
	}

	std::string Config::ToString()
	{
		std::ostringstream oss;
		for(std::map<std::string,std::string>::iterator i = pool.begin(); i != pool.end();i++)
		{
			oss<<(*i).first<<"=["<<(*i).second<<"]\n";
		}
		return oss.str();
	}
}
