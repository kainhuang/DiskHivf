#include <iostream>
#include "random.h"
#include "unity.h"
#include "def.h"
#include "common.h"
using namespace std;
using namespace disk_hivf;

int main() {
    TimeStat st("zxxxxxxxxxx");
    cout << disk_hivf::MAX_UINT <<endl;
    Kiss32Random ks;
        // 定义一个概率分布
    std::vector<double> dist = {0.1, 0.2, 0.3, 0.4};

    // 创建 randDist 对象
    randDist<double> rd(ks, dist);

    // 采样并统计结果
    std::vector<int> counts(dist.size(), 0);
    int num_samples = 10000;

    for (int i = 0; i < num_samples; ++i) {
        int sample = rd.sample();
        counts[sample]++;
    }

    // 输出结果
    std::cout << "采样结果：" << std::endl;
    for (size_t i = 0; i < counts.size(); ++i) {
        std::cout << "区间 " << i << ": " << counts[i] << " 次 (" << (counts[i] / static_cast<double>(num_samples)) * 100 << "%)" << std::endl;
    }
    return 0;
}
