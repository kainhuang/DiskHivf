import numpy as np
import matplotlib.pyplot as plt

# 示例数据

lines = open('length_of_search_list_data').readlines()
data = [int(a.strip()) for a in lines if int(a.strip()) < 400]

# 对数据进行排序
sorted_data = np.sort(data)

# 计算累积分布函数值
cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

# 计算80%分位和99%分位
percentile_80 = np.percentile(data, 80)
percentile_99 = np.percentile(data, 99)

# 绘制累积分布曲线
plt.plot(sorted_data, cdf, marker='.', linestyle='none', label='CDF')

# 标注80%分位
plt.axvline(x=percentile_80, color='r', linestyle='--', label='80th Percentile')
plt.text(percentile_80, 0.8, f'80th: {percentile_80:.0f}', color='r', ha='right')

# 标注99%分位
plt.axvline(x=percentile_99, color='g', linestyle='--', label='99th Percentile')
plt.text(percentile_99, 0.99, f'99th: {percentile_99:.0f}', color='g', ha='right')

# 添加标签和标题
plt.xlabel('length of search list')
plt.ylabel('CDF')
plt.title('Cumulative Distribution Function with Percentiles')
plt.legend()
plt.grid(True)
plt.show()