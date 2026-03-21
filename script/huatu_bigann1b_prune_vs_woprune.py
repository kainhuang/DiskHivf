import matplotlib.pyplot as plt

# DiskHivf 数据 (单位: 毫秒)
diskhivf_lacency = [
    5.99458,
    7.19837,
    8.21536,
    9.03153,
    10.5686,
    12.1715,
    15.2652,
    17.0178,
    18.5945,
    19.869,
]
diskhivf_recall = [
    89.3000,
    91.5770,
    92.7270,
    93.2440,
    94.3650,
    95.1560,
    96.2890,
    96.6600,
    96.9910,
    97.2300,
]

# DiskHivf 数据 (单位: 毫秒)
diskhivf_wo_prune_lacency = [
    7.86397, 
    10.2203, 
    12.6334, 
    15.243, 
    17.1636, 
    19.0333, 
    25.026, 
    26.8204, 
    28.4557, 
]
diskhivf_wo_prune_recall = [
    89.911,
    92.559,
    94.139,
    95.086,
    95.784,
    96.311,
    97.052,
    97.3,
    97.501,
]

# 创建折线图
plt.figure(figsize=(10, 6))

plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
plt.plot(diskhivf_wo_prune_lacency, diskhivf_wo_prune_recall, marker='o', label='DiskHivf-wo_prune')


# 添加标题和标签
plt.title('bigann1B 10-recall@10 vs. Latency')
plt.xlabel('Latency (ms)')
plt.ylabel('10-recall@10')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()