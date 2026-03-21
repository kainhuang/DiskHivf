import matplotlib.pyplot as plt

# DiskHivf 数据 (单位: 毫秒)
diskhivf_lacency = [
    7.21578,
    10.4984,
    17.3374,
    23.8222,
    33.8563,
    49.2766,
    65.3695,
    77.807,
]
diskhivf_recall = [
    84.500000,
    90.199997,
    93.900002,
    95.699997,
    97.599998,
    98.900002,
    99.300003,
    99.400002,
]



# Diskann 数据 (单位: 毫秒)
diskann_lacency = [
    9.10942,
    11.24608,
    13.37337,
    15.53745,
    17.65365,
    19.81222,
    21.91856,
    24.12696,
    35.06465,
    46.2173,
    57.40882,
    68.67006,
    80.03794,
    91.19913,
]
diskann_recall = [
    80.6,
    83.9,
    87.4,
    89.4,
    91.7,
    92.6,
    93.7,
    94.5,
    96.8,
    97.4,
    97.8,
    98.4,
    98.9,
    99.2,
]

# spann 数据 (单位: 毫秒)
spann_lacency = [
    10.686,
    14.076,
    17.445,
    25.697,
    33.794,
    49.362,
    64.547,
    79.477,
]
spann_recall = [
    80.6,
    83.5,
    85.2,
    88.7,
    90.9,
    93.8,
    95.7,
    96.6,
]


# 创建折线图
plt.figure(figsize=(10, 6))
plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
#plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
plt.plot(diskann_lacency, diskann_recall, marker='o', label='Diskann')
# plt.plot(diskann_lacency2, diskann_recall2, marker='o', label='Diskann2')
plt.plot(spann_lacency, spann_recall, marker='o', label='spann')

# 添加标题和标签
plt.title('gist 1-recall@1 vs. Latency')
plt.xlabel('Latency (ms)')
plt.ylabel('1-recall@1')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()