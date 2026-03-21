import matplotlib.pyplot as plt

# DiskHivf 数据 (单位: 毫秒)
diskhivf_lacency = [
    0.434579,
    0.755794,
    1.0641,
    1.3811,
    2.04308,
    2.87964,
    3.71196,
    4.810,
    5.92423,
]
diskhivf_recall = [
    83.361000,
    90.950996,
    93.950996,
    95.436996,
    97.447998,
    98.439003,
    99.033997,
    99.375999,
    99.570000,
]

# Diskann 数据 (单位: 毫秒)
diskann_lacency = [
    1.71664,
    1.95365,
    2.1355,
    2.34331,
    2.58927,
    2.78505,
    2.95647,
    3.20075,
    3.42737,
    3.6278,
    3.87892,
    5.03087,
    6.11716,
    7.30783,
]
diskann_recall = [
    80.49,
    85.75,
    89.07,
    91.27,
    92.82,
    93.93,
    94.83,
    95.54,
    96.07,
    96.52,
    96.92,
    98.15,
    98.75,
    99.12,
]

# spann 数据 (单位: 毫秒)
spann_lacency = [
    0.869,
    1.017,
    1.148,
    1.389,
    1.586,
    2.369,
    3.149,
    3.927,
    4.702,
    5.475,
]
spann_recall = [
    82.0275,
    84.8285,
    86.8976,
    88.4989,
    89.8199,
    93.2702,
    95.2535,
    96.4527,
    97.2699,
    97.8179,
]

# starling 数据 (单位: 毫秒)
starling_lacency = [
    1.9366,
    2.21618,
    2.44302,
    2.72525,
    3.05022,
    3.22576,
    3.4127,
    3.84125,
    3.9611,
    4.19057,
    4.48244,
    4.79951,
    5.05492,
    5.37486,
    6.63497,
]
starling_recall = [
    80.64,
    88.85,
    93.17,
    95.56,
    97.05,
    97.95,
    98.54,
    98.96,
    99.22,
    99.4,
    99.54,
    99.64,
    99.72,
    99.77,
    99.89,
]

# 创建折线图
plt.figure(figsize=(10, 6))
plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
#plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
plt.plot(diskann_lacency, diskann_recall, marker='o', label='Diskann')
plt.plot(spann_lacency, spann_recall, marker='o', label='spann')
plt.plot(starling_lacency, starling_recall, marker='o', label='starling')

# 添加标题和标签
plt.title('sift1M 10-recall@10 vs. Lacency')
plt.xlabel('Latency (ms)')
plt.ylabel('10-recall@10')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()