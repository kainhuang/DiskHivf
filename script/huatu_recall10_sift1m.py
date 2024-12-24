import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_lacency = [
    944.789,
    1296.66,
    1614.69,
    1956.9,
    2341.6,
    2703.93,
    3112.45,
    3732.85,
    4538.18,
    5041.76,
    6409.77,
]
diskhivf_recall = [
    92.2760,
    94.9230,
    96.3830,
    97.1740,
    97.7500,
    98.1000,
    98.4260,
    99.0150,
    99.4240,
    99.5780,
    99.8010,
]

# Diskann 数据
diskann_lacency = [
    1716.64,
    1953.65,
    2135.5,
    2343.31,
    2589.27,
    2785.05,
    2956.47,
    3200.75,
    3427.37,
    3627.8,
    3878.92,
    5030.87,
    6117.16,
    7307.83,
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

# spann 数据
spann_lacency = [
    869,
    1017,
    1148,
    1389,
    1586,
    2369,
    3149,
    3927,
    4702,
    5475,
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

# starling 数据
starling_lacency = [
    1936.6,
    2216.18,
    2443.02,
    2725.25,
    3050.22,
    3225.76,
    3412.7,
    3841.25,
    3961.1,
    4190.57,
    4482.44,
    4799.51,
    5054.92,
    5374.86,
    6634.97,
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
plt.plot(diskann_lacency, diskann_recall, marker='o', label='Diskann')
plt.plot(spann_lacency, spann_recall, marker='o', label='spann')
plt.plot(starling_lacency, starling_recall, marker='o', label='starling')

# 添加标题和标签
plt.title('sift1M 10-recall@10 vs. Lacency')
plt.xlabel('Lacency')
plt.ylabel('10-recall@10')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()