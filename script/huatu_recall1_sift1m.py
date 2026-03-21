import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_latency = [
    0.422471,
    0.749599,
    1.05386,
    1.35133,
    2.04264,
    2.83279,
    3.73326,
    4.86155,
    5.87176,
]
diskhivf_recall = [
    89.849998,
    94.790001,
    96.449997,
    97.480003,
    98.709999,
    99.239998,
    99.550003,
    99.760002,
    99.820000,

]

# Diskann 数据
diskann_latency = [
    1.08336,
    1.19443,
    1.29675,
    1.3855,
    1.4938,
    1.59942,
    1.71224,
    1.94023,
    2.12904,
    2.3723,
    2.56141,
    2.74953,
    3.41668,
    3.90004,
    4.51712,
    5.06978,
    6.212,
]
diskann_recall = [
    82.27,
    85.19,
    87.52,
    88.92,
    90.34,
    91.5,
    92.59,
    93.99,
    95.03,
    95.71,
    96.42,
    96.82,
    97.74,
    98.4,
    98.82,
    99.07,
    99.34,
]

# spann 数据
spann_latency = [
    0.638,
    0.688,
    0.703,
    0.744,
    0.884,
    1.002,
    1.164,
    1.586,
    1.978,
    2.955,
    3.927,
    5.861,
]
spann_recall = [
    80.64,
    81.77,
    83.01,
    85.5,
    88.59,
    90.64,
    92.06,
    94.17,
    95.64,
    97.37,
    98.23,
    99.03,
]

# starling 数据
starling_latency = [
    1.55968,
    1.65098,
    1.75475,
    1.86293,
    1.87914,
    1.8914,
    1.94838,
    1.98806,
    2.02438,
    2.25087,
    2.47131,
]
starling_recall = [
    82.55,
    90.27,
    94.22,
    95.77,
    96.78,
    97.41,
    97.94,
    98.33,
    98.58,
    99.17,
    99.5,
]

# 创建折线图
plt.figure(figsize=(10, 6))
plt.plot(diskhivf_latency, diskhivf_recall, marker='o', label='DiskHivf')
#plt.plot(diskhivf_latency, diskhivf_recall, marker='o', label='DiskHivf')
plt.plot(diskann_latency, diskann_recall, marker='o', label='Diskann')
plt.plot(spann_latency, spann_recall, marker='o', label='spann')
plt.plot(starling_latency, starling_recall, marker='o', label='starling')

# 添加标题和标签
plt.title('sift1M 1-recall@1 vs. Latency')
plt.xlabel('Latency (ms)')
plt.ylabel('1-recall@1')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()