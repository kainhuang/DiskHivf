import matplotlib.pyplot as plt

# DiskHivf 数据 (单位: 毫秒)
diskhivf_latency = [
    4.67243,
    5.62428,
    7.28138,
    9.3876,
    11.2459,
    13.1826,
    14.0249,
    16.2248,
    18.7072,
    23.6548,
]
diskhivf_recall = [
    83.789001,
    87.514999,
    91.324997,
    94.759003,
    95.978996,
    96.510002,
    96.772003,
    97.408997,
    97.870003,
    98.400002,
]

# Diskann 数据 (单位: 毫秒)
diskann_latency = [
    6.89459,
    8.17207,
    9.25769,
    11.62514,
    13.17463,
    13.94541,
    17.161,
    20.75029,

]
diskann_recall = [
    80.67,
    83.96,
    86.33,
    88.99,
    90.79,
    91.74,
    93.41,
    94.53,
]


# starling 数据 (单位: 毫秒)
starling_latency = [
    6.0574,
    6.28532,
    7.25123,
    7.91669,
    9.35602,
    11.01209,
    13.13781,
    15.00696,
    16.96092,
    18.77052,
    20.47578,
]
starling_recall = [
    81.92,
    83.94,
    86.99,
    89.14,
    92.07,
    93.9,
    95.38,
    96.33,
    96.99,
    97.48,
    97.88,
]

# SPANN 数据 (单位: 毫秒)
spann_latency = [
    3.991,
    4.989,
    5.987,
    6.984,
    7.981,
    8.979,
    9.975,
    10.973,
    11.970,
    13.965,
    15.958,
    17.952,
    19.945,
    21.938,
    23.931,
    25.924,
]

spann_recall = [
    80.1448,
    82.8416,
    84.8447,
    86.4368,
    87.7429,
    88.7969,
    89.6819,
    90.418,
    91.037,
    92.1951,
    93.0692,
    93.7894,
    94.3394,
    94.8025,
    95.2355,
    95.6265,
]

# 创建折线图
plt.figure(figsize=(10, 6))
plt.plot(diskhivf_latency, diskhivf_recall, marker='o', label='DiskHivf')
#plt.plot(diskhivf_latency2, diskhivf_recall2, marker='o', label='DiskHivf2')
#plt.plot(diskhivf_latency, diskhivf_recall, marker='o', label='DiskHivf')
plt.plot(diskann_latency, diskann_recall, marker='o', label='Diskann')
plt.plot(starling_latency, starling_recall, marker='o', label='starling', color='red')
plt.plot(spann_latency, spann_recall, marker='o', label='SPANN')

# 添加标题和标签
plt.title('bigann1B 10-recall@10 vs. Latency')
plt.xlabel('Latency (ms)')
plt.ylabel('10-recall@10')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()