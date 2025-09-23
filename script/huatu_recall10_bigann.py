import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_lacency = [
    4672.43,
    5624.28,
    7281.38,
    9387.6,
    11245.9,
    13182.6,
    14024.9,
    16224.8,
    18707.2,
    23654.8,
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

# Diskann 数据
diskann_lacency = [
    6894.59,
    8172.07,
    9257.69,
    11625.14,
    13174.63,
    13945.41,
    17161,
    20750.29,

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


# starling 数据
starling_lacency = [
    6057.4,
    6285.32,
    7251.23,
    7916.69,
    9356.02,
    11012.09,
    13137.81,
    15006.96,
    16960.92,
    18770.52,
    20475.78,
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

spann_lacency = [
    3991,
    4989,
    5987,
    6984,
    7981,
    8979,
    9975,
    10973,
    11970,
    13965,
    15958,
    17952,
    19945,
    21938,
    23931,
    25924,
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
plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
#plt.plot(diskhivf_lacency2, diskhivf_recall2, marker='o', label='DiskHivf2')
#plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
plt.plot(diskann_lacency, diskann_recall, marker='o', label='Diskann')
plt.plot(starling_lacency, starling_recall, marker='o', label='starling', color='red')
plt.plot(spann_lacency, spann_recall, marker='o', label='SPANN')

# 添加标题和标签
plt.title('bigann1B 10-recall@10 vs. Lacency')
plt.xlabel('Lacency')
plt.ylabel('10-recall@10')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()