import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_lacency2 = [
    7922.9,
    9417.04,
    10457.2,
    11237,
    12642.4,
    14268.3,
    17362.7,
    18991.7,
    20492.3,
    22227.1,
]
diskhivf_recall2 = [
    93.032997,
    94.824997,
    95.678001,
    96.107002,
    96.782997,
    97.311996,
    98.019997,
    98.250000,
    98.415001,
    98.545998,
]

# DiskHivf 数据
diskhivf_lacency = [
    5994.58,
    7198.37,
    8215.36,
    9031.53,
    10568.6,
    12171.5,
    15265.2,
    17017.8,
    18594.5,
    19869,
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

# 创建折线图
plt.figure(figsize=(10, 6))
plt.plot(diskhivf_lacency2, diskhivf_recall2, marker='o', label='DiskHivf2')
plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
plt.plot(diskann_lacency, diskann_recall, marker='o', label='Diskann')
plt.plot(starling_lacency, starling_recall, marker='o', label='starling', color='red')

# 添加标题和标签
plt.title('bigann1B 10-recall@10 vs. Lacency')
plt.xlabel('Lacency')
plt.ylabel('10-recall@10')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()