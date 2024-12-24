import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_lacency = [
    5965.07,
    7383.79,
    8372.17,
    9269.28,
    10767.5,
    12406.6,
    15512.5,
    17139.7,
    19166.3,
    24513.5,
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
    24123.89,
    27080.77,
    30203.95,

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
    95.4,
    96.11,
    96.63,
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
    22551.59,
    24752.2,
    27173.68,
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
    98.16,
    98.39,
    98.58,
]

# 创建折线图
plt.figure(figsize=(10, 6))

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