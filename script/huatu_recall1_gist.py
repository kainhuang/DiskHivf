import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_lacency = [
    6799.07,
    11183.3,
    14797,
    18065.5,
    21652.1,
    26830.8,
    32489.9,
    37548.2,
    42773.1,
    49330.1,
]
diskhivf_recall = [
    88.7000,
    93.5000,
    94.5000,
    94.9000,
    95.3000,
    95.9000,
    96.7000,
    97.8000,
    98.0000,
    98.3000,
]

# Diskann 数据
diskann_lacency = [
    6245.67,
    6941.42,
    10004.59,
    13170.83,
    16166.05,
    19304.59,
    22848.32,
    25366.12,
    31543.04,
    37355.37,
    44434.18,
    49618.37,
    62587.83,
]
diskann_recall = [
    80.6,
    82.3,
    87.5,
    90,
    92.3,
    93.6,
    94.5,
    95.1,
    96.4,
    97.6,
    98.2,
    98.9,
    99.5,
]

# spann 数据
spann_lacency = [
    10686,
    14076,
    17445,
    25697,
    33794,
    49362,
    64547,
    79477,
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
plt.plot(diskann_lacency, diskann_recall, marker='o', label='Diskann')
plt.plot(spann_lacency, spann_recall, marker='o', label='spann')

# 添加标题和标签
plt.title('gist 1-recall@1 vs. Lacency')
plt.xlabel('Lacency')
plt.ylabel('1-recall@1')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()