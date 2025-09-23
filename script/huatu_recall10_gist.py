import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_lacency = [
    7205.21,
    10453,
    17133.6,
    23597.6,
    33849.2,
    48873.6,
    65337.7,
    77869.3,
]
diskhivf_recall = [
    79.169998,
    84.930000,
    90.870003,
    93.910004,
    96.160004,
    97.769997,
    98.389999,
    98.730003,
]

# Diskann 数据
diskann_lacency = [
    15544.12,
    19804.34,
    24126.04,
    28431.85,
    37298.79,
    46246.24,
    57415.8,
    68670.4,
    79931.77,
    91212.76,
]
diskann_recall = [
    81.05,
    85.41,
    88.41,
    90.65,
    93.35,
    94.86,
    96.16,
    97.16,
    97.73,
    98.15,
]

# spann 数据
spann_lacency = [
    19131,
    20748,
    27328,
    33801,
    41681,
    49359,
    56952,
    64540,
    79485,
]
spann_recall = [
    80.51,
    81.5101,
    84.2301,
    86.6301,
    88.6201,
    90.5801,
    91.5801,
    92.5901,
    94.2502,
]


# 创建折线图
plt.figure(figsize=(10, 6))

plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
# plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
plt.plot(diskann_lacency, diskann_recall, marker='o', label='Diskann')
#plt.plot(diskann_lacency2, diskann_recall2, marker='o', label='Diskann2')
plt.plot(spann_lacency, spann_recall, marker='o', label='spann')

# 添加标题和标签
plt.title('gist 10-recall@10 vs. Lacency')
plt.xlabel('Lacency')
plt.ylabel('10-recall@10')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()