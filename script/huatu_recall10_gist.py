import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_lacency = [
    7205.23,
    12589.8,
    16746.2,
    20930.4,
    25061.7,
    31798.5,
    38617.2,
    43513.5,
    51159.6,
    57551.4,
]
diskhivf_recall = [
    84.1500,
    90.8500,
    92.9800,
    93.9800,
    94.7000,
    95.9600,
    96.7800,
    97.5600,
    97.8500,
    98.1100,
]

# Diskann 数据
diskann_lacency = [
    16575.64,
    19752.41,
    22522.41,
    25150.52,
    28556.18,
    31697.87,
    34380.03,
    40506.71,
    43947.08,
    47286.49,
    49344.21,
    53276.38,
    55332.44,
    58763.88,
    62602.97,
    65674.87,
    68209.4,
]
diskann_recall = [
    81.72,
    84.63,
    86.8,
    88.58,
    90.11,
    91.19,
    92.08,
    93.62,
    94.11,
    94.59,
    95.07,
    95.64,
    95.87,
    96.17,
    96.44,
    96.68,
    96.98,

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
plt.plot(diskann_lacency, diskann_recall, marker='o', label='Diskann')
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