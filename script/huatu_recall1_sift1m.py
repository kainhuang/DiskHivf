import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_lacency = [
    793.332,
    1047.54,
    1298.06,
    1544.46,
    1829.64,
    2152.77,
    2550.57,
    3022.04,
    3700.68,
    4107.12,
    5198.7,
]
diskhivf_recall = [
    95.2500,
    96.6800,
    97.5100,
    97.9800,
    98.2700,
    98.6100,
    98.8200,
    99.2500,
    99.5100,
    99.6500,
    99.8000,
]

# Diskann 数据
diskann_lacency = [
    1083.36,
    1194.43,
    1296.75,
    1385.5,
    1493.8,
    1599.42,
    1712.24,
    1940.23,
    2129.04,
    2372.3,
    2561.41,
    2749.53,
    3416.68,
    3900.04,
    4517.12,
    5069.78,
    6212,
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
spann_lacency = [
    638,
    688,
    703,
    744,
    884,
    1002,
    1164,
    1586,
    1978,
    2955,
    3927,
    5861,
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
starling_lacency = [
    1559.68,
    1650.98,
    1754.75,
    1862.93,
    1879.14,
    1891.4,
    1948.38,
    1988.06,
    2024.38,
    2250.87,
    2471.31,
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

plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
plt.plot(diskann_lacency, diskann_recall, marker='o', label='Diskann')
plt.plot(spann_lacency, spann_recall, marker='o', label='spann')
plt.plot(starling_lacency, starling_recall, marker='o', label='starling')

# 添加标题和标签
plt.title('sift1M 1-recall@1 vs. Lacency')
plt.xlabel('Lacency')
plt.ylabel('1-recall@1')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()