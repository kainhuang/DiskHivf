import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_lacency = [
    1613.07,
    1809.91,
    1970.97,
    2354.67,
    2538.6,
    2939.32,
    3050.12,
    3661.81,
    4249.36,
    4848.93,
    5775.05,
    6355.57,
    8348.84,
    8958.51,
    10969.2,
    12769.9,
    15942,
    19707.9,
    22379.1,
    22315.3,
    24314.1,
]
diskhivf_recall = [
    89.0935,
    89.9466,
    90.5910,
    91.4012,
    91.5424,
    92.2175,
    92.3219,
    92.9724,
    93.4635,
    93.7887,
    94.2429,
    94.4700,
    95.1206,
    95.2802,
    95.5502,
    95.7282,
    95.9860,
    96.3420,
    96.5445,
    96.6059,
    96.7164,
]

# Diskann 数据
diskann_lacency = [
    1794.52,
    2096.2,
    2307.03,
    2320.52,
    2975.43,
    3500.21,
    4627.61,
    5823.4,
    6979.65,
    8046.94,
    10502.12,
    12746.05,
    18495.18,
    24429.43,
]
diskann_recall = [
    81.13,
    83.83,
    85.45,
    86.11,
    88.28,
    89.68,
    91.24,
    92.04,
    92.58,
    92.95,
    93.47,
    93.78,
    94.36,
    94.68,
]

# spann 数据
spann_lacency = [
    1089,
    1553,
    2055,
    2556,
    3057,
    4057,
    5054,
    7544,
    10030,
    14976,
    19903,
    24808,
]
spann_recall = [
    80.7095,
    83.3794,
    85.1224,
    86.3561,
    87.2276,
    88.4674,
    89.2469,
    90.5235,
    91.2232,
    92.0273,
    92.5858,
    92.9356,
]

# starling 数据
starling_lacency = [
    2186.32,
    2418.17,
    2446.26,
    2565.01,
    2653.45,
    2907.78,
    3206.47,
    3805.49,
    4344.21,
    5764.39,
    8396.51,
    11657.42,
    14726.58,
    21317.12,
]
starling_recall = [
    80.72,
    86,
    87.8,
    88.84,
    89.23,
    90.67,
    91.41,
    92.43,
    92.96,
    93.62,
    94.18,
    94.59,
    94.89,
    95.2,
]

# 创建折线图
plt.figure(figsize=(10, 6))

plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
plt.plot(diskann_lacency, diskann_recall, marker='o', label='Diskann')
plt.plot(spann_lacency, spann_recall, marker='o', label='spann')
plt.plot(starling_lacency, starling_recall, marker='o', label='starling')

# 添加标题和标签
plt.title('index10M 1-recall@1 vs. Lacency')
plt.xlabel('Lacency')
plt.ylabel('1-recall@1')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()