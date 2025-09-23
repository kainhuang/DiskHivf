import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_lacency = [
    4008.31,
    4463.6,
    5361.76,
    6291.57,
    7182.71,
    7916.75,
    8566.72,
    9726.5,
    11179.7,
    13460.5,
    15319.7,
    17539.7,

]
diskhivf_recall = [
    77.349998,
    81.320000,
    85.769997,
    89.980003,
    91.720001,
    92.589996,
    93.040001,
    94.190002,
    94.980003,
    96.269997,
    96.779999,
    97.129997,
]

# Diskann 数据
diskann_lacency = [
    5520.25,
    6383.45,
    7095.14,
    7548.9,
    8134.14,
    8680.18,
    9524.64,
    9875.52,
    10954.63,
    13794.41,
    17471.39,
    21609.21,
    23832.64,
]
diskann_recall = [
    79,
    81.04,
    82.68,
    84.03,
    85.24,
    86.29,
    87.34,
    88.17,
    89.26,
    91.46,
    93.14,
    94.22,
    94.78,
]


# starling 数据
starling_lacency = [
    17898.19,
    18049.3,
    19606.59,
    21137.15,
    22667.96,
    24187.18,
    25711.63,
    27259.09,
    30329.35,
    33427.43,
    36538.6,
    39651.5,
    42788.34,
]
starling_recall = [
    74.21,
    74.53,
    77.47,
    79.74,
    81.64,
    83.21,
    84.49,
    85.65,
    87.33,
    88.65,
    89.85,
    90.72,
    91.49,
]

# 创建折线图
plt.figure(figsize=(10, 6))
plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
#plt.plot(diskhivf_lacency2, diskhivf_recall2, marker='o', label='DiskHivf2')
#plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
plt.plot(diskann_lacency, diskann_recall, marker='o', label='Diskann')
plt.plot(starling_lacency, starling_recall, marker='o', label='starling', color='red')

# 添加标题和标签
plt.title('deep1b 1-recall@1 vs. Lacency')
plt.xlabel('Lacency')
plt.ylabel('1-recall@1')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()