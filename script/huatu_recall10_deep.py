import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_lacency = [
    4147.02,
    4691.09,
    5577.99,
    7548.45,
    8715.49,
    9791.35,
    11170.9,
    13754.1,
    16754.8,
    19134,
    21936,
]
diskhivf_recall = [
    68.711998,
    73.757004,
    79.547997,
    86.028999,
    88.848999,
    90.338997,
    91.196999,
    92.550003,
    93.549004,
    94.982002,
    95.498001,
]

# Diskann 数据
diskann_lacency = [
    4227.89,
    5305.72,
    6421.65,
    7332.24,
    8899.6,
    11089.17,
    13042.74,
    15745.73,
    20490.54,
    22453.62,
    25390.27,

]
diskann_recall = [
    62.61,
    70.37,
    75.27,
    78.76,
    81.44,
    85.02,
    87.55,
    89.73,
    91.21,
    92.3,
    93.19,
]


# starling 数据
starling_lacency = [
    18810.18,
    21131.59,
    22656.48,
    24180.94,
    25704.04,
    27252.42,
    30323.36,
    33422.99,
    36531.91,
    39651.74,
]
starling_recall = [
    72.65,
    75.34,
    77.42,
    79.13,
    80.6,
    81.94,
    83.94,
    85.58,
    86.9,
    87.96,
]

# 创建折线图
plt.figure(figsize=(10, 6))
plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
#plt.plot(diskhivf_lacency2, diskhivf_recall2, marker='o', label='DiskHivf2')
#plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
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