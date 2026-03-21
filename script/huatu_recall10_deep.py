import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_lacency = [
    4.14702,
    4.69109,
    5.57799,
    7.54845,
    8.71549,
    9.79135,
    11.1709,
    13.7541,
    16.7548,
    19.134,
    21.936,
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
    4.22789,
    5.30572,
    6.42165,
    7.33224,
    8.8996,
    11.08917,
    13.04274,
    15.74573,
    20.49054,
    22.45362,
    25.39027,

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
    18.81018,
    21.13159,
    22.65648,
    24.18094,
    25.70404,
    27.25242,
    30.32336,
    33.42299,
    36.53191,
    39.65174,
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
plt.title('deep1B 10-recall@10 vs. Latency')
plt.xlabel('Latency (ms)')
plt.ylabel('10-recall@10')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()