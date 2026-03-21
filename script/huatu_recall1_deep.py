import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_latency = [
    4.00831,
    4.4636,
    5.36176,
    6.29157,
    7.18271,
    7.91675,
    8.56672,
    9.7265,
    11.1797,
    13.4605,
    15.3197,
    17.5397,

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
diskann_latency = [
    5.52025,
    6.38345,
    7.09514,
    7.5489,
    8.13414,
    8.68018,
    9.52464,
    9.87552,
    10.95463,
    13.79441,
    17.47139,
    21.60921,
    23.83264,
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
starling_latency = [
    17.89819,
    18.0493,
    19.60659,
    21.13715,
    22.66796,
    24.18718,
    25.71163,
    27.25909,
    30.32935,
    33.42743,
    36.5386,
    39.6515,
    42.78834,
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
plt.plot(diskhivf_latency, diskhivf_recall, marker='o', label='DiskHivf')
#plt.plot(diskhivf_latency2, diskhivf_recall2, marker='o', label='DiskHivf2')
#plt.plot(diskhivf_latency, diskhivf_recall, marker='o', label='DiskHivf')
plt.plot(diskann_latency, diskann_recall, marker='o', label='Diskann')
plt.plot(starling_latency, starling_recall, marker='o', label='starling', color='red')

# 添加标题和标签
plt.title('deep1b 1-recall@1 vs. Latency')
plt.xlabel('Latency (ms)')
plt.ylabel('1-recall@1')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()