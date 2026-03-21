import matplotlib.pyplot as plt

# DiskHivf 数据 (单位: 毫秒)
diskhivf_lacency = [
    7.20521,
    10.453,
    17.1336,
    23.5976,
    33.8492,
    48.8736,
    65.3377,
    77.8693,
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

# Diskann 数据 (单位: 毫秒)
diskann_lacency = [
    15.54412,
    19.80434,
    24.12604,
    28.43185,
    37.29879,
    46.24624,
    57.4158,
    68.6704,
    79.93177,
    91.21276,
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

# spann 数据 (单位: 毫秒)
spann_lacency = [
    19.131,
    20.748,
    27.328,
    33.801,
    41.681,
    49.359,
    56.952,
    64.540,
    79.485,
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
plt.xlabel('Latency (ms)')
plt.ylabel('10-recall@10')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()