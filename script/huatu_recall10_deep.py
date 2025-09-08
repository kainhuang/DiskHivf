import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_lacency3 = [
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
diskhivf_recall3 = [
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

# DiskHivf 数据
diskhivf_lacency2 = [
    7922.9,
    9417.04,
    10457.2,
    11237,
    12642.4,
    14268.3,
    17362.7,
    18991.7,
    20492.3,
    22227.1,
]
diskhivf_recall2 = [
    93.032997,
    94.824997,
    95.678001,
    96.107002,
    96.782997,
    97.311996,
    98.019997,
    98.250000,
    98.415001,
    98.545998,
]

# DiskHivf 数据
diskhivf_lacency = [
    5994.58,
    7198.37,
    8215.36,
    9031.53,
    10568.6,
    12171.5,
    15265.2,
    17017.8,
    18594.5,
    19869,
]
diskhivf_recall = [
    89.3000,
    91.5770,
    92.7270,
    93.2440,
    94.3650,
    95.1560,
    96.2890,
    96.6600,
    96.9910,
    97.2300,
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
    6057.4,
    6285.32,
    7251.23,
    7916.69,
    9356.02,
    11012.09,
    13137.81,
    15006.96,
    16960.92,
    18770.52,
    20475.78,
]
starling_recall = [
    81.92,
    83.94,
    86.99,
    89.14,
    92.07,
    93.9,
    95.38,
    96.33,
    96.99,
    97.48,
    97.88,
]

# 创建折线图
plt.figure(figsize=(10, 6))
plt.plot(diskhivf_lacency3, diskhivf_recall3, marker='o', label='DiskHivf')
#plt.plot(diskhivf_lacency2, diskhivf_recall2, marker='o', label='DiskHivf2')
#plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
plt.plot(diskann_lacency, diskann_recall, marker='o', label='Diskann')
# plt.plot(starling_lacency, starling_recall, marker='o', label='starling', color='red')

# 添加标题和标签
plt.title('bigann1B 10-recall@10 vs. Lacency')
plt.xlabel('Lacency')
plt.ylabel('10-recall@10')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()