import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_lacency3 = [
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
diskhivf_recall3 = [
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

# DiskHivf 数据
diskhivf_lacency2 = [
    5890.81,
    7442.84,
    8460.9,
    9327.11,
    9901.89,
    11283.5,
    12681.6,
    15355.2,
    16403.1,
    17742.6,
    18924,
]
diskhivf_recall2 = [
    92.900002,
    95.730003,
    96.830002,
    97.269997,
    97.410004,
    97.940002,
    98.250000,
    98.739998,
    98.919998,
    99.089996,
    99.239998,
]

# DiskHivf 数据
diskhivf_lacency = [
    4009.9,
    5514.38,
    6411.43,
    7125.62,
    7766.27,
    9087.5,
    10437.3,
    13314.3,
    14640,
    15867.6,
    17396.2,
]
diskhivf_recall = [
    88.1900,
    92.5700,
    93.9600,
    94.5300,
    94.8600,
    95.6600,
    96.4000,
    97.2000,
    97.5400,
    97.7800,
    98.0100,
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
    4793.15,
    5130.33,
    5533.48,
    5902.02,
    6247.53,
    6580.98,
    6976.3,
    7677.99,
    8365.18,
    9041.91,
    9856.84,
    10655.95,
    12221.21,
    13971.92,
    16095.72,
    18101.57,
]
starling_recall = [
    83.75,
    86.61,
    88.44,
    90.03,
    91.29,
    92.22,
    93.11,
    94.35,
    95.04,
    96.01,
    96.54,
    97.07,
    97.78,
    98.29,
    98.66,
    98.91,
]

# 创建折线图
plt.figure(figsize=(10, 6))
plt.plot(diskhivf_lacency3, diskhivf_recall3, marker='o', label='DiskHivf')
#plt.plot(diskhivf_lacency2, diskhivf_recall2, marker='o', label='DiskHivf2')
#plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
plt.plot(diskann_lacency, diskann_recall, marker='o', label='Diskann')
#plt.plot(starling_lacency, starling_recall, marker='o', label='starling', color='red')

# 添加标题和标签
plt.title('deep1b 1-recall@1 vs. Lacency')
plt.xlabel('Lacency')
plt.ylabel('1-recall@1')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()