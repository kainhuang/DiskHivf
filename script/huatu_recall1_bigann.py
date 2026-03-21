import matplotlib.pyplot as plt

# DiskHivf 数据 (单位：毫秒)
diskhivf_lacency = [
    4.57665,
    5.45209,
    6.29779,
    8.35375,
    9.74864,
    10.9053,
    11.9181,
    13.8578,
    15.6792,
    19.7422,
    21.7901,

]
diskhivf_recall = [
    88.589996,
    91.599998,
    94.269997,
    96.379997,
    97.070000,
    97.349998,
    97.540001,
    98.050003,
    98.519997,
    98.989998,
    99.050003,
]

# Diskann 数据 (单位：毫秒)
diskann_lacency = [
    4.87171,
    5.45875,
    6.14985,
    6.72555,
    7.49507,
    7.93895,
    9.48826,
    10.6052,
    12.05032,
    13.20636,
    14.47985,
    17.58026,
    20.75364,
]
diskann_recall = [
    82.11,
    84.57,
    86.45,
    88,
    89.04,
    90.08,
    91.75,
    92.76,
    93.7,
    94.52,
    95.2,
    96.29,
    96.99,
]


# starling 数据 (单位：毫秒)
starling_lacency = [
    4.79315,
    5.13033,
    5.53348,
    5.90202,
    6.24753,
    6.58098,
    6.9763,
    7.67799,
    8.36518,
    9.04191,
    9.85684,
    10.65595,
    12.22121,
    13.97192,
    16.09572,
    18.10157,
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


# spann 数据 (单位：毫秒)
spann_lacency = [
    2.993,
    3.493,
    3.991,
    4.989,
    5.986,
    7.981,
    9.976,
    11.970,
    13.964,
    19.945,
]

spann_recall = [
    83.12,
    84.8,
    86.26,
    88.3,
    89.86,
    92.23,
    93.76,
    94.71,
    95.47,
    96.96,
]


# 创建折线图
plt.figure(figsize=(10, 6))
plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
#plt.plot(diskhivf_lacency2, diskhivf_recall2, marker='o', label='DiskHivf2')
#plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
plt.plot(diskann_lacency, diskann_recall, marker='o', label='Diskann')
plt.plot(starling_lacency, starling_recall, marker='o', label='starling', color='red')
plt.plot(spann_lacency, spann_recall, marker='o', label='SPANN')

# 添加标题和标签
plt.title('bigann1B 1-recall@1 vs. Latency')
plt.xlabel('Latency (ms)')
plt.ylabel('1-recall@1')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()