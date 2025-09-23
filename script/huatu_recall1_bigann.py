import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_lacency = [
    4576.65,
    5452.09,
    6297.79,
    8353.75,
    9748.64,
    10905.3,
    11918.1,
    13857.8,
    15679.2,
    19742.2,
    21790.1,

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

# Diskann 数据
diskann_lacency = [
    4871.71,
    5458.75,
    6149.85,
    6725.55,
    7495.07,
    7938.95,
    9488.26,
    10605.2,
    12050.32,
    13206.36,
    14479.85,
    17580.26,
    20753.64,
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


# spann 数据
spann_lacency = [
    2993,
    3493,
    3991,
    4989,
    5986,
    7981,
    9976,
    11970,
    13964,
    19945,
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
plt.title('bigann1B 1-recall@1 vs. Lacency')
plt.xlabel('Lacency')
plt.ylabel('1-recall@1')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()