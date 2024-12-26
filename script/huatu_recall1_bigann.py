import matplotlib.pyplot as plt

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

# 创建折线图
plt.figure(figsize=(10, 6))

plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
plt.plot(diskann_lacency, diskann_recall, marker='o', label='Diskann')
plt.plot(starling_lacency, starling_recall, marker='o', label='starling', color='red')

# 添加标题和标签
plt.title('bigann1B 1-recall@1 vs. Lacency')
plt.xlabel('Lacency')
plt.ylabel('1-recall@1')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()