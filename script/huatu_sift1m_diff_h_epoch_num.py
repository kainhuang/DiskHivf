import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_lacency_1_epoch = [
    902.742,
    1325.73,
    1800.35,
    2324.75,
    2991.71,
    3634.79,
    4394.81,
    5196.48,
    6037.71,
    6662.88,
    8062.13,
]
diskhivf_recall_1_epoch = [
    92.330002,
    95.565002,
    97.378998,
    98.356003,
    99.042999,
    99.404999,
    99.633003,
    99.781998,
    99.862999,
    99.899002,
    99.950996,
]

# DiskHivf 数据
diskhivf_lacency_2_epoch = [
    952.205,
    1387.18,
    1864.6,
    2389.34,
    3065.89,
    3724.85,
    4491.37,
    5318.37,
    6215.82,
    6826.78,
    8260.35,
]
diskhivf_recall_2_epoch = [
    92.681000,
    95.681999,
    97.431999,
    98.375000,
    99.085999,
    99.424004,
    99.640999,
    99.772003,
    99.860001,
    99.900002,
    99.942001,
]

# DiskHivf 数据
diskhivf_lacency_5_epoch = [
    998.159,
    1433.17,
    1919.09,
    2476.69,
    3095.44,
    3757.25,
    4531.97,
    5320.23,
    6174.39,
    6748.52,
    8156.25,
]
diskhivf_recall_5_epoch = [
    93.017998,
    95.925003,
    97.539001,
    98.418999,
    99.039001,
    99.396004,
    99.614998,
    99.726997,
    99.823997,
    99.874001,
    99.918999,
]

# DiskHivf 数据
diskhivf_lacency_10_epoch = [
    1020.41,
    1480.46,
    1959.63,
    2536.88,
    3156.48,
    3827.88,
    4572.99,
    5412.12,
    6225.47,
    6815.41,
    8204.57,
]
diskhivf_recall_10_epoch = [
    93.047997,
    95.894997,
    97.491997,
    98.458000,
    99.036003,
    99.360001,
    99.600998,
    99.737999,
    99.831001,
    99.886002,
    99.931999,
]

# DiskHivf 数据
diskhivf_lacency_20_epoch = [
    1034.24,
    1475.96,
    1979.83,
    2524.27,
    3181.89,
    3887.54,
    4634.37,
    5388.05,
    6246.15,
    6851.97,
    8244.18,
]
diskhivf_recall_20_epoch = [
    93.213997,
    95.996002,
    97.527000,
    98.414001,
    99.049004,
    99.382004,
    99.602997,
    99.730003,
    99.834999,
    99.875999,
    99.931999,
]


# DiskHivf 数据
diskhivf_lacency_50_epoch = [
    1019.47,
    1458.87,
    1969.48,
    2500.9,
    3152.21,
    3799.69,
    4550.13,
    5368.47,
    6245.71,
    6842.27,
    8316.64,
]
diskhivf_recall_50_epoch = [
    93.213997,
    95.996002,
    97.527000,
    98.414001,
    99.049004,
    99.382004,
    99.602997,
    99.730003,
    99.834999,
    99.875999,
    99.931999,
]

# DiskHivf 数据
diskhivf_lacency_100_epoch = [
    1025.69,
    1472.48,
    1977.89,
    2513.38,
    3183.41,
    3849.8,
    4604.11,
    5349.48,
    6293.21,
    6961.59,
    8329.35,
]
diskhivf_recall_100_epoch = [
    93.213997,
    95.996002,
    97.527000,
    98.414001,
    99.049004,
    99.382004,
    99.602997,
    99.730003,
    99.834999,
    99.875999,
    99.931999,
]

# 创建折线图
plt.figure(figsize=(10, 6))

plt.plot(diskhivf_lacency_1_epoch, diskhivf_recall_1_epoch, marker='o', label='DiskHivf-1_epoch')
plt.plot(diskhivf_lacency_2_epoch, diskhivf_recall_2_epoch, marker='o', label='DiskHivf-2_epoch')
plt.plot(diskhivf_lacency_5_epoch, diskhivf_recall_5_epoch, marker='o', label='DiskHivf-5_epoch')
plt.plot(diskhivf_lacency_10_epoch, diskhivf_recall_10_epoch, marker='o', label='DiskHivf-10_epoch')
plt.plot(diskhivf_lacency_20_epoch, diskhivf_recall_20_epoch, marker='o', label='DiskHivf-20_epoch')
plt.plot(diskhivf_lacency_50_epoch, diskhivf_recall_50_epoch, marker='o', label='DiskHivf-50_epoch')
plt.plot(diskhivf_lacency_100_epoch, diskhivf_recall_100_epoch, marker='o', label='DiskHivf-100_epoch')

# 添加标题和标签
plt.title('sift1M 10-recall@10 vs. Lacency')
plt.xlabel('Lacency')
plt.ylabel('10-recall@10')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()