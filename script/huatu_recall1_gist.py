import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_lacency = [
    7215.78,
    10498.4,
    17337.4,
    23822.2,
    33856.3,
    49276.6,
    65369.5,
    77807,
]
diskhivf_recall = [
    84.500000,
    90.199997,
    93.900002,
    95.699997,
    97.599998,
    98.900002,
    99.300003,
    99.400002,
]



# Diskann 数据
diskann_lacency = [
    9109.42,
    11246.08,
    13373.37,
    15537.45,
    17653.65,
    19812.22,
    21918.56,
    24126.96,
    35064.65,
    46217.3,
    57408.82,
    68670.06,
    80037.94,
    91199.13,
]
diskann_recall = [
    80.6,
    83.9,
    87.4,
    89.4,
    91.7,
    92.6,
    93.7,
    94.5,
    96.8,
    97.4,
    97.8,
    98.4,
    98.9,
    99.2,
]

# spann 数据
spann_lacency = [
    10686,
    14076,
    17445,
    25697,
    33794,
    49362,
    64547,
    79477,
]
spann_recall = [
    80.6,
    83.5,
    85.2,
    88.7,
    90.9,
    93.8,
    95.7,
    96.6,
]


# 创建折线图
plt.figure(figsize=(10, 6))
plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
#plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
plt.plot(diskann_lacency, diskann_recall, marker='o', label='Diskann')
# plt.plot(diskann_lacency2, diskann_recall2, marker='o', label='Diskann2')
plt.plot(spann_lacency, spann_recall, marker='o', label='spann')

# 添加标题和标签
plt.title('gist 1-recall@1 vs. Lacency')
plt.xlabel('Lacency')
plt.ylabel('1-recall@1')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()