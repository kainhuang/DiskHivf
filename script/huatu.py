import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_lacency = [831.035, 1234.05, 1694.9, 2226.13, 2791.52, 3446.43, 4113.26, 4856.56, 5617.05, 6440.51, 7326.6]
diskhivf_recall = [92.6810, 95.7580, 97.4440, 98.4010, 98.9740, 99.3050, 99.5310, 99.7010, 99.7880, 99.8300, 99.8730]

# Diskann 数据
diskann_lacency = [1735.22, 2836.16, 3869.84, 4926.47, 5966.31, 11714.85]
diskann_recall = [88.32, 94.77, 97.21, 98.29, 98.82, 99.73]

# spann 数据
spann_lacency = [2037, 2666, 3043, 4048]
spann_recall = [93.9973, 95.8025, 96.5187, 97.7509]

# starling 数据
starling_lacency = [2001.47, 2235.64, 2476.27, 2660.18, 2911.77, 3451.26, 4024.48, 6703.77]
starling_recall = [96.21, 97.69, 98.51, 99.01, 99.29, 99.6, 99.78, 99.97]

# 创建折线图
plt.figure(figsize=(10, 6))

plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
plt.plot(diskann_lacency, diskann_recall, marker='o', label='Diskann')
plt.plot(spann_lacency, spann_recall, marker='o', label='spann')
plt.plot(starling_lacency, starling_recall, marker='o', label='starling')

# 添加标题和标签
plt.title('10-recall@10 vs. Lacency')
plt.xlabel('Lacency')
plt.ylabel('10-recall@10')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()