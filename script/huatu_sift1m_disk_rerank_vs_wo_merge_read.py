import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_lacency = [
    987.213,
    1423.67,
    1948.92,
    2458.4,
    3109.99,
    3747.82,
    4540.21,
    5332.55,
    6196.38,
    6833.12,
    8125.71, 
]
diskhivf_recall = [
    93.0600,
    95.8840,
    97.4470,
    98.3520,
    99.0540,
    99.3980,
    99.6380,
    99.7710,
    99.8600,
    99.8990,
    99.9370,
]

# DiskHivf 数据
diskhivf_wo_merge_read_lacency = [
    1235.98,
    1845.85,
    2573.32,
    3361.2,
    4383.4,
    5392.96,
    6624.2,
    7905.94,
    9387.82,
    10340.9,
    12794.8,
]
diskhivf_wo_merge_read_recall = [
    92.921997,
    95.760002,
    97.414001,
    98.358002,
    98.996002,
    99.362999,
    99.625000,
    99.750000,
    99.843002,
    99.891998,
    99.938004,
]

# 创建折线图
plt.figure(figsize=(10, 6))

plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
plt.plot(diskhivf_wo_merge_read_lacency, diskhivf_wo_merge_read_recall, marker='o', label='DiskHivf-wo_merge_read')


# 添加标题和标签
plt.title('sift1M 10-recall@10 vs. Lacency')
plt.xlabel('Lacency')
plt.ylabel('10-recall@10')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()