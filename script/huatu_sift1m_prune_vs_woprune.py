import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_lacency = [
    871.322,
    1177.65,
    1448.54,
    1753.84,
    2064.64,
    2345.52,
    2719.81,
    3314.81,
    3993.43,
    4379.26,
    5609.47,
]
diskhivf_recall = [
    92.4790,
    95.0610,
    96.4230,
    97.2190,
    97.8290,
    98.1930,
    98.4970,
    99.0170,
    99.4090,
    99.5800,
    99.7830,
]

# DiskHivf 数据
diskhivf_wo_prune_lacency = [
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
diskhivf_wo_prune_recall = [
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

# 创建折线图
plt.figure(figsize=(10, 6))

plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
plt.plot(diskhivf_wo_prune_lacency, diskhivf_wo_prune_recall, marker='o', label='DiskHivf-wo_prune')


# 添加标题和标签
plt.title('sift1M 10-recall@10 vs. Lacency')
plt.xlabel('Lacency')
plt.ylabel('10-recall@10')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()