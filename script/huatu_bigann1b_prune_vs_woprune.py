import matplotlib.pyplot as plt

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

# DiskHivf 数据
diskhivf_wo_prune_lacency = [
    7863.97, 
    10220.3, 
    12633.4, 
    15243, 
    17163.6, 
    19033.3, 
    25026, 
    26820.4, 
    28455.7, 
]
diskhivf_wo_prune_recall = [
    89.911,
    92.559,
    94.139,
    95.086,
    95.784,
    96.311,
    97.052,
    97.3,
    97.501,
]

# 创建折线图
plt.figure(figsize=(10, 6))

plt.plot(diskhivf_lacency, diskhivf_recall, marker='o', label='DiskHivf')
plt.plot(diskhivf_wo_prune_lacency, diskhivf_wo_prune_recall, marker='o', label='DiskHivf-wo_prune')


# 添加标题和标签
plt.title('bigann1B 10-recall@10 vs. Lacency')
plt.xlabel('Lacency')
plt.ylabel('10-recall@10')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()