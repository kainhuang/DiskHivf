import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_lacency = [
    6340.51, 
    7860.95, 
    8822.83, 
    9820.2, 
    11431, 
    13187.2, 
    16647.3, 
    18491.4, 
    19050.3, 
    20541.6, 
]
diskhivf_recall = [
    89.269,
    91.548,
    92.698,
    93.217,
    94.34,
    95.131,
    96.267,
    96.638,
    96.97,
    97.21,
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
plt.title('10-recall@10 vs. Lacency')
plt.xlabel('Lacency')
plt.ylabel('10-recall@10')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()