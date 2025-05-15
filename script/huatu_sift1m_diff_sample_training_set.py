import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_lacency_sample_1 = [
    659.544,
    1178.16,
    1730.91,
    2366.77,
    3002.71,
    3869.52,
    4775.05,
    5625.12,
    6698.16,
    7613.07,
    8021.59,
    8480.74,
]
diskhivf_recall_sample_1 = [
    89.775002,
    95.080002,
    97.389000,
    98.601997,
    99.177002,
    99.504997,
    99.704002,
    99.814003,
    99.865997,
    99.903000,
    99.917000,
    99.912003,
]

diskhivf_lacency_sample_5 = [
    485.411,
    806.739,
    1177.45,
    1571.84,
    2031.35,
    2559.73,
    3126.99,
    3723.3,
    4380.5,
    5100.9,
    5529.49,
    6654.27,
]
diskhivf_recall_sample_5 = [
    87.375000,
    93.047997,
    95.904999,
    97.514999,
    98.376999,
    98.958000,
    99.296997,
    99.542999,
    99.690002,
    99.781998,
    99.825996,
    99.869003,
]

diskhivf_lacency_sample_10 = [
    475.424,
    798.315,
    1140.27,
    1526.37,
    1933.81,
    2446.96,
    2997.65,
    3546.88,
    4289.29,
    4894.27,
    5449.11,
    6560.97,
]
diskhivf_recall_sample_10 = [
    87.490997,
    92.945000,
    95.883003,
    97.463997,
    98.356003,
    98.989998,
    99.304001,
    99.542999,
    99.669998,
    99.771004,
    99.816002,
    99.875999,
]


diskhivf_lacency_sample_20 = [
    471.695,
    786.288,
    1114.96,
    1501.7,
    1958.76,
    2447.86,
    2939.96,
    3560.54,
    4176.98,
    4968.63,
    5499.34,
    6572.24,
]
diskhivf_recall_sample_20 = [
    87.736000,
    93.194000,
    95.987999,
    97.516998,
    98.376999,
    98.994003,
    99.320000,
    99.542000,
    99.681000,
    99.765999,
    99.811996,
    99.860001,
]

diskhivf_lacency_sample_50 = [
    468.084,
    775.919,
    1133.28,
    1498.95,
    1917.73,
    2431.52,
    2950.29,
    3602.7,
    4128.44,
    4828.47,
    5312.8,
    6384.23,
]
diskhivf_recall_sample_50 = [
    87.603996,
    93.140999,
    95.955002,
    97.482002,
    98.346001,
    98.960999,
    99.280998,
    99.519997,
    99.663002,
    99.758003,
    99.801003,
    99.865997,
]

diskhivf_lacency_sample_100 = [
    479.985,
    780.05,
    1119.35,
    1523.7,
    1926.91,
    2429.54,
    2920.86,
    3541.68,
    4126.86,
    4858.44,
    5420.23,
    6442.8,
]
diskhivf_recall_sample_100 = [
    87.675003,
    93.148003,
    95.910004,
    97.485001,
    98.393997,
    98.978996,
    99.319000,
    99.547997,
    99.671997,
    99.748001,
    99.796997,
    99.864998,
]


# 创建折线图
plt.figure(figsize=(10, 6))

plt.plot(diskhivf_lacency_sample_1, diskhivf_recall_sample_1, marker='o', label='%1-base data')
plt.plot(diskhivf_lacency_sample_5, diskhivf_recall_sample_5, marker='o', label='%5-base data')
plt.plot(diskhivf_lacency_sample_10, diskhivf_recall_sample_10, marker='o', label='%10-base data')
plt.plot(diskhivf_lacency_sample_20, diskhivf_recall_sample_20, marker='o', label='%20-base data')
plt.plot(diskhivf_lacency_sample_50, diskhivf_recall_sample_50, marker='o', label='%50-base data')
plt.plot(diskhivf_lacency_sample_100, diskhivf_recall_sample_100, marker='o', label='%100-base data')

#plt.plot(diskhivf_lacency_300x300, diskhivf_recall_300x300, marker='o', label='DiskHivf-300x300-centers')
#plt.plot(diskhivf_lacency_333x333, diskhivf_recall_333x333, marker='o', label='DiskHivf-333x333-centers')
#plt.plot(diskhivf_lacency_500x200, diskhivf_recall_500x200, marker='o', label='DiskHivf-500x200-centers')
#plt.plot(diskhivf_lacency_800x125, diskhivf_recall_800x125, marker='o', label='DiskHivf-800x125-centers')
#plt.plot(diskhivf_lacency_500x500, diskhivf_recall_500x500, marker='o', label='DiskHivf-500x500-centers')
#plt.plot(diskhivf_lacency_1000x1000, diskhivf_recall_1000x1000, marker='o', label='DiskHivf-1000x1000-centers')

# 添加标题和标签
plt.title('sift1M 10-recall@10 vs. Lacency')
plt.xlabel('Lacency')
plt.ylabel('10-recall@10')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()