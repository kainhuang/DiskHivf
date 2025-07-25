import matplotlib.pyplot as plt

# DiskHivf 数据
diskhivf_lacency_100x100 = [
    781.403,
    1336.49,
    1929.5,
    2423.13,
    2975.36,
    3526.25,
    4515.92,
    6792.72,
    9202.13,
    10581.3,
    10789.2,
]
diskhivf_recall_100x100 = [
    88.046997,
    94.302002,
    96.716003,
    97.921997,
    98.593002,
    98.985001,
    99.404999,
    99.823997,
    99.939003,
    99.911003,
    99.703003,
]

diskhivf_lacency_300x300 = [
    953.276,
    1377.74,
    1853.49,
    2357.52,
    2985.35,
    3612.91,
    4332.55,
    5088.61,
    5909.95,
    6483.63,
    7796.08,
]
diskhivf_recall_300x300 = [
    93.054001,
    95.915001,
    97.536003,
    98.439003,
    99.017998,
    99.387001,
    99.633003,
    99.764999,
    99.846001,
    99.887001,
    99.931999,
]

diskhivf_lacency_333x333 = [
    441.679,
    721.961,
    1045.07,
    1397.76,
    1776.04,
    2257.59,
    2733.15,
    3356.79,
    3840.35,
    4451.57,
    4923.66,
    5918.96,
]
diskhivf_recall_333x333 = [
    85.786003,
    91.889999,
    95.045998,
    96.857002,
    97.917000,
    98.667999,
    99.069000,
    99.378998,
    99.560997,
    99.676003,
    99.754997,
    99.827003,
]


diskhivf_lacency_500x200 = [
    820.735,
    1211.22,
    1634.86,
    2104.98,
    2653.62,
    3247.22,
    3880.91,
    4573.16,
    5292.01,
    5846.19,
    7010.03,
]
diskhivf_recall_500x200 = [
    91.386002,
    94.684998,
    96.585999,
    97.742996,
    98.568001,
    99.017998,
    99.334999,
    99.536003,
    99.685997,
    99.764999,
    99.849998,
]

diskhivf_lacency_800x125 = [
    471.503,
    783.857,
    1144.95,
    1533.72,
    1983.55,
    2494.62,
    3008.44,
    3603.08,
    4227.68,
    4960.9,
    5428.23,
    6680.72,
]
diskhivf_recall_800x125 = [
    87.198997,
    92.779999,
    95.773003,
    97.424004,
    98.334000,
    98.935997,
    99.297997,
    99.542000,
    99.675003,
    99.755997,
    99.795998,
    99.857002,
]

diskhivf_lacency_500x500 = [
    685.089,
    972.869,
    1250.01,
    1627.07,
    1992.35,
    2428.12,
    2863.2,
    3337.41,
    3866.39,
    4220.93,
    5042.86,
]
diskhivf_recall_500x500 = [
    86.515999,
    90.901001,
    93.837997,
    95.616997,
    96.919998,
    97.742996,
    98.387001,
    98.764000,
    99.128998,
    99.278999,
    99.533997,
]


diskhivf_lacency_1000x1000 = [
    530.41,
    768.769,
    936.068,
    1236.09,
    1437.69,
    1781.86,
    2022.45,
    2434.94,
    2705.73,
    3060.67,
    3485.47,
]
diskhivf_recall_1000x1000 = [
    74.508003,
    80.607002,
    85.171997,
    88.417999,
    91.098999,
    92.969002,
    94.485001,
    95.516998,
    96.406998,
    96.907997,
    97.649002,
]


# 创建折线图
plt.figure(figsize=(10, 6))

plt.plot(diskhivf_lacency_100x100, diskhivf_recall_100x100, marker='o', label='DiskHivf-100x100-centers')
plt.plot(diskhivf_lacency_300x300, diskhivf_recall_300x300, marker='o', label='DiskHivf-300x300-centers')
#plt.plot(diskhivf_lacency_333x333, diskhivf_recall_333x333, marker='o', label='DiskHivf-333x333-centers')
plt.plot(diskhivf_lacency_500x200, diskhivf_recall_500x200, marker='o', label='DiskHivf-500x200-centers')
#plt.plot(diskhivf_lacency_800x125, diskhivf_recall_800x125, marker='o', label='DiskHivf-800x125-centers')
plt.plot(diskhivf_lacency_500x500, diskhivf_recall_500x500, marker='o', label='DiskHivf-500x500-centers')
plt.plot(diskhivf_lacency_1000x1000, diskhivf_recall_1000x1000, marker='o', label='DiskHivf-1000x1000-centers')

# 添加标题和标签
plt.title('sift1M 10-recall@10 vs. Lacency')
plt.xlabel('Lacency')
plt.ylabel('10-recall@10')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()