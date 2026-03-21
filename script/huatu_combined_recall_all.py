import matplotlib.pyplot as plt

# ==================== SIFT1M 数据 ====================
# SIFT1M 1-recall@1 数据
sift1m_diskhivf_latency_r1 = [0.422471, 0.749599, 1.05386, 1.35133, 2.04264, 2.83279, 3.73326, 4.86155, 5.87176]
sift1m_diskhivf_recall_r1 = [89.849998, 94.790001, 96.449997, 97.480003, 98.709999, 99.239998, 99.550003, 99.760002, 99.820000]

sift1m_diskann_latency_r1 = [1.08336, 1.19443, 1.29675, 1.3855, 1.4938, 1.59942, 1.71224, 1.94023, 2.12904, 2.3723, 2.56141, 2.74953, 3.41668, 3.90004, 4.51712, 5.06978, 6.212]
sift1m_diskann_recall_r1 = [82.27, 85.19, 87.52, 88.92, 90.34, 91.5, 92.59, 93.99, 95.03, 95.71, 96.42, 96.82, 97.74, 98.4, 98.82, 99.07, 99.34]

sift1m_spann_latency_r1 = [0.638, 0.688, 0.703, 0.744, 0.884, 1.002, 1.164, 1.586, 1.978, 2.955, 3.927, 5.861]
sift1m_spann_recall_r1 = [80.64, 81.77, 83.01, 85.5, 88.59, 90.64, 92.06, 94.17, 95.64, 97.37, 98.23, 99.03]

sift1m_starling_latency_r1 = [1.55968, 1.65098, 1.75475, 1.86293, 1.87914, 1.8914, 1.94838, 1.98806, 2.02438, 2.25087, 2.47131]
sift1m_starling_recall_r1 = [82.55, 90.27, 94.22, 95.77, 96.78, 97.41, 97.94, 98.33, 98.58, 99.17, 99.5]

# SIFT1M 10-recall@10 数据
sift1m_diskhivf_latency_r10 = [0.434579, 0.755794, 1.0641, 1.3811, 2.04308, 2.87964, 3.71196, 4.810, 5.92423]
sift1m_diskhivf_recall_r10 = [83.361000, 90.950996, 93.950996, 95.436996, 97.447998, 98.439003, 99.033997, 99.375999, 99.570000]

sift1m_diskann_latency_r10 = [1.71664, 1.95365, 2.1355, 2.34331, 2.58927, 2.78505, 2.95647, 3.20075, 3.42737, 3.6278, 3.87892, 5.03087, 6.11716, 7.30783]
sift1m_diskann_recall_r10 = [80.49, 85.75, 89.07, 91.27, 92.82, 93.93, 94.83, 95.54, 96.07, 96.52, 96.92, 98.15, 98.75, 99.12]

sift1m_spann_latency_r10 = [0.869, 1.017, 1.148, 1.389, 1.586, 2.369, 3.149, 3.927, 4.702, 5.475]
sift1m_spann_recall_r10 = [82.0275, 84.8285, 86.8976, 88.4989, 89.8199, 93.2702, 95.2535, 96.4527, 97.2699, 97.8179]

sift1m_starling_latency_r10 = [1.9366, 2.21618, 2.44302, 2.72525, 3.05022, 3.22576, 3.4127, 3.84125, 3.9611, 4.19057, 4.48244, 4.79951, 5.05492, 5.37486, 6.63497]
sift1m_starling_recall_r10 = [80.64, 88.85, 93.17, 95.56, 97.05, 97.95, 98.54, 98.96, 99.22, 99.4, 99.54, 99.64, 99.72, 99.77, 99.89]

# ==================== GIST 数据 ====================
# GIST 1-recall@1 数据
gist_diskhivf_latency_r1 = [7.21578, 10.4984, 17.3374, 23.8222, 33.8563, 49.2766, 65.3695, 77.807]
gist_diskhivf_recall_r1 = [84.500000, 90.199997, 93.900002, 95.699997, 97.599998, 98.900002, 99.300003, 99.400002]

gist_diskann_latency_r1 = [9.10942, 11.24608, 13.37337, 15.53745, 17.65365, 19.81222, 21.91856, 24.12696, 35.06465, 46.2173, 57.40882, 68.67006, 80.03794, 91.19913]
gist_diskann_recall_r1 = [80.6, 83.9, 87.4, 89.4, 91.7, 92.6, 93.7, 94.5, 96.8, 97.4, 97.8, 98.4, 98.9, 99.2]

gist_spann_latency_r1 = [10.686, 14.076, 17.445, 25.697, 33.794, 49.362, 64.547, 79.477]
gist_spann_recall_r1 = [80.6, 83.5, 85.2, 88.7, 90.9, 93.8, 95.7, 96.6]

# GIST 10-recall@10 数据
gist_diskhivf_latency_r10 = [7.20521, 10.453, 17.1336, 23.5976, 33.8492, 48.8736, 65.3377, 77.8693]
gist_diskhivf_recall_r10 = [79.169998, 84.930000, 90.870003, 93.910004, 96.160004, 97.769997, 98.389999, 98.730003]

gist_diskann_latency_r10 = [15.54412, 19.80434, 24.12604, 28.43185, 37.29879, 46.24624, 57.4158, 68.6704, 79.93177, 91.21276]
gist_diskann_recall_r10 = [81.05, 85.41, 88.41, 90.65, 93.35, 94.86, 96.16, 97.16, 97.73, 98.15]

gist_spann_latency_r10 = [19.131, 20.748, 27.328, 33.801, 41.681, 49.359, 56.952, 64.540, 79.485]
gist_spann_recall_r10 = [80.51, 81.5101, 84.2301, 86.6301, 88.6201, 90.5801, 91.5801, 92.5901, 94.2502]

# ==================== BIGANN1B 数据 ====================
# BIGANN1B 1-recall@1 数据
bigann_diskhivf_latency_r1 = [4.57665, 5.45209, 6.29779, 8.35375, 9.74864, 10.9053, 11.9181, 13.8578, 15.6792, 19.7422, 21.7901]
bigann_diskhivf_recall_r1 = [88.589996, 91.599998, 94.269997, 96.379997, 97.070000, 97.349998, 97.540001, 98.050003, 98.519997, 98.989998, 99.050003]

bigann_diskann_latency_r1 = [4.87171, 5.45875, 6.14985, 6.72555, 7.49507, 7.93895, 9.48826, 10.6052, 12.05032, 13.20636, 14.47985, 17.58026, 20.75364]
bigann_diskann_recall_r1 = [82.11, 84.57, 86.45, 88, 89.04, 90.08, 91.75, 92.76, 93.7, 94.52, 95.2, 96.29, 96.99]

bigann_starling_latency_r1 = [4.79315, 5.13033, 5.53348, 5.90202, 6.24753, 6.58098, 6.9763, 7.67799, 8.36518, 9.04191, 9.85684, 10.65595, 12.22121, 13.97192, 16.09572, 18.10157]
bigann_starling_recall_r1 = [83.75, 86.61, 88.44, 90.03, 91.29, 92.22, 93.11, 94.35, 95.04, 96.01, 96.54, 97.07, 97.78, 98.29, 98.66, 98.91]

bigann_spann_latency_r1 = [2.993, 3.493, 3.991, 4.989, 5.986, 7.981, 9.976, 11.970, 13.964, 19.945]
bigann_spann_recall_r1 = [83.12, 84.8, 86.26, 88.3, 89.86, 92.23, 93.76, 94.71, 95.47, 96.96]

# BIGANN1B 10-recall@10 数据
bigann_diskhivf_latency_r10 = [4.67243, 5.62428, 7.28138, 9.3876, 11.2459, 13.1826, 14.0249, 16.2248, 18.7072, 23.6548]
bigann_diskhivf_recall_r10 = [83.789001, 87.514999, 91.324997, 94.759003, 95.978996, 96.510002, 96.772003, 97.408997, 97.870003, 98.400002]

bigann_diskann_latency_r10 = [6.89459, 8.17207, 9.25769, 11.62514, 13.17463, 13.94541, 17.161, 20.75029]
bigann_diskann_recall_r10 = [80.67, 83.96, 86.33, 88.99, 90.79, 91.74, 93.41, 94.53]

bigann_starling_latency_r10 = [6.0574, 6.28532, 7.25123, 7.91669, 9.35602, 11.01209, 13.13781, 15.00696, 16.96092, 18.77052, 20.47578]
bigann_starling_recall_r10 = [81.92, 83.94, 86.99, 89.14, 92.07, 93.9, 95.38, 96.33, 96.99, 97.48, 97.88]

bigann_spann_latency_r10 = [3.991, 4.989, 5.987, 6.984, 7.981, 8.979, 9.975, 10.973, 11.970, 13.965, 15.958, 17.952, 19.945, 21.938, 23.931, 25.924]
bigann_spann_recall_r10 = [80.1448, 82.8416, 84.8447, 86.4368, 87.7429, 88.7969, 89.6819, 90.418, 91.037, 92.1951, 93.0692, 93.7894, 94.3394, 94.8025, 95.2355, 95.6265]

# ==================== DEEP1B 数据 ====================
# DEEP1B 1-recall@1 数据
deep_diskhivf_latency_r1 = [4.00831, 4.4636, 5.36176, 6.29157, 7.18271, 7.91675, 8.56672, 9.7265, 11.1797, 13.4605, 15.3197, 17.5397]
deep_diskhivf_recall_r1 = [77.349998, 81.320000, 85.769997, 89.980003, 91.720001, 92.589996, 93.040001, 94.190002, 94.980003, 96.269997, 96.779999, 97.129997]

deep_diskann_latency_r1 = [5.52025, 6.38345, 7.09514, 7.5489, 8.13414, 8.68018, 9.52464, 9.87552, 10.95463, 13.79441, 17.47139, 21.60921, 23.83264]
deep_diskann_recall_r1 = [79, 81.04, 82.68, 84.03, 85.24, 86.29, 87.34, 88.17, 89.26, 91.46, 93.14, 94.22, 94.78]

deep_starling_latency_r1 = [17.89819, 18.0493, 19.60659, 21.13715, 22.66796, 24.18718, 25.71163, 27.25909, 30.32935, 33.42743, 36.5386, 39.6515, 42.78834]
deep_starling_recall_r1 = [74.21, 74.53, 77.47, 79.74, 81.64, 83.21, 84.49, 85.65, 87.33, 88.65, 89.85, 90.72, 91.49]

# DEEP1B 10-recall@10 数据
deep_diskhivf_latency_r10 = [4.14702, 4.69109, 5.57799, 7.54845, 8.71549, 9.79135, 11.1709, 13.7541, 16.7548, 19.134, 21.936]
deep_diskhivf_recall_r10 = [68.711998, 73.757004, 79.547997, 86.028999, 88.848999, 90.338997, 91.196999, 92.550003, 93.549004, 94.982002, 95.498001]

deep_diskann_latency_r10 = [4.22789, 5.30572, 6.42165, 7.33224, 8.8996, 11.08917, 13.04274, 15.74573, 20.49054, 22.45362, 25.39027]
deep_diskann_recall_r10 = [62.61, 70.37, 75.27, 78.76, 81.44, 85.02, 87.55, 89.73, 91.21, 92.3, 93.19]

deep_starling_latency_r10 = [18.81018, 21.13159, 22.65648, 24.18094, 25.70404, 27.25242, 30.32336, 33.42299, 36.53191, 39.65174]
deep_starling_recall_r10 = [72.65, 75.34, 77.42, 79.13, 80.6, 81.94, 83.94, 85.58, 86.9, 87.96]

# ==================== 绘图 ====================
# 参考 data_pics_single(1).ipynb 的尺寸比例: figsize=(20, 8)，2行4列
fig, axs = plt.subplots(2, 4, figsize=(20, 8))

# 定义颜色和标记
colors = {'DiskHivf': 'C0', 'DiskANN': 'C1', 'SPANN': 'C2', 'Starling': 'C3'}
markers = {'DiskHivf': 'o', 'DiskANN': 's', 'SPANN': '^', 'Starling': 'd'}

# 定义模型列表（用于统一图例）
models = ['DiskHivf', 'DiskANN', 'SPANN', 'Starling']

# ========== 第一行: 1-recall@1 ==========
# SIFT1M 1-recall@1
axs[0, 0].plot(sift1m_diskhivf_latency_r1, sift1m_diskhivf_recall_r1, marker='o', label='DiskHivf', color=colors['DiskHivf'])
axs[0, 0].plot(sift1m_diskann_latency_r1, sift1m_diskann_recall_r1, marker='s', label='DiskANN', color=colors['DiskANN'])
axs[0, 0].plot(sift1m_spann_latency_r1, sift1m_spann_recall_r1, marker='^', label='SPANN', color=colors['SPANN'])
axs[0, 0].plot(sift1m_starling_latency_r1, sift1m_starling_recall_r1, marker='d', label='Starling', color=colors['Starling'])
axs[0, 0].set_title('SIFT1M 1-recall@1')
axs[0, 0].set_xlabel('Latency (ms)')
axs[0, 0].set_ylabel('1-recall@1')
axs[0, 0].grid(True)

# GIST 1-recall@1
axs[0, 1].plot(gist_diskhivf_latency_r1, gist_diskhivf_recall_r1, marker='o', label='DiskHivf', color=colors['DiskHivf'])
axs[0, 1].plot(gist_diskann_latency_r1, gist_diskann_recall_r1, marker='s', label='DiskANN', color=colors['DiskANN'])
axs[0, 1].plot(gist_spann_latency_r1, gist_spann_recall_r1, marker='^', label='SPANN', color=colors['SPANN'])
axs[0, 1].set_title('GIST 1-recall@1')
axs[0, 1].set_xlabel('Latency (ms)')
axs[0, 1].set_ylabel('1-recall@1')
axs[0, 1].grid(True)

# BIGANN1B 1-recall@1
axs[0, 2].plot(bigann_diskhivf_latency_r1, bigann_diskhivf_recall_r1, marker='o', label='DiskHivf', color=colors['DiskHivf'])
axs[0, 2].plot(bigann_diskann_latency_r1, bigann_diskann_recall_r1, marker='s', label='DiskANN', color=colors['DiskANN'])
axs[0, 2].plot(bigann_spann_latency_r1, bigann_spann_recall_r1, marker='^', label='SPANN', color=colors['SPANN'])
axs[0, 2].plot(bigann_starling_latency_r1, bigann_starling_recall_r1, marker='d', label='Starling', color=colors['Starling'])
axs[0, 2].set_title('BIGANN1B 1-recall@1')
axs[0, 2].set_xlabel('Latency (ms)')
axs[0, 2].set_ylabel('1-recall@1')
axs[0, 2].grid(True)

# DEEP1B 1-recall@1
axs[0, 3].plot(deep_diskhivf_latency_r1, deep_diskhivf_recall_r1, marker='o', label='DiskHivf', color=colors['DiskHivf'])
axs[0, 3].plot(deep_diskann_latency_r1, deep_diskann_recall_r1, marker='s', label='DiskANN', color=colors['DiskANN'])
axs[0, 3].plot(deep_starling_latency_r1, deep_starling_recall_r1, marker='d', label='Starling', color=colors['Starling'])
axs[0, 3].set_title('DEEP1B 1-recall@1')
axs[0, 3].set_xlabel('Latency (ms)')
axs[0, 3].set_ylabel('1-recall@1')
axs[0, 3].grid(True)

# ========== 第二行: 10-recall@10 ==========
# SIFT1M 10-recall@10
axs[1, 0].plot(sift1m_diskhivf_latency_r10, sift1m_diskhivf_recall_r10, marker='o', label='DiskHivf', color=colors['DiskHivf'])
axs[1, 0].plot(sift1m_diskann_latency_r10, sift1m_diskann_recall_r10, marker='s', label='DiskANN', color=colors['DiskANN'])
axs[1, 0].plot(sift1m_spann_latency_r10, sift1m_spann_recall_r10, marker='^', label='SPANN', color=colors['SPANN'])
axs[1, 0].plot(sift1m_starling_latency_r10, sift1m_starling_recall_r10, marker='d', label='Starling', color=colors['Starling'])
axs[1, 0].set_title('SIFT1M 10-recall@10')
axs[1, 0].set_xlabel('Latency (ms)')
axs[1, 0].set_ylabel('10-recall@10')
axs[1, 0].grid(True)

# GIST 10-recall@10
axs[1, 1].plot(gist_diskhivf_latency_r10, gist_diskhivf_recall_r10, marker='o', label='DiskHivf', color=colors['DiskHivf'])
axs[1, 1].plot(gist_diskann_latency_r10, gist_diskann_recall_r10, marker='s', label='DiskANN', color=colors['DiskANN'])
axs[1, 1].plot(gist_spann_latency_r10, gist_spann_recall_r10, marker='^', label='SPANN', color=colors['SPANN'])
axs[1, 1].set_title('GIST 10-recall@10')
axs[1, 1].set_xlabel('Latency (ms)')
axs[1, 1].set_ylabel('10-recall@10')
axs[1, 1].grid(True)

# BIGANN1B 10-recall@10
axs[1, 2].plot(bigann_diskhivf_latency_r10, bigann_diskhivf_recall_r10, marker='o', label='DiskHivf', color=colors['DiskHivf'])
axs[1, 2].plot(bigann_diskann_latency_r10, bigann_diskann_recall_r10, marker='s', label='DiskANN', color=colors['DiskANN'])
axs[1, 2].plot(bigann_spann_latency_r10, bigann_spann_recall_r10, marker='^', label='SPANN', color=colors['SPANN'])
axs[1, 2].plot(bigann_starling_latency_r10, bigann_starling_recall_r10, marker='d', label='Starling', color=colors['Starling'])
axs[1, 2].set_title('BIGANN1B 10-recall@10')
axs[1, 2].set_xlabel('Latency (ms)')
axs[1, 2].set_ylabel('10-recall@10')
axs[1, 2].grid(True)

# DEEP1B 10-recall@10
axs[1, 3].plot(deep_diskhivf_latency_r10, deep_diskhivf_recall_r10, marker='o', label='DiskHivf', color=colors['DiskHivf'])
axs[1, 3].plot(deep_diskann_latency_r10, deep_diskann_recall_r10, marker='s', label='DiskANN', color=colors['DiskANN'])
axs[1, 3].plot(deep_starling_latency_r10, deep_starling_recall_r10, marker='d', label='Starling', color=colors['Starling'])
axs[1, 3].set_title('DEEP1B 10-recall@10')
axs[1, 3].set_xlabel('Latency (ms)')
axs[1, 3].set_ylabel('10-recall@10')
axs[1, 3].grid(True)

# 调整布局，为顶部图例留出空间
plt.tight_layout()
plt.subplots_adjust(top=0.90)

# 添加统一的顶部图例
fig.legend(models, loc='upper center', ncol=len(models), frameon=False, prop={'size': 14})

# 保存图片
plt.savefig('/Users/kainhuang/Desktop/work/DiskHivf/script/combined_recall_all.png', dpi=150, bbox_inches='tight')
print("图片已保存: combined_recall_all.png")

# 显示图表
plt.show()
