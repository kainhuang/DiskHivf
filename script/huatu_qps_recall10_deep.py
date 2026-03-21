import matplotlib.pyplot as plt
import numpy as np

# 并发数
CONCURRENCY = 32

# 计算 QPS: QPS = 并发数 * 1000 / latency(毫秒)
def latency_to_qps(latency_list, concurrency=CONCURRENCY):
    return [concurrency * 1000 / lat for lat in latency_list]

# ==================== DEEP 数据集 ====================

# DiskHivf 数据 (单位: 毫秒)
diskhivf_latency = [
    4.14702, 4.69109, 5.57799, 7.54845, 8.71549,
    9.79135, 11.1709, 13.7541, 16.7548, 19.134, 21.936,
]
diskhivf_recall = [
    68.711998, 73.757004, 79.547997, 86.028999, 88.848999,
    90.338997, 91.196999, 92.550003, 93.549004, 94.982002, 95.498001,
]

# Diskann 数据 (单位: 毫秒)
diskann_latency = [
    4.22789, 5.30572, 6.42165, 7.33224, 8.8996,
    11.08917, 13.04274, 15.74573, 20.49054, 22.45362, 25.39027,
]
diskann_recall = [
    62.61, 70.37, 75.27, 78.76, 81.44,
    85.02, 87.55, 89.73, 91.21, 92.3, 93.19,
]

# Starling 数据 (单位: 毫秒)
starling_latency = [
    18.81018, 21.13159, 22.65648, 24.18094, 25.70404,
    27.25242, 30.32336, 33.42299, 36.53191, 39.65174,
]
starling_recall = [
    72.65, 75.34, 77.42, 79.13, 80.6,
    81.94, 83.94, 85.58, 86.9, 87.96,
]

# 计算 QPS
diskhivf_qps = latency_to_qps(diskhivf_latency)
diskann_qps = latency_to_qps(diskann_latency)
starling_qps = latency_to_qps(starling_latency)

# 创建折线图
plt.figure(figsize=(10, 6))
plt.plot(diskhivf_recall, diskhivf_qps, marker='o', label='DiskHivf', linewidth=2, markersize=6)
plt.plot(diskann_recall, diskann_qps, marker='s', label='DiskANN', linewidth=2, markersize=6)
plt.plot(starling_recall, starling_qps, marker='^', label='Starling', color='red', linewidth=2, markersize=6)

# 添加标题和标签
plt.title('DEEP1B: QPS vs. Recall@10 (Concurrency=32)', fontsize=14)
plt.xlabel('Recall@10 (%)', fontsize=12)
plt.ylabel('QPS', fontsize=12)

# 显示图例
plt.legend(fontsize=10)

# 显示网格
plt.grid(True, alpha=0.3)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('/Users/kainhuang/Desktop/work/DiskHivf/script/deep_qps_recall10.png', dpi=150)
print("图片已保存: deep_qps_recall10.png")

# 显示图表
plt.show()
