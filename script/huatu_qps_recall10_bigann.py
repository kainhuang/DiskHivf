import matplotlib.pyplot as plt
import numpy as np

# 并发数
CONCURRENCY = 32

# 计算 QPS: QPS = 并发数 * 1000 / latency(毫秒)
def latency_to_qps(latency_list, concurrency=CONCURRENCY):
    return [concurrency * 1_000 / lat for lat in latency_list]

# ==================== BIGANN 数据集 ====================

# DiskHivf 数据 (单位: 毫秒)
diskhivf_latency = [
    4.67243, 5.62428, 7.28138, 9.3876, 11.2459,
    13.1826, 14.0249, 16.2248, 18.7072, 23.6548,
]
diskhivf_recall = [
    83.789001, 87.514999, 91.324997, 94.759003, 95.978996,
    96.510002, 96.772003, 97.408997, 97.870003, 98.400002,
]

# Diskann 数据 (单位: 毫秒)
diskann_latency = [
    6.89459, 8.17207, 9.25769, 11.62514, 13.17463,
    13.94541, 17.161, 20.75029,
]
diskann_recall = [
    80.67, 83.96, 86.33, 88.99, 90.79,
    91.74, 93.41, 94.53,
]

# Starling 数据 (单位: 毫秒)
starling_latency = [
    6.0574, 6.28532, 7.25123, 7.91669, 9.35602,
    11.01209, 13.13781, 15.00696, 16.96092, 18.77052, 20.47578,
]
starling_recall = [
    81.92, 83.94, 86.99, 89.14, 92.07,
    93.9, 95.38, 96.33, 96.99, 97.48, 97.88,
]

# SPANN 数据 (单位: 毫秒)
spann_latency = [
    3.991, 4.989, 5.987, 6.984, 7.981, 8.979, 9.975, 10.973,
    11.970, 13.965, 15.958, 17.952, 19.945, 21.938, 23.931, 25.924,
]
spann_recall = [
    80.1448, 82.8416, 84.8447, 86.4368, 87.7429, 88.7969, 89.6819, 90.418,
    91.037, 92.1951, 93.0692, 93.7894, 94.3394, 94.8025, 95.2355, 95.6265,
]

# 计算 QPS
diskhivf_qps = latency_to_qps(diskhivf_latency)
diskann_qps = latency_to_qps(diskann_latency)
starling_qps = latency_to_qps(starling_latency)
spann_qps = latency_to_qps(spann_latency)

# 创建折线图
plt.figure(figsize=(10, 6))
plt.plot(diskhivf_recall, diskhivf_qps, marker='o', label='DiskHivf', linewidth=2, markersize=6)
plt.plot(diskann_recall, diskann_qps, marker='s', label='DiskANN', linewidth=2, markersize=6)
plt.plot(starling_recall, starling_qps, marker='^', label='Starling', color='red', linewidth=2, markersize=6)
plt.plot(spann_recall, spann_qps, marker='d', label='SPANN', color='green', linewidth=2, markersize=6)

# 添加标题和标签
plt.title('BIGANN1B: QPS vs. Recall@10 (Concurrency=32)', fontsize=14)
plt.xlabel('Recall@10 (%)', fontsize=12)
plt.ylabel('QPS', fontsize=12)

# 显示图例
plt.legend(fontsize=10)

# 显示网格
plt.grid(True, alpha=0.3)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('/Users/kainhuang/Desktop/work/DiskHivf/script/bigann_qps_recall10.png', dpi=150)
print("图片已保存: bigann_qps_recall10.png")

# 显示图表
plt.show()
