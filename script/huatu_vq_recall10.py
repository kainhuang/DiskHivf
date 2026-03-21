import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

"""
VQ 指标折线图
VQ = (向量数 / 内存消耗KB) × QPS
    = (vectors / memory_KB) × QPS

横坐标: Recall@10 (%)
纵坐标: VQ
"""

# 并发数
CONCURRENCY = 32

# 计算 QPS: QPS = 并发数 * 1000 / latency(毫秒)
def latency_to_qps(latency_list, concurrency=CONCURRENCY):
    return [concurrency * 1000 / lat for lat in latency_list]

# 内存消耗 (单位: MB)，来自 performance_comparison.md
# 转换为 KB (乘以 1024)
memory_mb = {
    'BIGANN1B': {
        'DiskHivf': 1213,
        'DiskANN': 32768,
        'SPANN': 32524,
        'Starling': 32425,
    },
    'DEEP1B': {
        'DiskHivf': 1210,
        'DiskANN': 32675,
        'Starling': 32625,
    },
}

# 向量数
vectors = {
    'BIGANN1B': 1_000_000_000,  # 1B
    'DEEP1B': 1_000_000_000,    # 1B
}

# 计算 VQ = (向量数 / 内存KB) × QPS
def calculate_vq(qps_list, vec_count, mem_mb):
    mem_kb = mem_mb * 1024  # MB 转 KB
    return [(vec_count / mem_kb) * qps for qps in qps_list]

# ==================== BIGANN 数据集 ====================

# DiskHivf 数据 (单位: 毫秒)
bigann_diskhivf_latency = [
    4.67243, 5.62428, 7.28138, 9.3876, 11.2459,
    13.1826, 14.0249, 16.2248, 18.7072, 23.6548,
]
bigann_diskhivf_recall = [
    83.789001, 87.514999, 91.324997, 94.759003, 95.978996,
    96.510002, 96.772003, 97.408997, 97.870003, 98.400002,
]

# Diskann 数据 (单位: 毫秒)
bigann_diskann_latency = [
    6.89459, 8.17207, 9.25769, 11.62514, 13.17463,
    13.94541, 17.161, 20.75029,
]
bigann_diskann_recall = [
    80.67, 83.96, 86.33, 88.99, 90.79,
    91.74, 93.41, 94.53,
]

# Starling 数据 (单位: 毫秒)
bigann_starling_latency = [
    6.0574, 6.28532, 7.25123, 7.91669, 9.35602,
    11.01209, 13.13781, 15.00696, 16.96092, 18.77052, 20.47578,
]
bigann_starling_recall = [
    81.92, 83.94, 86.99, 89.14, 92.07,
    93.9, 95.38, 96.33, 96.99, 97.48, 97.88,
]

# SPANN 数据 (单位: 毫秒)
bigann_spann_latency = [
    3.991, 4.989, 5.987, 6.984, 7.981, 8.979, 9.975, 10.973,
    11.970, 13.965, 15.958, 17.952, 19.945, 21.938, 23.931, 25.924,
]
bigann_spann_recall = [
    80.1448, 82.8416, 84.8447, 86.4368, 87.7429, 88.7969, 89.6819, 90.418,
    91.037, 92.1951, 93.0692, 93.7894, 94.3394, 94.8025, 95.2355, 95.6265,
]

# ==================== DEEP 数据集 ====================

# DiskHivf 数据 (单位: 毫秒)
deep_diskhivf_latency = [
    4.14702, 4.69109, 5.57799, 7.54845, 8.71549,
    9.79135, 11.1709, 13.7541, 16.7548, 19.134, 21.936,
]
deep_diskhivf_recall = [
    68.711998, 73.757004, 79.547997, 86.028999, 88.848999,
    90.338997, 91.196999, 92.550003, 93.549004, 94.982002, 95.498001,
]

# Diskann 数据 (单位: 毫秒)
deep_diskann_latency = [
    4.22789, 5.30572, 6.42165, 7.33224, 8.8996,
    11.08917, 13.04274, 15.74573, 20.49054, 22.45362, 25.39027,
]
deep_diskann_recall = [
    62.61, 70.37, 75.27, 78.76, 81.44,
    85.02, 87.55, 89.73, 91.21, 92.3, 93.19,
]

# Starling 数据 (单位: 毫秒)
deep_starling_latency = [
    18.81018, 21.13159, 22.65648, 24.18094, 25.70404,
    27.25242, 30.32336, 33.42299, 36.53191, 39.65174,
]
deep_starling_recall = [
    72.65, 75.34, 77.42, 79.13, 80.6,
    81.94, 83.94, 85.58, 86.9, 87.96,
]

# 计算 QPS
bigann_diskhivf_qps = latency_to_qps(bigann_diskhivf_latency)
bigann_diskann_qps = latency_to_qps(bigann_diskann_latency)
bigann_starling_qps = latency_to_qps(bigann_starling_latency)
bigann_spann_qps = latency_to_qps(bigann_spann_latency)

deep_diskhivf_qps = latency_to_qps(deep_diskhivf_latency)
deep_diskann_qps = latency_to_qps(deep_diskann_latency)
deep_starling_qps = latency_to_qps(deep_starling_latency)

# 计算 VQ = (向量数 / 内存KB) × QPS
bigann_diskhivf_vq = calculate_vq(bigann_diskhivf_qps, vectors['BIGANN1B'], memory_mb['BIGANN1B']['DiskHivf'])
bigann_diskann_vq = calculate_vq(bigann_diskann_qps, vectors['BIGANN1B'], memory_mb['BIGANN1B']['DiskANN'])
bigann_starling_vq = calculate_vq(bigann_starling_qps, vectors['BIGANN1B'], memory_mb['BIGANN1B']['Starling'])
bigann_spann_vq = calculate_vq(bigann_spann_qps, vectors['BIGANN1B'], memory_mb['BIGANN1B']['SPANN'])

deep_diskhivf_vq = calculate_vq(deep_diskhivf_qps, vectors['DEEP1B'], memory_mb['DEEP1B']['DiskHivf'])
deep_diskann_vq = calculate_vq(deep_diskann_qps, vectors['DEEP1B'], memory_mb['DEEP1B']['DiskANN'])
deep_starling_vq = calculate_vq(deep_starling_qps, vectors['DEEP1B'], memory_mb['DEEP1B']['Starling'])

# 打印 VQ 数据样例
print("=" * 60)
print("VQ 指标样例 (VQ = vectors/KB × QPS)")
print("=" * 60)
print(f"BIGANN1B DiskHivf VQ (recall={bigann_diskhivf_recall[0]:.1f}%): {bigann_diskhivf_vq[0]:.2f}")
print(f"BIGANN1B DiskANN VQ (recall={bigann_diskann_recall[0]:.1f}%): {bigann_diskann_vq[0]:.2f}")
print(f"BIGANN1B Starling VQ (recall={bigann_starling_recall[0]:.1f}%): {bigann_starling_vq[0]:.2f}")
print(f"BIGANN1B SPANN VQ (recall={bigann_spann_recall[0]:.1f}%): {bigann_spann_vq[0]:.2f}")
print("=" * 60)

# 定义颜色和标记
colors = {'DiskHivf': 'C0', 'DiskANN': 'C1', 'SPANN': 'C2', 'Starling': 'C3'}
markers = {'DiskHivf': 'o', 'DiskANN': 's', 'SPANN': '^', 'Starling': 'd'}

# 定义模型列表（用于统一图例）
models = ['DiskHivf', 'DiskANN', 'SPANN', 'Starling']

# 创建 1x2 子图
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# ========== 左图: BIGANN1B ==========
axs[0].plot(bigann_diskhivf_recall, bigann_diskhivf_vq, marker='o', label='DiskHivf', color=colors['DiskHivf'], linewidth=2, markersize=6)
axs[0].plot(bigann_diskann_recall, bigann_diskann_vq, marker='s', label='DiskANN', color=colors['DiskANN'], linewidth=2, markersize=6)
axs[0].plot(bigann_spann_recall, bigann_spann_vq, marker='^', label='SPANN', color=colors['SPANN'], linewidth=2, markersize=6)
axs[0].plot(bigann_starling_recall, bigann_starling_vq, marker='d', label='Starling', color=colors['Starling'], linewidth=2, markersize=6)
axs[0].set_title('BIGANN1B', fontsize=12)
axs[0].set_xlabel('Recall@10 (%)', fontsize=10)
axs[0].set_ylabel('VQ (vectors/KB × QPS)', fontsize=10)
axs[0].grid(True, alpha=0.3)
# 直接显示实际数值，使用千位分隔符
axs[0].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))

# ========== 右图: DEEP1B ==========
axs[1].plot(deep_diskhivf_recall, deep_diskhivf_vq, marker='o', label='DiskHivf', color=colors['DiskHivf'], linewidth=2, markersize=6)
axs[1].plot(deep_diskann_recall, deep_diskann_vq, marker='s', label='DiskANN', color=colors['DiskANN'], linewidth=2, markersize=6)
axs[1].plot(deep_starling_recall, deep_starling_vq, marker='d', label='Starling', color=colors['Starling'], linewidth=2, markersize=6)
axs[1].set_title('DEEP1B', fontsize=12)
axs[1].set_xlabel('Recall@10 (%)', fontsize=10)
axs[1].set_ylabel('VQ (vectors/KB × QPS)', fontsize=10)
axs[1].grid(True, alpha=0.3)
# 直接显示实际数值，使用千位分隔符
axs[1].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))

# 调整布局，为顶部图例留出空间
plt.tight_layout()
plt.subplots_adjust(top=0.90)

# 添加统一的顶部图例
fig.legend(models, loc='upper center', ncol=len(models), frameon=False, prop={'size': 12})

# 保存图片
output_path = '/Users/kainhuang/Desktop/work/DiskHivf/script/vq_recall10.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n图片已保存: {output_path}")

# 关闭图表
plt.close()
