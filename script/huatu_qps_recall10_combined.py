import matplotlib.pyplot as plt
import numpy as np

# 并发数
CONCURRENCY = 32

# 计算 QPS: QPS = 并发数 * 1000 / latency(毫秒)
def latency_to_qps(latency_list, concurrency=CONCURRENCY):
    return [concurrency * 1000 / lat for lat in latency_list]

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

# ==================== 输出 QPS 数值 ====================
print("=" * 60)
print("BIGANN 数据集 QPS 数值")
print("=" * 60)
print(f"{'方法':<12} {'Recall@10 (%)':<15} {'Latency (ms)':<15} {'QPS':<10}")
print("-" * 60)
print("DiskHIVF:")
for r, l, q in zip(bigann_diskhivf_recall, bigann_diskhivf_latency, bigann_diskhivf_qps):
    print(f"{'':12} {r:<15.2f} {l:<15.3f} {q:<10.2f}")
print("\nDiskANN:")
for r, l, q in zip(bigann_diskann_recall, bigann_diskann_latency, bigann_diskann_qps):
    print(f"{'':12} {r:<15.2f} {l:<15.3f} {q:<10.2f}")
print("\nStarling:")
for r, l, q in zip(bigann_starling_recall, bigann_starling_latency, bigann_starling_qps):
    print(f"{'':12} {r:<15.2f} {l:<15.3f} {q:<10.2f}")
print("\nSPANN:")
for r, l, q in zip(bigann_spann_recall, bigann_spann_latency, bigann_spann_qps):
    print(f"{'':12} {r:<15.2f} {l:<15.3f} {q:<10.2f}")

print("\n" + "=" * 60)
print("DEEP 数据集 QPS 数值")
print("=" * 60)
print(f"{'方法':<12} {'Recall@10 (%)':<15} {'Latency (ms)':<15} {'QPS':<10}")
print("-" * 60)
print("DiskHIVF:")
for r, l, q in zip(deep_diskhivf_recall, deep_diskhivf_latency, deep_diskhivf_qps):
    print(f"{'':12} {r:<15.2f} {l:<15.3f} {q:<10.2f}")
print("\nDiskANN:")
for r, l, q in zip(deep_diskann_recall, deep_diskann_latency, deep_diskann_qps):
    print(f"{'':12} {r:<15.2f} {l:<15.3f} {q:<10.2f}")
print("\nStarling:")
for r, l, q in zip(deep_starling_recall, deep_starling_latency, deep_starling_qps):
    print(f"{'':12} {r:<15.2f} {l:<15.3f} {q:<10.2f}")

# ==================== 计算 QPS 提升 ====================
def interpolate_qps_at_recall(recalls, qps_list, target_recall):
    """在目标召回率处插值计算QPS"""
    recalls = np.array(recalls)
    qps_list = np.array(qps_list)
    # 按recall排序
    sorted_idx = np.argsort(recalls)
    recalls = recalls[sorted_idx]
    qps_list = qps_list[sorted_idx]
    
    if target_recall < recalls.min() or target_recall > recalls.max():
        return None
    return np.interp(target_recall, recalls, qps_list)

print("\n" + "=" * 60)
print("QPS 提升分析 (在 90% Recall@10 附近)")
print("=" * 60)

# BIGANN 数据集 - 90% recall@10
target_recall = 90.0
print(f"\n【BIGANN 数据集】目标 Recall@10 = {target_recall}%")
bigann_diskhivf_qps_at_90 = interpolate_qps_at_recall(bigann_diskhivf_recall, bigann_diskhivf_qps, target_recall)
bigann_diskann_qps_at_90 = interpolate_qps_at_recall(bigann_diskann_recall, bigann_diskann_qps, target_recall)
bigann_starling_qps_at_90 = interpolate_qps_at_recall(bigann_starling_recall, bigann_starling_qps, target_recall)
bigann_spann_qps_at_90 = interpolate_qps_at_recall(bigann_spann_recall, bigann_spann_qps, target_recall)

print(f"  DiskHIVF QPS: {bigann_diskhivf_qps_at_90:.2f}" if bigann_diskhivf_qps_at_90 else "  DiskHIVF: 未达到目标召回率")
print(f"  DiskANN QPS: {bigann_diskann_qps_at_90:.2f}" if bigann_diskann_qps_at_90 else "  DiskANN: 未达到目标召回率")
print(f"  Starling QPS: {bigann_starling_qps_at_90:.2f}" if bigann_starling_qps_at_90 else "  Starling: 未达到目标召回率")
print(f"  SPANN QPS: {bigann_spann_qps_at_90:.2f}" if bigann_spann_qps_at_90 else "  SPANN: 未达到目标召回率")

print("\n  QPS 提升比例:")
if bigann_diskhivf_qps_at_90 and bigann_diskann_qps_at_90:
    print(f"    DiskHIVF vs DiskANN: {bigann_diskhivf_qps_at_90/bigann_diskann_qps_at_90:.2f}x")
if bigann_diskhivf_qps_at_90 and bigann_starling_qps_at_90:
    print(f"    DiskHIVF vs Starling: {bigann_diskhivf_qps_at_90/bigann_starling_qps_at_90:.2f}x")
if bigann_diskhivf_qps_at_90 and bigann_spann_qps_at_90:
    print(f"    DiskHIVF vs SPANN: {bigann_diskhivf_qps_at_90/bigann_spann_qps_at_90:.2f}x")

# DEEP 数据集 - 90% recall@10
print(f"\n【DEEP 数据集】目标 Recall@10 = {target_recall}%")
deep_diskhivf_qps_at_90 = interpolate_qps_at_recall(deep_diskhivf_recall, deep_diskhivf_qps, target_recall)
deep_diskann_qps_at_90 = interpolate_qps_at_recall(deep_diskann_recall, deep_diskann_qps, target_recall)
deep_starling_qps_at_90 = interpolate_qps_at_recall(deep_starling_recall, deep_starling_qps, target_recall)

print(f"  DiskHIVF QPS: {deep_diskhivf_qps_at_90:.2f}" if deep_diskhivf_qps_at_90 else "  DiskHIVF: 未达到目标召回率")
print(f"  DiskANN QPS: {deep_diskann_qps_at_90:.2f}" if deep_diskann_qps_at_90 else "  DiskANN: 未达到目标召回率")
print(f"  Starling QPS: {deep_starling_qps_at_90:.2f}" if deep_starling_qps_at_90 else "  Starling: 未达到目标召回率")

print("\n  QPS 提升比例:")
if deep_diskhivf_qps_at_90 and deep_diskann_qps_at_90:
    print(f"    DiskHIVF vs DiskANN: {deep_diskhivf_qps_at_90/deep_diskann_qps_at_90:.2f}x")
if deep_diskhivf_qps_at_90 and deep_starling_qps_at_90:
    print(f"    DiskHIVF vs Starling: {deep_diskhivf_qps_at_90/deep_starling_qps_at_90:.2f}x")
else:
    print("    Starling 在 DEEP 数据集上未达到 90% recall@10")

print("\n" + "=" * 60)

# 定义颜色和标记
colors = {'DiskHivf': 'C0', 'DiskANN': 'C1', 'SPANN': 'C2', 'Starling': 'C3'}
markers = {'DiskHivf': 'o', 'DiskANN': 's', 'SPANN': '^', 'Starling': 'd'}

# 定义模型列表（用于统一图例）
models = ['DiskHivf', 'DiskANN', 'SPANN', 'Starling']

# 创建 1x2 子图
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# ========== 左图: BIGANN1B ==========
axs[0].plot(bigann_diskhivf_recall, bigann_diskhivf_qps, marker='o', label='DiskHivf', color=colors['DiskHivf'], linewidth=2, markersize=6)
axs[0].plot(bigann_diskann_recall, bigann_diskann_qps, marker='s', label='DiskANN', color=colors['DiskANN'], linewidth=2, markersize=6)
axs[0].plot(bigann_spann_recall, bigann_spann_qps, marker='^', label='SPANN', color=colors['SPANN'], linewidth=2, markersize=6)
axs[0].plot(bigann_starling_recall, bigann_starling_qps, marker='d', label='Starling', color=colors['Starling'], linewidth=2, markersize=6)
axs[0].set_title('BIGANN1B', fontsize=12)
axs[0].set_xlabel('Recall@10 (%)', fontsize=10)
axs[0].set_ylabel('QPS', fontsize=10)
axs[0].grid(True, alpha=0.3)

# ========== 右图: DEEP1B ==========
axs[1].plot(deep_diskhivf_recall, deep_diskhivf_qps, marker='o', label='DiskHivf', color=colors['DiskHivf'], linewidth=2, markersize=6)
axs[1].plot(deep_diskann_recall, deep_diskann_qps, marker='s', label='DiskANN', color=colors['DiskANN'], linewidth=2, markersize=6)
axs[1].plot(deep_starling_recall, deep_starling_qps, marker='d', label='Starling', color=colors['Starling'], linewidth=2, markersize=6)
axs[1].set_title('DEEP1B', fontsize=12)
axs[1].set_xlabel('Recall@10 (%)', fontsize=10)
axs[1].set_ylabel('QPS', fontsize=10)
axs[1].grid(True, alpha=0.3)

# 调整布局，为顶部图例留出空间
plt.tight_layout()
plt.subplots_adjust(top=0.85)

# 添加统一的顶部图例
fig.legend(models, loc='upper center', ncol=len(models), frameon=False, prop={'size': 12})

# 保存图片
plt.savefig('/Users/kainhuang/Desktop/work/DiskHivf/script/combined_qps_recall10.png', dpi=150, bbox_inches='tight')
print("图片已保存: combined_qps_recall10.png")

# 关闭图表
plt.close()
