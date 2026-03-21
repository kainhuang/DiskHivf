import matplotlib.pyplot as plt
import numpy as np

"""
VQ 指标对比图
VQ = 向量数 (Number of Vectors) / 内存消耗 (Memory in MB)
单位: vectors/MB

数据来源: performance_comparison.md
"""

# 数据集和方法
datasets = ['SIFT1M', 'GIST', 'BIGANN1B', 'DEEP1B']
methods = ['DiskANN', 'SPANN', 'Starling', 'DiskHIVF']

# 向量数 (单位: 百万)
vectors = {
    'SIFT1M': 1_000_000,      # 1M
    'GIST': 1_000_000,         # 1M
    'BIGANN1B': 1_000_000_000, # 1B
    'DEEP1B': 1_000_000_000,   # 1B
}

# 内存消耗 (单位: MB)，来自 performance_comparison.md
memory = {
    'SIFT1M': {
        'DiskANN': 50,
        'SPANN': 181,
        'Starling': 30,
        'DiskHIVF': 2.9,
    },
    'GIST': {
        'DiskANN': 205,
        'SPANN': 1207,
        'Starling': None,  # 数据不可用
        'DiskHIVF': 6.9,
    },
    'BIGANN1B': {
        'DiskANN': 32768,
        'SPANN': 32524,
        'Starling': 32425,
        'DiskHIVF': 1213,
    },
    'DEEP1B': {
        'DiskANN': 32675,
        'SPANN': None,     # 数据不可用
        'Starling': 32625,
        'DiskHIVF': 1210,
    },
}

# 计算 VQ 指标 (vectors/MB)
def calculate_vq(vec_count, mem_mb):
    if mem_mb is None or mem_mb == 0:
        return None
    return vec_count / mem_mb

# 计算每个方法在每个数据集上的 VQ
vq_data = {method: [] for method in methods}
for dataset in datasets:
    vec = vectors[dataset]
    for method in methods:
        mem = memory[dataset].get(method)
        vq = calculate_vq(vec, mem)
        vq_data[method].append(vq)

# 打印 VQ 数据
print("=" * 60)
print("VQ 指标 (vectors/MB)")
print("=" * 60)
print(f"{'Method':<12}", end="")
for dataset in datasets:
    print(f"{dataset:>15}", end="")
print()
print("-" * 60)
for method in methods:
    print(f"{method:<12}", end="")
    for i, dataset in enumerate(datasets):
        vq = vq_data[method][i]
        if vq is not None:
            print(f"{vq:>15.0f}", end="")
        else:
            print(f"{'N/A':>15}", end="")
    print()
print("=" * 60)

# 绘图
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# 颜色和样式
colors = {'DiskANN': 'C1', 'SPANN': 'C2', 'Starling': 'C3', 'DiskHIVF': 'C0'}
hatches = {'DiskANN': '/', 'SPANN': '\\', 'Starling': 'x', 'DiskHIVF': ''}

# ========== 左图: 百万级数据集 (SIFT1M, GIST) ==========
small_datasets = ['SIFT1M', 'GIST']
x_small = np.arange(len(small_datasets))
width = 0.2

for i, method in enumerate(methods):
    vq_values = []
    for dataset in small_datasets:
        idx = datasets.index(dataset)
        vq = vq_data[method][idx]
        vq_values.append(vq if vq is not None else 0)
    
    bars = axs[0].bar(x_small + i * width, vq_values, width, 
                      label=method, color=colors[method], 
                      hatch=hatches[method], edgecolor='black', linewidth=0.5)
    
    # 在柱子上添加数值标签
    for j, (bar, vq) in enumerate(zip(bars, vq_values)):
        if vq > 0:
            axs[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                       f'{vq:.0f}', ha='center', va='bottom', fontsize=8, rotation=45)

axs[0].set_xlabel('Dataset', fontsize=11)
axs[0].set_ylabel('VQ (vectors/MB)', fontsize=11)
axs[0].set_title('Million-scale Datasets (1M vectors)', fontsize=12)
axs[0].set_xticks(x_small + width * 1.5)
axs[0].set_xticklabels(small_datasets, fontsize=10)
axs[0].grid(True, alpha=0.3, axis='y')
axs[0].set_ylim(0, max([v for v in vq_data['DiskHIVF'][:2] if v]) * 1.3)

# ========== 右图: 十亿级数据集 (BIGANN1B, DEEP1B) ==========
large_datasets = ['BIGANN1B', 'DEEP1B']
x_large = np.arange(len(large_datasets))

for i, method in enumerate(methods):
    vq_values = []
    for dataset in large_datasets:
        idx = datasets.index(dataset)
        vq = vq_data[method][idx]
        vq_values.append(vq if vq is not None else 0)
    
    bars = axs[1].bar(x_large + i * width, vq_values, width, 
                      label=method, color=colors[method],
                      hatch=hatches[method], edgecolor='black', linewidth=0.5)
    
    # 在柱子上添加数值标签
    for j, (bar, vq) in enumerate(zip(bars, vq_values)):
        if vq > 0:
            axs[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10000, 
                       f'{vq/1e6:.2f}M', ha='center', va='bottom', fontsize=8, rotation=45)

axs[1].set_xlabel('Dataset', fontsize=11)
axs[1].set_ylabel('VQ (vectors/MB)', fontsize=11)
axs[1].set_title('Billion-scale Datasets (1B vectors)', fontsize=12)
axs[1].set_xticks(x_large + width * 1.5)
axs[1].set_xticklabels(large_datasets, fontsize=10)
axs[1].grid(True, alpha=0.3, axis='y')
axs[1].set_ylim(0, max([v for v in vq_data['DiskHIVF'][2:] if v]) * 1.3)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.85)

# 添加统一的顶部图例
fig.legend(methods, loc='upper center', ncol=len(methods), frameon=False, prop={'size': 11})

# 添加总标题
fig.suptitle('VQ Index Comparison (VQ = Vectors / Memory)', fontsize=14, y=0.98)

# 保存图片
output_path = '/Users/kainhuang/Desktop/work/DiskHivf/script/vq_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n图片已保存: {output_path}")

plt.close()
