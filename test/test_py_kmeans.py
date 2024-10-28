import numpy as np
from sklearn.cluster import KMeans
import struct
import sys
import time

def read_binary_file(file_path):
    with open(file_path, 'rb') as f:
        # 读取向量维度
        dim = struct.unpack('I', f.read(4))[0]
        # 读取向量数量
        num_vecs = struct.unpack('Q', f.read(8))[0]
        # 读取所有向量数据
        data = np.fromfile(f, dtype=np.float32, count=num_vecs * dim)
        # 重塑数据为二维数组
        data = data.reshape((num_vecs, dim))
        print (data)
    return data

def main(file_path, n_clusters=300):
    # 读取二进制文件
    data = read_binary_file(file_path)
    print (data.shape)
    # 运行KMeans算法
    st = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=20)
    kmeans.fit(data)
    ed = time.time()
    labels = kmeans.predict(data)
    
    print ("time cost ", ed - st)
    
    # 输出结果
    print("Cluster centers:\n", kmeans.cluster_centers_)
    print("Labels:\n", labels)
    print("Inertia (loss):", kmeans.inertia_ / data.shape[0])

if __name__ == "__main__":
    # 替换为你的二进制文件路径
    file_path = sys.argv[1]
    main(file_path)