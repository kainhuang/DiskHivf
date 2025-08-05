import struct
import argparse

def read_idx_file(filename):
    with open(filename, 'rb') as f:
        # 读取维度和向量个数
        dim = struct.unpack('I', f.read(4))[0]
        n = struct.unpack('Q', f.read(8))[0]
        # 读取向量数据
        data = f.read(n * dim * 4)  # 每个int32占4字节
    return dim, n, data

def read_dis_file(filename):
    with open(filename, 'rb') as f:
        # 读取维度和向量个数
        dim = struct.unpack('I', f.read(4))[0]
        n = struct.unpack('Q', f.read(8))[0]
        # 读取向量数据
        data = f.read(n * dim * 4)  # 每个float占4字节
    return dim, n, data

def write_combined_file(output_filename, dim, n, idx_data, dis_data):
    with open(output_filename, 'wb') as f:
        # 写入维度和向量个数
        f.write(struct.pack('I', dim))
        f.write(struct.pack('Q', n))
        # 写入idx数据
        f.write(idx_data)
        # 写入dis数据
        f.write(dis_data)

def main():
    parser = argparse.ArgumentParser(description='Combine idx and dis files into a single file.')
    parser.add_argument('idx_file', type=str, help='Path to the idx.dim.ivec file')
    parser.add_argument('dis_file', type=str, help='Path to the dis.dim.fvec file')
    parser.add_argument('output_file', type=str, help='Path to the output idx_dis.dim.ivec file')

    args = parser.parse_args()

    # 读取两个文件的数据
    idx_dim, idx_n, idx_data = read_idx_file(args.idx_file)
    dis_dim, dis_n, dis_data = read_dis_file(args.dis_file)

    # 检查维度和向量个数是否匹配
    if idx_dim != dis_dim or idx_n != dis_n:
        raise ValueError("The dimensions or number of vectors in the files do not match.")

    # 写入合并后的文件
    write_combined_file(args.output_file, idx_dim, idx_n, idx_data, dis_data)

if __name__ == '__main__':
    main()