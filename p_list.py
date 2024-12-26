import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":
    print (sys.argv)
    '''
    parser = argparse.ArgumentParser()
    # 添加参数
    parser.add_argument('--input', type=str, default='learn_hp_data', help='输入文件的路径')
    parser.add_argument('--percent', type=float, default=99, help='XX%分位')
    parser.add_argument('--bais', type=float, default=10, help='人工偏移量')
    args = parser.parse_args()

    input_file = args.input
    percent = args.percent
    '''
    percent = 99
    up = 5
    rank = []
    dis = []
    rank99 = []
    avg_dis = 0
    for line in open('learn_hp_data'):
        cols = line.strip().split('\t')
        if cols[1] == 'm_build_index_loss':
            avg_dis = float(cols[2])
            continue

        rank.append(int(cols[2]))
        percentile_99 = np.percentile(rank, percent)

        ds = float(cols[6]) / avg_dis
        if ds > 1.5:
            continue
        rank99.append(percentile_99)
        dis.append(ds)
        #print(f'{dis[-1]}\t{percentile_99}')
    x = dis
    y = rank99

    # 多项式拟合（例如，二次多项式）
    degree = 2
    coefficients = np.polyfit(x, y, degree)


    # 生成多项式函数
    polynomial = np.poly1d(coefficients)

    # 生成拟合曲线的x值
    x_fit = np.linspace(min(x), max(x), 100)

    
    # 计算拟合曲线的y值
    y_fit = polynomial(x_fit) + up
    # 输出多项式的参数，保留1位小数
    coefficients[2] += up

    formatted_coefficients = [f"{(coef) :.1f}" for coef in coefficients]

    print (polynomial)
    print ("Please add the following parameters to your configuration file")
    print ("dynamic_prune_switch = 1")
    print (f"dynamic_prune_a = {formatted_coefficients[0]}")
    print (f"dynamic_prune_b = {formatted_coefficients[1]}")
    print (f"dynamic_prune_c = {formatted_coefficients[2]}")
    
    # 创建一个新的图形
    plt.figure()

    # 绘制原始数据点
    plt.scatter(x, y, color='red', label='Original Data')

    # 绘制拟合曲线
    plt.plot(x_fit, y_fit, color='blue', label=f'Polynomial Fit (degree={degree})')

    # 添加标题和标签
    plt.title('Polynomial Fit')
    plt.xlabel('current_distance / avg_distance')
    plt.ylabel('99% length of search list')

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True)

    # 显示图形
    plt.show()