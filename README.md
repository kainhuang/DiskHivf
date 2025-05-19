# DiskHIVF: Disk-Resident Hierarchical Inverted File Index for Billion-scale Approximate Nearest Neighbor Search

## Train model abd build index
训练模型和建立索引
``` train and build 
./bin/disk_hivf_train_and_build <conf_file>
```
**conf_file:** 配置文件，具有以下参数：
```
train_data_file = ./data/sift/sift_learn.dim.fvecs # 训练集
index_data_file = ./data/sift/sift_base.dim.fvecs # 索引向量原始文件
model_file = ./output/sift1m_model # 模型输出文件
index_dir = .//output/sift1m_index # 索引输出文件
index_file_num = 1  # 索引保存的文件数
dim = 128 # 向量长度
kmeans_epoch = 20 # kmeans训练epoch数
kmeans_sample_rete = 1 # 对训练集的采样比例
batch_size = 128 # kmeans训练的batch size， 通常不用调整
kmeans_centers_select_type = 3 # kmeans的初始化中心方法 1-随机选择初始化中心 2-直接取前k个向量做初始化 3-用kmeans++的方法初始化中心，通常不用调
first_cluster_num = 300 # 一级中心数
second_cluster_num = 300 # 二级中心数
hierachical_cluster_epoch = 5   # 分层kmeans的训练epoch数
read_file_batch_size = 128  # 读取数据的batch size
build_index_search_first_center_num = 20    # 训练时搜索的一级中心数量，即论文中的r_building
search_first_center_num = 20    # 搜索时搜索一级中心的数量
search_second_center_num = 400  # 搜索时搜索二级中心的数量
is_disk = 1 # 0-纯内存模式，1-开启磁盘模式 默认是磁盘模式
search_neighbors = 100000   # 搜索时，最多搜索的向量数
search_block_num = 20000    # 搜索时，最多查询的磁盘块/内存块的数量
search_top_cut = 7  # 搜索时，对于距离大于当前最优解 search_top_cut 倍以上的单元格不再搜索
hs_mode = 1 # 0-将向量id放磁盘 1-将向量id放内存，通常hs_mode = 1比hs_mode = 0快10%，但需要更多内存
thread_num = 32 # 训练的线程数
read_index_file_thread_num = 5  # 读取索引的线程数
build_index_num = -1 # 取前build_index_num个index_data_file中的向量建立索引，-1-使用全部数据
train_data_num = -1 # 取前train_data_num个train_data_file中的向量训练模型，-1-使用全部数据
use_uint8_data = 0  # 表示train_data_file和index_data_file的类型 0-uint8类型，1-float类型
dynamic_prune_switch = 0 # 0-关闭动态剪枝 1-开启动态剪枝
dynamic_prune_a = 165.4 # dynamic_prune_switch = 1情况下生效，用learn_dynamic_prune_hyperparameter.sh生成
dynamic_prune_b = -30.9 # dynamic_prune_switch = 1情况下生效，用learn_dynamic_prune_hyperparameter.sh生成
dynamic_prune_c = 2.4 # dynamic_prune_switch = 1情况下生效，用learn_dynamic_prune_hyperparameter.sh生成
io_thread_num = 0 # 未完成
is_async_read = 0 # 未完成
use_cache = 0 # 未完成
cache_capacity = 0 # 未完成
cache_segment = 0 # 未完成
```

## Dynamic Prune
使用最小二乘法生成动态剪枝的超参数
```
sh learn_dynamic_prune_hyperparameter.sh <conf_file> <query_file>
```
**conf_file:** 配置文件，参数同上
**query_file:** 查询向量文件

## Run Test Set
运行测试集，进行评测
```
./bin/run_test_set <conf_file> <query_file> <groundtruth_file> <topk> <at_num> <thread_num> <first_centers_num> <second_centers_num> <debug_log> <use_cache> <is_query_uint8> <use_dist> <search_neighbors> <search_blocks>
```
**conf_file:** 配置文件，参数同上  
**query_file:** 查询向量文件  
**groundtruth_file:** 查询向量的groundtruth  
**topk:** 评估指标 topk-recall@at_num  
**at_num:** 评估指标 topk-recall@at_num  
**thread_num:** 评估时的线程数，通常取1  
**first_centers_num:** 搜索时搜索一级中心的数量，会覆盖conf_file的参数  
**second_centers_num:** 搜索时搜索二级中心的数量，会覆盖conf_file的参数   
**debug_log:** 0-关闭debug_log 1-打开debug_log  
**use_cache:** 未完成，通常设置为0  
**is_query_uint8:** 1-query向量是uint8 0-query向量是float  
**use_dist:** groundtruth_file中带有距离用于评估  
**search_neighbors:** 搜索时，最多搜索的向量数，会覆盖conf_file的参数  
**search_blocks:** 搜索时，最多查询的磁盘块/内存块的数量，会覆盖conf_file的参数   

### 例子
```
./bin/run_test_set conf/hivf.conf data/sift/sift_query.dim.fvecs data/sift/sift_groundtruth.dim.ivecs 10 10 1 70 5000 0 0 0 1
```
