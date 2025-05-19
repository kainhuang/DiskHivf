# DiskHIVF: Disk-Resident Hierarchical Inverted File Index for Billion-scale Approximate Nearest Neighbor Search

Here is the official implementation of the experiments of DiskHIVF: Disk-Resident Hierarchical Inverted File Index for Billion-scale Approximate Nearest Neighbor Search.

>Our experiments mainly include comparisons with three models: DiskANN, SPANN, and Starling.

## Models

The models are sourced from the following content respectively.

>DiskANN: https://github.com/microsoft/DiskANN  
>SPANN: https://github.com/microsoft/SPTAG  
>starling: https://github.com/zilliztech/starling  

## Training

To obtain the data as in the paper, please run the following commands:

#### SIFT1M-DiskANN:

```train
 ./apps/utils/compute_groundtruth  --data_type float --dist_fn l2 --base_file data/sift/sift_learn.fbin --query_file  data/sift/sift_query.fbin --gt_file data/sift/sift_query_learn_gt100 --K 100
 ./apps/build_disk_index --data_type float --dist_fn l2 --data_path data/sift/sift_learn.fbin --index_path_prefix data/sift/disk_index_sift_base_R64_L100_A1.3 -R 64 -L 100 -B 0.05 -M 64 -T 32 --build_PQ_bytes 32
```

#### SIFT1M-SPANN:

```train
./ssdserving buildconfig.ini
```
Among them, the buildconfig.ini should be set as follows.
```train
[Base]
ValueType=Float
DistCalcMethod=L2
IndexAlgoType=BKT
Dim=128
VectorPath=sift1m/sift_base.fvecs
VectorType=XVEC
QueryPath=sift1m/sift_query.fvecs
QueryType=XVEC
WarmupPath=sift1m/sift_query.fvecs
WarmupType=XVEC
TruthPath=sift1m/sift_groundtruth.ivecs
TruthType=XVEC
IndexDirectory=sift1m

[SelectHead]
isExecute=true
TreeNumber=1
BKTKmeansK=32
BKTLeafSize=8
SamplesNumber=1000
SaveBKT=false
SelectThreshold=50
SplitFactor=6
SplitThreshold=100
Ratio=0.16
NumberOfThreads=32
BKTLambdaFactor=-1

[BuildHead]
isExecute=true
NeighborhoodSize=32
TPTNumber=32
TPTLeafSize=2000
MaxCheck=8192
MaxCheckForRefineGraph=8192
RefineIterations=3
NumberOfThreads=32
BKTLambdaFactor=-1

[BuildSSDIndex]
isExecute=true
BuildSsdIndex=true
InternalResultNum=64
ReplicaCount=4
PostingPageLimit=12
NumberOfThreads=32
MaxCheck=8192
TmpDir=/tmp/
```

#### SIFT1M-starling:

```train
 ./run_benchmark.sh release build knn
 ./run_benchmark.sh release gp knn
```
Among them, the config_sample.sh should be set as follows.
```train
dataset_sift_1m() {
  BASE_PATH=data/sift/sift_base.fbin
  QUERY_FILE=data/sift/sift_query.fbin
  GT_FILE=data/sift_query_base_gt100
  PREFIX=sift_1m
  DATA_TYPE=float
  DIST_FN=l2
  B=0.3
  K=10
  DATA_DIM=128
  DATA_N=1000000
}
```
Among them, the config_local.sh should be set as follows.
```train
##################
#   Disk Build   #
##################
R=48
BUILD_L=128
M=32
BUILD_T=32
```

#### BIGANN1B-DiskANN:

```train
 ./apps/utils/compute_groundtruth  --data_type uint8 --dist_fn l2 --base_file data/bigann/bigann_base.bbin --query_file  data/bigann/bigann_query.bbin --gt_file data/bigann/bigann_query_base_gt100 --K 100
 ./apps/build_disk_index --data_type uint8 --dist_fn l2 --data_path data/bigann/bigann_base.bbin --index_path_prefix data/bigann/disk_index_bigann_base_R64_L50_A1.2 -R 64 -L 50 -B 32 -M 32 -T 32 --build_PQ_bytes 32 
```

#### BIGANN1B-starling:

```train
./run_benchmark.sh release build knn
./run_benchmark.sh release gp knn
```
Among them, the config_sample.sh should be set as follows.
```train
dataset_bigann_1B() {
  BASE_PATH=data/bigann/bigann_base.bbin
  QUERY_FILE=data/bigann/bigann_query.bbin
  GT_FILE=data/bigann/bigann_query_base_gt100
  PREFIX=bigann_1b
  DATA_TYPE=uint8
  DIST_FN=l2
  B=32
  K=1
  DATA_DIM=128
  DATA_N=1000000000
}
```
Among them, the config_local.sh should be set as follows.
```train
##################
#   Disk Build   #
##################
R=32
BUILD_L=100
M=32
BUILD_T=32
```

#### GIST1M-DiskANN:

```train
 ./apps/utils/compute_groundtruth  --data_type float --dist_fn l2 --base_file data/gist/gist_base.fbin --query_file  data/gist/gist_query.fbin --gt_file data/gist/gist_query_base_gt100 --K 100
 ./apps/build_disk_index --data_type float --dist_fn l2 --data_path data/gist/gist_base.fbin --index_path_prefix data/gist/disk_index_gist_base_R64_L100_A1.2 -R 64 -L 100 -B 0.05 -M 32 -T 32 --build_PQ_bytes 240
```

#### GIST1M-SPANN:
```train
 ./ssdserving buildconfig.ini
```
Among them, the buildconfig.ini should be set as follows.
```train
[Base]
ValueType=Float
DistCalcMethod=L2
IndexAlgoType=BKT
Dim=960
VectorPath=data/gist/gist_base.fvecs
VectorType=XVEC
QueryPath=data/gist/gist_query.fvecs
QueryType=XVEC
WarmupPath=data/gist/gist_query.fvecs
WarmupType=XVEC
TruthPath=data/gist/gist_groundtruth.ivecs
TruthType=XVEC
IndexDirectory=gist1m

[SelectHead]
isExecute=true
TreeNumber=1
BKTKmeansK=32 #
BKTLeafSize=8
SamplesNumber=1000
SaveBKT=false
SelectThreshold=50
SplitFactor=6
SplitThreshold=100
Ratio=0.16
NumberOfThreads=32
BKTLambdaFactor=-1

[BuildHead]
isExecute=true
NeighborhoodSize=32
TPTNumber=32
TPTLeafSize=2000
MaxCheck=8192
MaxCheckForRefineGraph=8192
RefineIterations=3
NumberOfThreads=32
BKTLambdaFactor=-1

[BuildSSDIndex]
isExecute=true
BuildSsdIndex=true
InternalResultNum=64
ReplicaCount=4
PostingPageLimit=12
NumberOfThreads=32
MaxCheck=8192
TmpDir=/tmp/
```
#### INDEX10M-DiskANN:

```train
 ./apps/utils/compute_groundtruth  --data_type float --dist_fn l2 --base_file data/index10m/index10m.fbin --query_file data/index10m/query.fbin --gt_file data/index10m/data_query_index_gt100 --K 100
 ./apps/build_disk_index --data_type float --dist_fn l2 --data_path data/index10m/index10m.fbin --index_path_prefix data/index10m/disk_index_data_learn_R64_L100_A1.2 -R 64 -L 100 -B 1.6 -M 64 -T 32 --build_PQ_bytes 128
```

#### INDEX10M-SPANN:

```train
./ssdserving buildconfig.ini
```
Among them, the buildconfig.ini should be set as follows.
```train
[Base]
ValueType=Float
DistCalcMethod=L2
IndexAlgoType=BKT
Dim=512
VectorPath=data/index10m/index.fbin
VectorType=DEFAULT
QueryPath=data/index10m/query.fbin
QueryType=DEFAULT
WarmupPath=data/index10m/query.fbin
WarmupType=DEFAULT
TruthPath=data/index10m/query_gt.ivecs
TruthType=XVEC
IndexDirectory=index10m_2

[SelectHead]
isExecute=true
TreeNumber=1
BKTKmeansK=32
BKTLeafSize=8
SamplesNumber=1000
SaveBKT=false
SelectThreshold=50
SplitFactor=6
SplitThreshold=100
Ratio=0.16
NumberOfThreads=32
BKTLambdaFactor=-1

[BuildHead]
isExecute=true
NeighborhoodSize=32
TPTNumber=32
TPTLeafSize=2000
MaxCheck=8192
MaxCheckForRefineGraph=8192
RefineIterations=3
NumberOfThreads=32
BKTLambdaFactor=-1

[BuildSSDIndex]
isExecute=true
BuildSsdIndex=true
InternalResultNum=64
ReplicaCount=4
PostingPageLimit=12
NumberOfThreads=32
MaxCheck=8192
TmpDir=/tmp/
```

#### INDEX10M-starling:

```train
 ./run_benchmark.sh release build knn
 ./run_benchmark.sh release gp knn
```
Among them, the config_sample.sh should be set as follows.
```train
dataset_index_10m() {
  BASE_PATH=data/index10m/index.fbin
  QUERY_FILE=data/index10m/query.fbin
  GT_FILE=data/index10m/query_gt.bin
  PREFIX=index2_10m
  DATA_TYPE=float
  DIST_FN=l2
  B=1.6
  K=1
  DATA_DIM=512
  DATA_N=10000000
}
```
Among them, the config_local.sh should be set as follows.
```train
##################
#   Disk Build   #
##################
R=32
BUILD_L=100
M=32
BUILD_T=32
```

## Evaluation

To obtain the data as in the paper, please run the following commands:

#### SIFT1M-DiskANN:
```eval
 ./apps/search_disk_index  --data_type float --dist_fn l2 --index_path_prefix data/sift/disk_index_sift_base_R64_L100_A1.3 --query_file data/sift/sift_query.fbin  --gt_file data/sift/sift_query_learn_gt100 -K 1 -L 4 5 6 7 8 9 10 12 14 16 18 20 25 30 35 40 50 75 100 --result_path data/sift/res --num_nodes_to_cache 10000
 ./apps/search_disk_index  --data_type float --dist_fn l2 --index_path_prefix data/sift/disk_index_sift_base_R64_L100_A1.3 --query_file data/sift/sift_query.fbin  --gt_file data/sift/sift_query_learn_gt100 -K 10 -L 10 12 14 16 18 20 22 24 26 28 30 40 50 60 --result_path data/sift/res --num_nodes_to_cache 10000
```

#### SIFT1M-SPANN:
Since the multi-round execution of SPANN is relatively complex, only the values of `InternalResultNum` in the `[SearchSSDIndex]` section of `buildconfig.ini` are demonstrated here.
```eval
Recall@1: 10 11 12 15 20 25 30 40 50 75 100 150
Recall@10: 20 25 30 35 40 60 80 100 120 140
```

#### SIFT1M-starling:

```eval
./run_benchmark.sh release search knn
```
For Recall@1, the config_sample.sh should be set as follows.
```eval
dataset_sift_1m() {
  BASE_PATH=data/sift/sift_base.fbin
  QUERY_FILE=data/sift/sift_query.fbin
  GT_FILE=data/sift_query_base_gt100
  PREFIX=sift_1m
  DATA_TYPE=float
  DIST_FN=l2
  B=0.3
  K=1
  DATA_DIM=128
  DATA_N=1000000
}
```
For Recall@1, the config_local.sh should be set as follows.
```eval
# KNN
LS="2 3 4 5 6 7 8 9 10 15 20"
```

For Recall@10, the config_sample.sh should be set as follows.
```eval
dataset_sift_1m() {
  BASE_PATH=data/sift/sift_base.fbin
  QUERY_FILE=data/sift/sift_query.fbin
  GT_FILE=data/sift_query_base_gt100
  PREFIX=sift_1m
  DATA_TYPE=float
  DIST_FN=l2
  B=0.3
  K=10
  DATA_DIM=128
  DATA_N=1000000
}
```
For Recall@10, the config_local.sh should be set as follows.
```eval
# KNN
LS="10 15 20 25 30 35 40 45 50 55 60 65 70 75 100"
```

#### BIGANN1B-DiskANN:
```eval
 ./apps/search_disk_index  --data_type uint8 --dist_fn l2 --index_path_prefix data/bigann/disk_index_bigann_base_R64_L50_A1.2 --query_file data/gist/gist_query.fbin  --gt_file data/bigann/bigann_query_base_gt100 -K 1 -L 25 30 35 40 45 50 60 70 80 90 100 125 150 175 200 250 300 --result_path data/bigann/res --num_nodes_to_cache 10000
 ./apps/search_disk_index  --data_type uint8 --dist_fn l2 --index_path_prefix data/bigann/disk_index_bigann_base_R64_L50_A1.2 --query_file data/gist/gist_query.fbin  --gt_file data/bigann/bigann_query_base_gt100 -K 10 -L 40 50 60 70 80 90 100 125 150 175 200 225 250 275 300 350 400 450 --result_path data/bigann/res --num_nodes_to_cache 10000
```

BIGANN1B-starling:
```eval
./run_benchmark.sh release search knn
```
For Recall@1, the config_sample.sh should be set as follows.
```eval
dataset_bigann_1B() {
  BASE_PATH=data/bigann/bigann_base.bbin
  QUERY_FILE=data/bigann/bigann_query.bbin
  GT_FILE=data/bigann/bigann_query_base_gt100
  PREFIX=bigann_1b
  DATA_TYPE=uint8
  DIST_FN=l2
  B=32
  K=1
  DATA_DIM=128
  DATA_N=1000000000
}
```
For Recall@1, the config_local.sh should be set as follows.
```eval
# KNN
LS="20 25 30 35 40 45 50 60 70 80 90 100 125 150 175 200 250 300"
```

For Recall@10, the config_sample.sh should be set as follows.
```eval
dataset_bigann_1B() {
  BASE_PATH=data/bigann/bigann_base.bbin
  QUERY_FILE=data/bigann/bigann_query.bbin
  GT_FILE=data/bigann/bigann_query_base_gt100
  PREFIX=bigann_1b
  DATA_TYPE=uint8
  DIST_FN=l2
  B=32
  K=10
  DATA_DIM=128
  DATA_N=1000000000
}
```
For Recall@10, the config_local.sh should be set as follows.
```eval
# KNN
LS="35 40 50 60 80 100 125 150 175 200 225 250 275 300"
```

#### GIST1M-DiskANN:
```eval
 ./apps/search_disk_index  --data_type float --dist_fn l2 --index_path_prefix data/gist/disk_index_gist_base_R64_L100_A1.2 --query_file data/bigann/bigann_query.bbin  --gt_file data/gist/gist_query_base_gt100 -K 1 -L 15 20 25 30 35 40 45 50 75 100 125 150 175 200 250 300 350 400 --result_path data/gist/res --num_nodes_to_cache 10000
 ./apps/search_disk_index  --data_type float --dist_fn l2 --index_path_prefix data/gist/disk_index_gist_base_R64_L100_A1.2 --query_file data/bigann/bigann_query.bbin  --gt_file data/gist/gist_query_base_gt100 -K 10 -L 30 40 50 60 80 100 125 `50 175 200 250 300 --result_path data/gist/res --num_nodes_to_cache 10000
```

#### GIST1M-SPANN:
Since the multi-round execution of SPANN is relatively complex, only the values of `InternalResultNum` in the `[SearchSSDIndex]` section of `buildconfig.ini` are demonstrated here.
```eval
Recall@1: 30 40 50 75 100 150 200 250 300 350 400
Recall@10: 55 60 80 100 125 150 175 200 250 300 400 500
```

#### INDEX10M-DiskANN:
```eval
 ./apps/search_disk_index  --data_type float --dist_fn l2 --index_path_prefix data/index10m/disk_index_data_learn_R64_L100_A1.2 --query_file data/index10m/query.fbin  --gt_file data/index10m/index_query_learn_gt100 -K 1 -L 5 7 9 10 15 20 30 40 50 60 80 100 150 200 250 300 400 --result_path data/index10m/res --num_nodes_to_cache 10000
 ./apps/search_disk_index  --data_type float --dist_fn l2 --index_path_prefix data/index10m/disk_index_data_learn_R64_L100_A1.2 --query_file data/index10m/query.fbin  --gt_file data/index10m/index_query_learn_gt100 -K 10 -L 12 14 16 18 20 25 30 40 50 60 80 100 150 200 250 300 400 --result_path data/index10m/res --num_nodes_to_cache 10000
```

#### INDEX10M-SPANN:
Since the multi-round execution of SPANN is relatively complex, only the values of `InternalResultNum` in the `[SearchSSDIndex]` section of `buildconfig.ini` are demonstrated here.
```eval
Recall@1: 10 15 20 25 30 40 50 75 100 150 200 250 300 400
Recall@10: 12 13 16 18 20 25 30 35 40 60 80 100 200 250 300 350 400 500
```

#### INDEX10M-starling:

```eval
./run_benchmark.sh release search knn
```
For Recall@1, the config_sample.sh should be set as follows.
```eval
dataset_index_10m() {
  BASE_PATH=data/index10m/index.fbin
  QUERY_FILE=data/index10m/query.fbin
  GT_FILE=data/index10m/query_gt.bin
  PREFIX=index2_10m
  DATA_TYPE=float
  DIST_FN=l2
  B=1.6
  K=10
  DATA_DIM=512
  DATA_N=10000000
}
```
For Recall@1, the config_local.sh should be set as follows.
```eval
# KNN
LS="3 5 7 9 10 15 20 30 40 60 100 150 200 300"
```

For Recall@10, the config_sample.sh should be set as follows.
```eval
dataset_index_10m() {
  BASE_PATH=data/index10m/index.fbin
  QUERY_FILE=data/index10m/query.fbin
  GT_FILE=data/index10m/query_gt.bin
  PREFIX=index2_10m
  DATA_TYPE=float
  DIST_FN=l2
  B=1.6
  K=1
  DATA_DIM=512
  DATA_N=10000000
}
```
For Recall@10, the config_local.sh should be set as follows.
```eval
# KNN
LS="10 12 14 15 16 18 20 30 40 50 75 100 125 150 175 200 250 300"
```

