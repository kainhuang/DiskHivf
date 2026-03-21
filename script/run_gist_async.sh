set -x
# 异步IO模式 - 使用 gist_async.conf 配置，方便与同步模式 (run_gist.sh) 做对比
./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 1 1 1 20 200 0 0 0 1
./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 1 1 1 20 300 0 0 0 1
./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 1 1 1 40 500 0 0 0 1
./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 1 1 1 60 700 0 0 0 1
./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 1 1 1 60 1000 0 0 0 1
./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 1 1 1 60 1500 0 0 0 1
./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 1 1 1 60 2000 0 0 0 1
./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 1 1 1 60 2500 0 0 0 1
#./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 1 1 1 60 3000 0 0 0 1
#./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 1 1 1 60 3500 0 0 0 1
#./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 1 1 1 70 4000 0 0 0 1
#./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 1 1 1 70 4500 0 0 0 1
#./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 1 1 1 70 5000 0 0 0 1

./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 10 10 1 20 200 0 0 0 1
./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 10 10 1 20 300 0 0 0 1
./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 10 10 1 40 500 0 0 0 1
./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 10 10 1 60 700 0 0 0 1
./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 10 10 1 60 1000 0 0 0 1
./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 10 10 1 60 1500 0 0 0 1
./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 10 10 1 60 2000 0 0 0 1
./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 10 10 1 60 2500 0 0 0 1
#./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 10 10 1 60 3000 0 0 0 1
#./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 10 10 1 60 3500 0 0 0 1
#./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 10 10 1 70 4000 0 0 0 1
#./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 10 10 1 70 4500 0 0 0 1
#./bin/run_test_set conf/gist_async.conf data/gist/gist_query.dim.fvecs data/gist/gist_query_base_gt100.dim.ivecs 10 10 1 70 5000 0 0 0 1
