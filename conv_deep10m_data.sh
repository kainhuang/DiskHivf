set -x
mkdir -p data/deep10m/
./bin/convert_fbin2dim_vecs /mnt/DEEP/base.10M.fbin data/deep10m/base.10M.dim.fvecs 10000 0
./bin/convert_bin2dim_ivecs /mnt/DEEP/deep_query_base10M_gt100 ./data/deep10m/deep_query_base10M_gt100.dim.ivecs 10000
./bin/convert_fbin2dim_vecs /mnt/DEEP/query.public.10K.fbin ./data/deep10m/query.public.10K.dim.fvecs 1000 0
./bin/rand_train_set data/deep10m/base.10M.dim.fvecs data/deep10m/learn.1M.dim.fvecs 1000000 0