set -x
./bin/convert_fbin2dim_vecs /root/home/data/DEEP/base.1B.fbin data/deep/base.1B.dim.fvecs 10000
./bin/convert_bin2dim_ivecs /root/home/data/DEEP/deep_new_groundtruth.public.10K.bin ./data/deep/deep_new_groundtruth.public.10K.dim.ivecs 10000
./bin/convert_fbin2dim_vecs /root/home/data/DEEP/query.public.10K.fbin ./data/deep/query.public.10K.dim.fvecs 1000
./bin/rand_train_set data/deep/base.1B.dim.fvecs data/deep/learn.100M.dim.fvecs 100000000 0