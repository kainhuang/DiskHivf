set -x
./bin/convert_vecs2dim_vecs  /mnt/vdc/BIGANN/bigann_base.bvecs data/bigann/bigann_base.dim.bvecs 10000 0 0
./bin/convert_vecs2dim_vecs /root/home/data/BIGANN/gnd/idx_1000M.ivecs  data/bigann/gnd/idx_1000M.dim.ivecs 10000 0 0
./bin/convert_vecs2dim_vecs /root/home/data/BIGANN/gnd/dis_1000M.fvecs  data/bigann/gnd/dis_1000M.dim.fvecs 10000 0 0
./bin/convert_vecs2dim_vecs /mnt/vdc/BIGANN/bigann_query.bvecs data/bigann/bigann_query.dim.fvecs 1000 1 0 
./bin/rand_train_set data/bigann/bigann_base.dim.bvecs  ./data/bigann/bigann_learn.dim.bvecs 100000000 1