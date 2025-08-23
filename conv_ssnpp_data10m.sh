set -x
mkdir -p data/ssnpp10m
./bin/convert_fbin2dim_vecs /mnt/Facebook/FB_ssnpp_public_queries.u8bin data/ssnpp10m/FB_ssnpp_public_queries.dim.bvecs 1000 1
./bin/convert_fbin2dim_vecs /mnt/Facebook/FB_ssnpp_database10M.u8bin data/ssnpp10m/FB_ssnpp_database10M.dim.bvecs 10000 1
./bin/rand_train_set data/ssnpp10m/FB_ssnpp_database10M.dim.bvecs  ./data/ssnpp10m/FB_ssnpp_database1M_learn.dim.bvecs 1000000 1
./bin/convert_bin2dim_ivecs /mnt/Facebook/FB_ssnpp_queries_base10M_gt100 ./data/ssnpp10m/FB_ssnpp_queries_base10M_gt100.dim.ivecs 10000