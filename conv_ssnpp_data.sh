set -x
mkdir -p data/ssnpp
./bin/convert_fbin2dim_vecs /mnt/Facebook/FB_ssnpp_public_queries.u8bin data/ssnpp/FB_ssnpp_public_queries.dim.bvecs 1000 1
./bin/convert_fbin2dim_vecs /mnt/Facebook/FB_ssnpp_database.u8bin data/ssnpp/FB_ssnpp_database.dim.bvecs 10000 1
./bin/rand_train_set data/ssnpp/FB_ssnpp_database.dim.bvecs  ./data/ssnpp/FB_ssnpp_database_learn.dim.bvecs 25000000 1
./bin/convert_bin2dim_ivecs /mnt/Facebook/FB_ssnpp_queries_base_gt100 ./data/ssnpp/FB_ssnpp_queries_base_gt100.dim.ivecs 10000