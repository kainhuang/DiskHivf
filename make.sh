mkdir -p bin
cd src
# make clean
make run_test_set disk_hivf_train_and_build convert_vecs2dim_vecs build_index \
    test_search convert_fbin2dim_vecs rand_train_set convert_bin2dim_ivecs \
    fbin_top_n_vecs convert_bin2ivecs make_gt filter
