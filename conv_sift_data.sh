set -x
mkdir -p data/sift
./bin/convert_vecs2dim_vecs /mnt/SIFT/sift/sift_base.fvecs data/sift/sift_base.dim.fvecs 1000 0 0
./bin/convert_vecs2dim_vecs /mnt/SIFT/sift/sift_groundtruth.ivecs data/sift/sift_groundtruth.dim.ivecs 1000 0 0
./bin/convert_vecs2dim_vecs /mnt/SIFT/sift/sift_learn.fvecs data/sift/sift_learn.dim.fvecs  1000 0 0
./bin/convert_vecs2dim_vecs /mnt/SIFT/sift/sift_query.fvecs data/sift/sift_query.dim.fvecs  1000 0 0