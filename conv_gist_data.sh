set -x
./bin/convert_vecs2dim_vecs /root/home/data/GIST/gist/gist_base.fvecs data/gist/gist_base.dim.fvecs 1000 0 0
./bin/convert_vecs2dim_vecs /root/home/data/GIST/gist/gist_groundtruth.ivecs data/gist/gist_groundtruth.dim.ivecs 1000 0 0
./bin/convert_vecs2dim_vecs /root/home/data/GIST/gist/gist_learn.fvecs data/gist/gist_learn.dim.fvecs  1000 0 0
./bin/convert_vecs2dim_vecs /root/home/data/GIST/gist/gist_query.fvecs data/gist/gist_query.dim.fvecs  1000 0 0