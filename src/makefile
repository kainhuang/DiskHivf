CXX = g++
OUT_PATH = ../bin
CFLAGS = -g -std=c++11 -Wall -O3 -march=native -mtune=native -ggdb -pipe -W -fPIC -ffast-math -mavx -mfma 
FFLAGS = -lpthread -fopenmp -lssl -lrt -lz -lboost_thread -lboost_system
#FFLAGS = -lpthread -fopenmp -lssl -lrt -lz -L/opt/intel/oneapi/mkl/2024.0/lib -lmkl_rt
INCLUDE = -Ieigen-3.4.0/
#INCLUDE = -Ieigen-3.4.0/ -I/opt/intel/oneapi/mkl/2024.0/include

filter:filter.cpp file_read_write.h matrix.h common.h file_read_write.o random.o
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/filter filter.cpp file_read_write.o random.o $(FFLAGS)
make_gt:make_gt.cpp file_read_write.h matrix.h common.h file_read_write.o random.o
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/make_gt make_gt.cpp file_read_write.o random.o $(FFLAGS)
rand_train_set:rand_train_set.cpp file_read_write.h common.h file_read_write.o random.o
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/rand_train_set rand_train_set.cpp file_read_write.o random.o $(FFLAGS)
convert_bin2ivecs:convert_bin2ivecs.cpp file_read_write.h common.h file_read_write.o
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/convert_bin2ivecs convert_bin2ivecs.cpp file_read_write.o $(FFLAGS)
fbin_top_n_vecs:fbin_top_n_vecs.cpp file_read_write.h common.h file_read_write.o
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/fbin_top_n_vecs fbin_top_n_vecs.cpp file_read_write.o $(FFLAGS)
convert_bin2dim_ivecs:convert_bin2dim_ivecs.cpp file_read_write.h common.h file_read_write.o
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/convert_bin2dim_ivecs convert_bin2dim_ivecs.cpp file_read_write.o $(FFLAGS)
convert_fbin2dim_vecs:convert_fbin2dim_vecs.cpp file_read_write.h common.h file_read_write.o
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/convert_fbin2dim_vecs convert_fbin2dim_vecs.cpp file_read_write.o $(FFLAGS)
test_search: test_search.cpp common.h lru_cache.h hierachical_cluster.o kmeans.o matrix.o random.o unity.o file_read_write.o conf.o common.h thread_pool.o lru_cache.o
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/test_search test_search.cpp hierachical_cluster.o kmeans.o matrix.o random.o unity.o file_read_write.o conf.o thread_pool.o lru_cache.o ../env/tcmalloc/lib/libtcmalloc.a $(FFLAGS)
build_index: build_index.cpp common.h hierachical_cluster.o kmeans.o random.o unity.o file_read_write.o conf.o common.h matrix.o thread_pool.o lru_cache.o
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/build_index build_index.cpp hierachical_cluster.o kmeans.o matrix.o random.o unity.o file_read_write.o conf.o thread_pool.o lru_cache.o ../env/tcmalloc/lib/libtcmalloc.a $(FFLAGS)
run_test_set: run_test_set.cpp common.h hierachical_cluster.o kmeans.o matrix.o random.o unity.o file_read_write.o conf.o common.h thread_pool.o lru_cache.o
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/run_test_set run_test_set.cpp hierachical_cluster.o kmeans.o matrix.o random.o unity.o file_read_write.o conf.o thread_pool.o lru_cache.o ../env/tcmalloc/lib/libtcmalloc.a $(FFLAGS)
convert_vecs2dim_vecs:convert_vecs2dim_vecs.cpp file_read_write.h common.h file_read_write.o
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/convert_vecs2dim_vecs convert_vecs2dim_vecs.cpp file_read_write.o
disk_hivf_train_and_build: disk_hivf_train_and_build.cpp common.h hierachical_cluster.o kmeans.o random.o unity.o file_read_write.o conf.o common.h matrix.o thread_pool.o lru_cache.o
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/disk_hivf_train_and_build disk_hivf_train_and_build.cpp hierachical_cluster.o kmeans.o matrix.o random.o unity.o file_read_write.o conf.o thread_pool.o lru_cache.o ../env/tcmalloc/lib/libtcmalloc.a $(FFLAGS)
test_hierachical_cluster: disk_hivf_train_and_build.cpp common.h hierachical_cluster.o kmeans.o random.o unity.o file_read_write.o conf.o common.h lru_cache.o
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/test_hierachical_cluster test_hierachical_cluster.cpp hierachical_cluster.o kmeans.o random.o unity.o file_read_write.o conf.o lru_cache.o ../env/tcmalloc/lib/libtcmalloc.a $(FFLAGS)
test_kmeans:test_kmeans.cpp kmeans.o random.o common.h
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/test_kmeans test_kmeans.cpp kmeans.o random.o ../env/tcmalloc/lib/libtcmalloc.a $(FFLAGS)
test_eigen3:test_eigen3.cpp
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/test_eigen3 test_eigen3.cpp $(FFLAGS)
test_eigen:test_eigen.cpp file_read_write.o common.h
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/test_eigen test_eigen.cpp file_read_write.o ../env/tcmalloc/lib/libtcmalloc.a $(FFLAGS)
test_read_vec_file:test_read_vec_file.cpp file_read_write.o common.h
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/test_read_vec_file test_read_vec_file.cpp file_read_write.o
testrand:testrand.cpp unity.o Log.o random.o common.h
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/testrand testrand.cpp unity.o random.o
test_thread_pool: test_thread_pool.cpp thread_pool.o
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/test_thread_pool test_thread_pool.cpp thread_pool.o $(FFLAGS)
hierachical_cluster.o: hierachical_cluster.cpp hierachical_cluster.h matrix.o kmeans.o random.o unity.o file_read_write.o conf.o lru_cache.o
	$(CXX) $(CFLAGS) $(INCLUDE) -c hierachical_cluster.cpp kmeans.o matrix.o random.o unity.o file_read_write.o conf.o lru_cache.o $(FFLAGS)
file_read_write.o: file_read_write.cpp file_read_write.h unity.o common.h
	$(CXX) $(CFLAGS) $(INCLUDE) -c file_read_write.cpp unity.o $(FFLAGS)
kmeans.o: kmeans.cpp kmeans.h random.o common.h
	$(CXX) $(CFLAGS) $(INCLUDE) -c kmeans.cpp random.o $(FFLAGS)
lru_cache.o: lru_cache.cpp lru_cache.h common.h
	$(CXX) $(CFLAGS) $(INCLUDE) -c lru_cache.cpp $(FFLAGS)
matrix.o:matrix.cpp matrix.h common.h 
	$(CXX) $(CFLAGS) $(INCLUDE) -c matrix.cpp
random.o: random.cpp random.h
	$(CXX) $(CFLAGS) $(INCLUDE) -c random.cpp
conf.o: conf.cpp conf.h unity.o
	$(CXX) $(CFLAGS) $(INCLUDE) -c conf.cpp unity.o
unity.o:unity.cpp unity.h
	$(CXX) $(CFLAGS) $(INCLUDE) -c unity.cpp
thread_pool.o: thread_pool.cpp thread_pool.h
	$(CXX) $(CFLAGS) $(INCLUDE) -c thread_pool.cpp $(FFLAGS)
clean:
	rm *.o $(OUT_PATH)/disk_hivf_train_and_build $(OUT_PATH)/run_test_set $(OUT_PATH)/convert_vecs2dim_vecs $(OUT_PATH)/build_index $(OUT_PATH)/convert_fbin2dim_vecs
