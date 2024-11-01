CXX = g++
OUT_PATH = ./
CFLAGS = -g -std=c++11 -Wall -O3 -march=native -mtune=native -flto -ggdb -pipe -W -fPIC -std=c++11 -ffast-math -mavx -mfma
FFLAGS = -lpthread -fopenmp -lssl -lrt -lz
INCLUDE = -Ieigen-3.4.0/


run_test_set: run_test_set.cpp common.h hierachical_cluster.o kmeans.o random.o unity.o Log.o file_read_write.o conf.o common.h
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/run_test_set run_test_set.cpp hierachical_cluster.o kmeans.o random.o unity.o Log.o file_read_write.o conf.o ../env/tcmalloc/lib/libtcmalloc.a $(FFLAGS)
convert_vecs2dim_vecs:convert_vecs2dim_vecs.cpp file_read_write.h common.h file_read_write.o
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/convert_vecs2dim_vecs convert_vecs2dim_vecs.cpp file_read_write.o
disk_hivf_train_and_build: disk_hivf_train_and_build.cpp common.h hierachical_cluster.o kmeans.o random.o unity.o Log.o file_read_write.o conf.o common.h
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/disk_hivf_train_and_build disk_hivf_train_and_build.cpp hierachical_cluster.o kmeans.o random.o unity.o Log.o file_read_write.o conf.o ../env/tcmalloc/lib/libtcmalloc.a $(FFLAGS)
test_hierachical_cluster: disk_hivf_train_and_build.cpp common.h hierachical_cluster.o kmeans.o random.o unity.o Log.o file_read_write.o conf.o common.h
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/test_hierachical_cluster test_hierachical_cluster.cpp hierachical_cluster.o kmeans.o random.o unity.o Log.o file_read_write.o conf.o ../env/tcmalloc/lib/libtcmalloc.a $(FFLAGS)
test_kmeans:test_kmeans.cpp kmeans.o random.o common.h
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/test_kmeans test_kmeans.cpp kmeans.o random.o ../env/tcmalloc/lib/libtcmalloc.a $(FFLAGS)
test_eigen3:test_eigen3.cpp
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/test_eigen3 test_eigen3.cpp $(FFLAGS)
test_eigen:test_eigen.cpp file_read_write.o common.h
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/test_eigen test_eigen.cpp file_read_write.o ../env/tcmalloc/lib/libtcmalloc.a $(FFLAGS)
test_read_vec_file:test_read_vec_file.cpp file_read_write.o common.h
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/test_read_vec_file test_read_vec_file.cpp file_read_write.o
testrand:testrand.cpp unity.o Log.o random.o common.h
	$(CXX) $(CFLAGS) $(INCLUDE) -o $(OUT_PATH)/testrand testrand.cpp unity.o Log.o random.o
hierachical_cluster.o: hierachical_cluster.cpp hierachical_cluster.h matrix.h kmeans.o random.o unity.o Log.o file_read_write.o conf.o
	$(CXX) $(CFLAGS) $(INCLUDE) -c hierachical_cluster.cpp kmeans.o random.o unity.o Log.o file_read_write.o conf.o $(FFLAGS)
file_read_write.o: file_read_write.cpp file_read_write.h unity.o common.h
	$(CXX) $(CFLAGS) $(INCLUDE) -c file_read_write.cpp unity.o $(FFLAGS)
kmeans.o: kmeans.cpp kmeans.h random.o common.h
	$(CXX) $(CFLAGS) $(INCLUDE) -c kmeans.cpp random.o $(FFLAGS)
random.o: random.cpp random.h
	$(CXX) $(CFLAGS) $(INCLUDE) -c random.cpp
conf.o: conf.cpp conf.h unity.o
	$(CXX) $(CFLAGS) $(INCLUDE) -c conf.cpp unity.o
unity.o:unity.cpp unity.h Log.o
	$(CXX) $(CFLAGS) $(INCLUDE) -c unity.cpp Log.o
Log.o:Log.cpp Log.h
	$(CXX) $(CFLAGS) -c Log.cpp

