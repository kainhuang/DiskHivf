
./bin/test_search $1 $2 1 200 5000 |grep NOTICE > tmp_learn_hp_data
cat tmp_learn_hp_data|grep m_build_index_loss > learn_hp_data
sort -g -k7,7 tmp_learn_hp_data |grep -v m_build_index_loss >> learn_hp_data
rm tmp_learn_hp_data
python3 p_list.py