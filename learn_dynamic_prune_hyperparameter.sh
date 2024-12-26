./bin/test_search $1 $2 $3 $4 5000 |grep NOTICE| sort -n -k7 > learn_hp_data
python p_list.py