#!/bin/bash
cd data
for i in `seq 0 0`
do
    python ../dataset_convert.py -dataset sim_1_tiny -id $i -mr 0.3 -size 5000 -func load
    
    python ../impute.py --input_file train_test_split/sim_1_tiny_train.tsv --output_file imputations/sim_1_tiny_imputed.tsv --one_hot_max_sizes 3 2 3 4 --num_imputations 10 --epochs 50 --validation_ratio 0.15
    #python evaluate_results.py sim_1_tiny 3 2 3 4
    
    #python ../impute.py --input_file train_test_split/acs_train.tsv --output_file imputations/acs_imputed.tsv --one_hot_max_sizes 3 4 3 3 2 4 4 4 2 2 7 2 2 2 2 3 2 2 3 4 4 9 4 2 2 2 3 3 4 4 3 2 8 2 2 2 2 2 1 1 1 1 1 1 1 1 --num_imputations 10 --epochs 7 --validation_ratio 0.15
    #python evaluate_results.py acs 3 4 3 3 2 4 4 4 2 2 7 2 2 2 2 3 2 2 3 4 4 9 4 2 2 2 3 3 4 4 3 2 8 2 2 2 2 2 1 1 1 1 1 1 1 1
    
    python ../dataset_convert.py -dataset sim_1_tiny -id $i -mr 0.3 -size 5000 -func convert
done

#python ../impute.py --input_file train_test_split/boston_train.tsv --output_file imputations/boston_imputed.tsv --one_hot_max_sizes 1 9 1 1 1 1 1 1 1 1 1 1 1 1 --num_imputations 10 --epochs 100 --validation_ratio 0.15
#python evaluate_results.py boston 1 9 1 1 1 1 1 1 1 1 1 1 1 1

#python ../impute.py --input_file train_test_split/yeast_train.tsv --output_file imputations/yeast_imputed.tsv --one_hot_max_sizes 1 1 1 1 1 1 1 1 10 --num_imputations 10 --epochs 300 --validation_ratio 0.15
#python evaluate_results.py yeast 1 1 1 1 1 1 1 1 10

#python ../impute.py --input_file train_test_split/acs_train.tsv --output_file imputations/acs_imputed.tsv --one_hot_max_sizes 3 4 3 3 2 4 4 4 2 2 7 2 2 2 2 3 2 2 3 4 4 9 4 2 2 2 3 3 4 4 3 2 8 2 2 2 2 2 1 1 1 1 1 1 1 1 --num_imputations 10 --epochs 7 --validation_ratio 0.15
#python evaluate_results.py acs 3 4 3 3 2 4 4 4 2 2 7 2 2 2 2 3 2 2 3 4 4 9 4 2 2 2 3 3 4 4 3 2 8 2 2 2 2 2 1 1 1 1 1 1 1 1

#python ../impute.py --input_file train_test_split/credit_train.tsv --output_file imputations/credit_imputed.tsv --one_hot_max_sizes 2 7 4 11 11 11 11 10 10 1 1 1 1 1 1 1 1 1 1 1 1 1 1 --num_imputations 10 --epochs 5 --validation_ratio 0.15
#python evaluate_results.py credit 2 7 4 11 11 11 11 10 10 1 1 1 1 1 1 1 1 1 1 1 1 1 1

cd ..
echo 'finish'