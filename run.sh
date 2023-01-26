#!/bin/bash
cd data
python ../impute.py --input_file train_test_split/boston_train.tsv --output_file imputations/boston_imputed.tsv --one_hot_max_sizes 1 9 1 1 1 1 1 1 1 1 1 1 1 1 --num_imputations 10 --epochs 300 --validation_ratio 0.15
python evaluate_results.py boston 1 9 1 1 1 1 1 1 1 1 1 1 1 1

#python ../impute.py --input_file train_test_split/yeast_train.tsv --output_file imputations/yeast_imputed.tsv --one_hot_max_sizes 1 1 1 1 1 1 1 1 10 --num_imputations 10 --epochs 300 --validation_ratio 0.15
#python evaluate_results.py yeast 1 1 1 1 1 1 1 1 10

#python ../impute.py --input_file train_test_split/acs_train.tsv --output_file imputations/acs_imputed.tsv --one_hot_max_sizes 3 4 3 3 2 4 4 4 2 2 7 2 2 2 2 3 2 2 3 4 4 9 4 2 2 2 3 3 4 4 3 2 8 2 2 2 2 2 1 1 1 1 1 1 1 1 --num_imputations 10 --epochs 7 --validation_ratio 0.15
#python evaluate_results.py acs 3 4 3 3 2 4 4 4 2 2 7 2 2 2 2 3 2 2 3 4 4 9 4 2 2 2 3 3 4 4 3 2 8 2 2 2 2 2 1 1 1 1 1 1 1 1
cd ..
echo 'finish'