#!/bin/bash
cd data
python prepare_data.py
mkdir -p imputations
python ../impute.py --input_file train_test_split/boston_train.tsv\ 
--output_file imputations/boston_imputed.tsv\ 
--one_hot_max_sizes 1 9 1 1 1 1 1 1 1 1 1 1 1 1\ 
--num_imputations 10 --epochs 300 --validation_ratio 0.15
python evaluate_results.py boston 1 9 1 1 1 1 1 1 1 1 1 1 1 1
cd ..
echo 'finish'