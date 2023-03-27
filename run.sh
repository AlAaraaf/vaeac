#!/bin/bash
lr_list=(2e-4 1e-4 1e-3 2e-3 5e-3 1e-2)
l2reg=(2e-5 2e-4 1e-4 2e-3 1e-3)
depth_list=(8 10 12 15 24)
model_name="vaeac"
dataset="income"
sample_id=$1
mr=0.3
sample_size=10000

for d_i in `seq 0 4`
do
    for lr_i in `seq 0 5`
    do
        for l2_i in `seq 0 4`
        do
            cd data
            # python ../dataset_convert.py -dataset house -id $sample_id -mr $mr -size $sample_size -func load
            
            python ../dataset_convert.py -dataset insome -id $sample_id -mr $mr -size $sample_size -func load

            # python ../impute.py --input_file train_test_split/house_train.tsv --output_file imputations/house_imputed.tsv \
            # --one_hot_max_sizes 3 4 3 3 2 4 4 4 2 2 7 2 2 2 2 3 2 2 3 4 4 9 4 2 2 2 3 3 4 4 3 2 8 2 2 2 2 2 1 1 1 1 1 1 1 1 \
            # --num_imputations 10 \
            # --epochs 1 \
            # --validation_ratio 0.15 \
            # --lr ${lr_list[$lr_i]} \
            # --l2reg ${l2reg[$l2_i]} \
            # --depth ${depth_list[$d_i]} \
            # --log_name vaeac_house_${lr_i}_${l2_i}/tuning/
            # #python evaluate_results.py sim_1 3 2 3 4

            python ../impute.py --input_file train_test_split/income_train.tsv --output_file imputations/house_imputed.tsv \
            --one_hot_max_sizes 4 2 2 2 3 3 4 4 3 1 2 2 4 2 3 3 1 8 1 1 2 \
            --num_imputations 10 \
            --epochs 50 \
            --validation_ratio 0.15 \
            --lr ${lr_list[$lr_i]} \
            --l2reg ${l2reg[$l2_i]} \
            --depth ${depth_list[$d_i]} \
            --log_name vaeac_income_${lr_i}_${l2_i}/tuning/
            #python evaluate_results.py sim_1 3 2 3 4 

            #python ../impute.py --input_file train_test_split/acs_train.tsv --output_file imputations/acs_imputed.tsv --one_hot_max_sizes 3 4 3 3 2 4 4 4 2 2 7 2 2 2 2 3 2 2 3 4 4 9 4 2 2 2 3 3 4 4 3 2 8 2 2 2 2 2 1 1 1 1 1 1 1 1 --num_imputations 10 --epochs 7 --validation_ratio 0.15
            #python evaluate_results.py acs 3 4 3 3 2 4 4 4 2 2 7 2 2 2 2 3 2 2 3 4 4 9 4 2 2 2 3 3 4 4 3 2 8 2 2 2 2 2 1 1 1 1 1 1 1 1

            # python ../dataset_convert.py -dataset house -id $sample_id -mr $mr -size $sample_size -func convert

            python --/dataset_convert.py -dataset income -id $sample_id -mr $mr -size $sample_size -fun convert
            #python ../impute.py --input_file train_test_split/boston_train.tsv --output_file imputations/boston_imputed.tsv --one_hot_max_sizes 1 9 1 1 1 1 1 1 1 1 1 1 1 1 --num_imputations 10 --epochs 100 --validation_ratio 0.15
            #python evaluate_results.py boston 1 9 1 1 1 1 1 1 1 1 1 1 1 1

            #python ../impute.py --input_file train_test_split/yeast_train.tsv --output_file imputations/yeast_imputed.tsv --one_hot_max_sizes 1 1 1 1 1 1 1 1 10 --num_imputations 10 --epochs 300 --validation_ratio 0.15
            #python evaluate_results.py yeast 1 1 1 1 1 1 1 1 10

            #python ../impute.py --input_file train_test_split/acs_train.tsv --output_file imputations/acs_imputed.tsv --one_hot_max_sizes 3 4 3 3 2 4 4 4 2 2 7 2 2 2 2 3 2 2 3 4 4 9 4 2 2 2 3 3 4 4 3 2 8 2 2 2 2 2 1 1 1 1 1 1 1 1 --num_imputations 10 --epochs 7 --validation_ratio 0.15
            #python evaluate_results.py acs 3 4 3 3 2 4 4 4 2 2 7 2 2 2 2 3 2 2 3 4 4 9 4 2 2 2 3 3 4 4 3 2 8 2 2 2 2 2 1 1 1 1 1 1 1 1

            #python ../impute.py --input_file train_test_split/credit_train.tsv --output_file imputations/credit_imputed.tsv --one_hot_max_sizes 2 7 4 11 11 11 11 10 10 1 1 1 1 1 1 1 1 1 1 1 1 1 1 --num_imputations 10 --epochs 5 --validation_ratio 0.15
            #python evaluate_results.py credit 2 7 4 11 11 11 11 10 10 1 1 1 1 1 1 1 1 1 1 1 1 1 1

            cd ../../MissingData_DL/
            python ./calculate_estimands.py -dataset $dataset -model $model_name -num 1 -mr $mr -size $sample_size -completedir ../training_data/samples/${dataset}/complete_${mr}_${sample_size}/ -missingdir ../training_data/samples/${dataset}/MCAR_${mr}_${sample_size}/ -imputedir ../training_data/results/${dataset}/MCAR_${mr}_${sample_size}/${model_name}/
            python ./evaluate_estimands.py -dataset $dataset -model $model_name
            python ./show_tables.py -dataset $dataset -output ../metrics/${model_name}_${mr}_${sample_size}_${sample_id}_${depth_list[$d_i]}_${lr_list[$lr_i]}_${l2reg[$l2_i]}
            echo 'finish'
            cd ../vaeac/
        done
    done
done