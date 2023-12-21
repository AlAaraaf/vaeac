#!/bin/bash

lr_list=(1e-4 5e-3 2e-3 1e-3)
l2reg=(2e-4 1e-4 2e-3)
depth_list=(6 10 12)
model_name="vaeac"
dataset=("sim1" "sim2" "sim3" "sim4" "boston" "credit" "nhanes" "house")
sample_id=0
mr=0.3
sample_size=(10000 10000 10000 10000 500 10000 10000 10000)
miss_loc=(5 5 5 5 1 3 4 14)
batch_size=512

for data_i in `seq 0 7`
do
    for d_i in `seq 0 2`
    do
        for lr_i in `seq 0 3`
        do
            for l2_i in `seq 0 2`
            do
                cd data
                
                python ../dataset_convert.py -dataset ${dataset[$data_i]} -id $sample_id -mr $mr -size ${sample_size[$data_i]} -func load

                if [ $data_i -le 3 ]; then
                    python ../impute.py --input_file train_test_split/${dataset[$data_i]}_train.tsv --output_file imputations/${dataset[$data_i]}_imputed.tsv \
                    --one_hot_max_sizes 2 3 5 8 10 2 \
                    --num_imputations 10 \
                    --batch_size $batch_size \
                    --epochs 20 \
                    --validation_ratio 0.15 \
                    --miss_loc ${miss_loc[$data_i]} \
                    --lr ${lr_list[$lr_i]} \
                    --l2reg ${l2reg[$l2_i]} \
                    --depth ${depth_list[$d_i]} \
                    --log_name vaeac_income_${lr_i}_${l2_i}/tuning/
                elif [ $data_i == 4 ]; then
                    python ../impute.py --input_file train_test_split/${dataset[$data_i]}_train.tsv --output_file imputations/${dataset[$data_i]}_imputed.tsv \
                    --one_hot_max_sizes 2 9 1 1 1 1 1 1 1 1 1 1 1 1 \
                    --num_imputations 10 \
                    --batch_size $batch_size \
                    --epochs 10 \
                    --validation_ratio 0.15 \
                    --miss_loc ${miss_loc[$data_i]} \
                    --lr ${lr_list[$lr_i]} \
                    --l2reg ${l2reg[$l2_i]} \
                    --depth ${depth_list[$d_i]} \
                    --log_name vaeac_income_${lr_i}_${l2_i}/tuning/
                elif [ $data_i == 5 ]; then
                    python ../impute.py --input_file train_test_split/${dataset[$data_i]}_train.tsv --output_file imputations/${dataset[$data_i]}_imputed.tsv \
                    --one_hot_max_sizes 2 7 4 11 11 11 11 10 10 1 1 1 1 1 1 1 1 1 1 1 1 1 1 \
                    --num_imputations 10 \
                    --batch_size $batch_size \
                    --epochs 20 \
                    --validation_ratio 0.15 \
                    --miss_loc ${miss_loc[$data_i]} \
                    --lr ${lr_list[$lr_i]} \
                    --l2reg ${l2reg[$l2_i]} \
                    --depth ${depth_list[$d_i]} \
                    --log_name vaeac_income_${lr_i}_${l2_i}/tuning/
                elif [ $data_i == 6 ]; then
                    python ../impute.py --input_file train_test_split/${dataset[$data_i]}_train.tsv --output_file imputations/${dataset[$data_i]}_imputed.tsv \
                    --one_hot_max_sizes 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 \
                    --num_imputations 10 \
                    --batch_size $batch_size \
                    --epochs 20 \
                    --validation_ratio 0.15 \
                    --miss_loc ${miss_loc[$data_i]} \
                    --lr ${lr_list[$lr_i]} \
                    --l2reg ${l2reg[$l2_i]} \
                    --depth ${depth_list[$d_i]} \
                    --log_name vaeac_income_${lr_i}_${l2_i}/tuning/
                else
                    python ../impute.py --input_file train_test_split/${dataset[$data_i]}_train.tsv --output_file imputations/${dataset[$data_i]}_imputed.tsv \
                    --one_hot_max_sizes 3 4 3 3 2 4 4 4 2 2 7 2 2 2 2 3 2 2 3 4 4 9 4 2 2 2 3 3 4 4 3 2 8 2 2 2 2 2 1 1 1 1 1 1 1 1 \
                    --num_imputations 10 \
                    --batch_size $batch_size \
                    --epochs 30 \
                    --validation_ratio 0.15 \
                    --miss_loc ${miss_loc[$data_i]} \
                    --lr ${lr_list[$lr_i]} \
                    --l2reg ${l2reg[$l2_i]} \
                    --depth ${depth_list[$d_i]} \
                    --log_name vaeac_income_${lr_i}_${l2_i}/tuning/
                fi

                python ../dataset_convert.py -dataset ${dataset[$data_i]} -id $sample_id -mr $mr -size ${sample_size[$data_i]} -func convert
                
                cd ../../MissingData_DL/
                python ./eval_main.py -id $sample_id \
                -dataset ${dataset[$data_i]} \
                -model $model_name \
                -mr $mr \
                -size ${sample_size[$data_i]} \
                -batch_size $batch_size \
                -alpha 1 \
                -iterations 1 \
                -dlr 1 \
                -glr 1 \
                -d_gradstep 1 \
                -g_gradstep 1 \
                -onlylog 1 \
                -prefix ${batch_size}_${depth_list[$d_i]}_${lr_list[$lr_i]}_${l2reg[$l2_i]}.npy
                echo 'finish'
                cd ../vaeac/
            done
        done
    done
done