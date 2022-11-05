#!/bin/bash
code_dir=$1
dataset=$2
backbone=$3
save_results="False"
save_models="False"
p=0
while(($p<10))
do
    k=0
    while(($k<5))
    do
        echo "------- This is p$p-k$k time, dataset is $dataset, backbone is $backbone. -------"
        python $code_dir \
            --dataset $dataset \
            --p_value $p \
            --k_value $k \
            --backbone $backbone \
            --save_results $save_results \
            --save_models $save_models
        let "k++"
    done
    let "p++"
done
