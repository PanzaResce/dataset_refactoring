#!/bin/bash

GPU_NUMBER=0
LOWER_CASE='True'
ACCUMULATION_STEPS=1
TASK='unfair_tos'

MODEL=('prajjwal1/bert-mini')
CATEGORIES=('a' 'ch' 'cr' 'j' 'law' 'ltd' 'ter' 'use' 'pinc')
EPOCHS=10
NUM_SEEDS=5
BATCH_SIZE=4

for CAT in "${CATEGORIES[@]}"; do
    for SEED in $(seq 1 ${NUM_SEEDS}); do
        echo "-------- Training model $MODEL with seed $SEED on category <$CAT> --------"
        CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python unfair_tos.py \
        --model_name_or_path ${MODEL} \
        --do_lower_case ${LOWER_CASE} \
        --output_dir logs/${TASK}_head/${MODEL}/${CAT}/seed_${SEED}\
        --do_train \
        --do_eval \
        --do_pred \
        --overwrite_output_dir \
        --load_best_model_at_end \
        --metric_for_best_model macro-f1 \
        --greater_is_better True \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 3 \
        --num_train_epochs ${EPOCHS} \
        --learning_rate 3e-5 \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --per_device_eval_batch_size ${BATCH_SIZE} \
        --seed ${SEED} \
        --fp16 \
        --fp16_full_eval \
        --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
        --eval_accumulation_steps ${ACCUMULATION_STEPS} \
        --binary False \
        --head_category ${CAT}
    done
done

python statistics/compute_avg_scores.py --dataset ${TASK}
