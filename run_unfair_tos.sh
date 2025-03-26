#!/bin/bash

GPU_NUMBER=0
LOWER_CASE='True'
ACCUMULATION_STEPS=1
TASK='unfair_tos'
# MODELS=('nlpaueb/legal-bert-base-uncased' 'zlucia/custom-legalbert')
# MODELS=('bert-base-uncased' 'roberta-base' 'microsoft/deberta-base' 'allenai/longformer-base-4096' 'google/bigbird-roberta-base' 'nlpaueb/legal-bert-base-uncased' 'zlucia/custom-legalbert' 'roberta-large')

if [[ "$1" == "-h" ]]; then
  echo "Usage: $0 [MODEL_NAME] [EPOCHS] [BATCH_SIZE] [NUM_SEEDS] [BINARY] or $0 -all [EPOCHS] [NUM_SEEDS] [BINARY]"
  echo ""
  echo "Description:"
  echo "  This script trains and evaluates models on the unfair_tos task using specified parameters."
  echo ""
  echo "Options:"
  echo "  MODEL_NAME    Name of the model to be used"
  echo "  EPOCHS        Number of training epochs"
  echo "  BATCH_SIZE    Batch size for training and evaluation "
  echo "  NUM_SEEDS     Number of random seeds to run "
  echo "  BINARY        Train the model in binary mode instead of multi-label "
  echo "  -all          Runs the script for a set of predefined model names with NUM_SEEDS repetitions for each."
  echo "  -h            Displays this help message."
  exit 0
fi

if [[ "$1" == "-all" ]]; then
  
  EPOCHS=${2}
  NUM_SEEDS=${3}
  BINARY=${4}

  if $BINARY; then
    OUTPUT_DIR=logs/${TASK}_bin
  else
    OUTPUT_DIR=logs/${TASK}
  fi


  for MODEL_NAME in "${MODELS[@]}"; do
    for SEED in $(seq 1 ${NUM_SEEDS}); do
      echo "-------- Training model $MODEL_NAME with seed $SEED --------"
      CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python unfair_tos.py \
        --model_name_or_path ${MODEL_NAME} \
        --do_lower_case ${LOWER_CASE} \
        --output_dir ${OUTPUT_DIR}/${MODEL_NAME}/seed_${SEED} \
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
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --seed ${SEED} \
        --fp16 \
        --fp16_full_eval \
        --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
        --eval_accumulation_steps ${ACCUMULATION_STEPS} \
        --binary ${BINARY}
    done
  done
else
  MODEL_NAME=${1}
  EPOCHS=${2}
  BATCH_SIZE=${3}
  NUM_SEEDS=${4}
  BINARY=${5}

  if $BINARY; then
    OUTPUT_DIR=logs/${TASK}_bin
  else
    OUTPUT_DIR=logs/${TASK}
  fi

  for SEED in $(seq 1 ${NUM_SEEDS}); do
    echo "-------- Training model $MODEL_NAME with seed $SEED --------"
    CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python unfair_tos.py \
      --model_name_or_path ${MODEL_NAME} \
      --do_lower_case ${LOWER_CASE} \
      --output_dir ${OUTPUT_DIR}/${MODEL_NAME}/seed_${SEED} \
      --do_train \
      --do_eval \
      --do_pred \
      --overwrite_output_dir \
      --load_best_model_at_end \
      --metric_for_best_model micro-f1 \
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
      --binary ${BINARY}
  done
fi

python statistics/compute_avg_scores.py --dataset ${TASK}
