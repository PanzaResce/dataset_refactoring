if [[ "$1" == "-h" ]]; then
  echo "Usage: $0 [MODEL_NAME] [NUM_SEEDS] [BINARY] "
  echo ""
  echo "Description:"
  echo "  This script trains and evaluates models on the unfair_tos task using specified parameters."
  echo ""
  echo "Options:"
  echo "  MODEL_NAME    Name of the model to be used"
  echo "  NUM_SEED      Number of random seed to run "
  echo "  BINARY        Train the model in binary mode instead of multi-label (default: True)"
  echo "  -h            Displays this help message."
  exit 0
fi

GPU_NUMBER=1
LOWER_CASE='True'
ACCUMULATION_STEPS=1
TASK='unfair_tos'
MODEL_NAME=${1}
EPOCHS=5
NUM_SEED=${2}
BINARY=${3:-True}
BATCH_SIZE=4

python unfair_tos.py \
        --model_name_or_path ${MODEL_NAME} \
        --do_lower_case ${LOWER_CASE} \
        --output_dir logs/${TASK}/${MODEL_NAME}/seed_${NUM_SEED} \
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
        --seed ${NUM_SEED} \
        --fp16 \
        --fp16_full_eval \
        --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
        --eval_accumulation_steps ${ACCUMULATION_STEPS} \
        --binary ${BINARY}