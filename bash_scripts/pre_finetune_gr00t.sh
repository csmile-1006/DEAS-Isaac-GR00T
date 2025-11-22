#!/bin/bash                                                                                                                                                  

STEPS=${1:-30000}
NUM_GPUS=${2:-2}
BATCH_SIZE=${3:-16}
TOTAL_BATCH_SIZE=$((NUM_GPUS * BATCH_SIZE))

BASE_PATH=~/
dataset_path="${BASE_PATH}/datasets/robocasa_mg_gr00t_100"

SCRIPT="
    WANDB_PROJECT=gr00t-deas-finetune \
    python scripts/gr00t_finetune.py \
    --dataset-path "${dataset_path}" \
    --num-gpus ${NUM_GPUS} \
    --output-dir "${BASE_PATH}/ckpts/gr00tn15_bs${TOTAL_BATCH_SIZE}_step${STEPS}" \
    --max-steps "${STEPS}" \
    --data-config single_panda_gripper \
    --batch-size ${BATCH_SIZE} \
    --save-steps 10000 \
    --run-name "gr00tn15_bs${TOTAL_BATCH_SIZE}_step${STEPS}" 
"
echo $SCRIPT
eval $SCRIPT