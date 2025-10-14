#!/bin/bash                                                                                                                                                  

STEPS=${1:-30000}
NUM_GPUS=${2:-2}
BATCH_SIZE=${3:-16}
TOTAL_BATCH_SIZE=$((NUM_GPUS * BATCH_SIZE))

BASE_PATH=~/
TASK_NAMES=(
    "CoffeeSetupMug"
    "PnPMicrowaveToCounter"
    "TurnOffStove"
    "PnPCounterToMicrowave"
)

dataset_path=""
for TASK_NAME in ${TASK_NAMES[@]}; do
    dataset_path="${dataset_path} ${BASE_PATH}/deas_robocasa/demos/${TASK_NAME}"
    dataset_path="${dataset_path} ${BASE_PATH}/deas_robocasa/success_rollouts/${TASK_NAME}"
done

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