#!/bin/bash                                                                                                                                                  

STEPS=$1
CRITIC_ACTION_HORIZON=$2
DISCOUNT1=${3:-0.9}
DISCOUNT2=${4:-0.99}
EXPECTILE=${5:-0.7}
NUM_GPUS=${6:-2}
BATCH_SIZE=${7:-16}

BASE_PATH=~/
TOTAL_BATCH_SIZE=$((NUM_GPUS * BATCH_SIZE))

TASK_NAMES=(
    "CoffeeSetupMug"
    "PnPMicrowaveToCounter"
    "TurnOffStove"
    "PnPCounterToMicrowave"
)

dataset_path=""
for TASK in ${TASK_NAMES[@]}; do
    TASK_NAME=$TASK
    dataset_path="${dataset_path} ${BASE_PATH}/deas_robocasa/demos/${TASK_NAME}"
    dataset_path="${dataset_path} ${BASE_PATH}/deas_robocasa/rollouts/${TASK_NAME}"
done

RUN_NAME=DEAS_Critic_as${CRITIC_ACTION_HORIZON}_e${EXPECTILE}_d1${DISCOUNT1}_d2${DISCOUNT2}_bs${TOTAL_BATCH_SIZE}_steps${RUN_NAME}
script="
    WANDB_PROJECT=gr00t-deas-critic-finetune \
    python scripts/gr00t_deas_critic_finetune.py \
    --dataset-path ${dataset_path} \
    --num-gpus ${NUM_GPUS} \
    --output-dir ${BASE_PATH}/ckpts/critics/${RUN_NAME}/ \
    --max-steps ${STEPS} \
    --data-config single_panda_gripper_rl \
    --batch-size ${BATCH_SIZE} \
    --save-steps 5000 \
    --run-name ${RUN_NAME} \
    --critic-action-horizon ${CRITIC_ACTION_HORIZON} \
    --discount1 ${DISCOUNT1} \
    --discount2 ${DISCOUNT2} \
    --expectile ${EXPECTILE} \
"
echo $script
eval $script