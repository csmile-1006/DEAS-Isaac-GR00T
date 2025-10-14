#!/bin/bash

TASK_NAMES=(
    "CoffeeSetupMug"
    "PnPMicrowaveToCounter"
    "TurnOffStove"
    "PnPCounterToMicrowave"
)

TASK_ID=$1
TASK_NAME=${TASK_NAMES[$TASK_ID]}
MODEL_TYPE="deas"

ACTION_HORIZON=16
ACTOR_CKPT_PATH=$2
CRITIC_CKPT_PATH=$3
SEED=${4:-42}
NUM_ENVS=${5:-5}
NUM_ROLLOUTS=${6:-50}
NUM_SAMPLES=${7:-10}
TEMPERATURE=${8:-0.0}

ROOT_PATH=~/debug/

CKPT_FOLDER=$(basename "$(dirname "${CRITIC_CKPT_PATH}")")
CKPT_STEP=$(basename "${CRITIC_CKPT_PATH}")
OUTPUT_PATH=${ROOT_PATH}/gr00tn15_robocasa/evaluations/${CKPT_FOLDER}/${CKPT_STEP}_s${SEED}_${TASK_NAME}_eval_n${NUM_ROLLOUTS}_bo${NUM_SAMPLES}_t${TEMPERATURE}
script="
    MUJOCO_GL=egl \
    python scripts/eval_policy_robocasa.py \
    --host localhost \
    --port 5555 \
    --model_type ${MODEL_TYPE} \
    --data_config single_panda_gripper_rl_inference \
    --action_horizon ${ACTION_HORIZON} \
    --embodiment_tag new_embodiment \
    --actor_model_path ${ACTOR_CKPT_PATH} \
    --critic_model_path ${CRITIC_CKPT_PATH} \
    --env_name ${TASK_NAME} \
    --num_episodes ${NUM_ROLLOUTS} \
    --noise 0.0 \
    --n_envs ${NUM_ENVS} \
    --output_path ${OUTPUT_PATH} \
    --num_samples ${NUM_SAMPLES} \
    --temperature ${TEMPERATURE} \
    --seed ${SEED} \
"
echo $script
eval $script
