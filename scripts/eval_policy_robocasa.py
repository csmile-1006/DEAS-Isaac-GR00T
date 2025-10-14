# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import csv
import datetime
import json
import os
import random
import time
import warnings
from glob import glob
from pathlib import Path

import h5py
import mujoco
import numpy as np
import robocasa
import robosuite
import torch
from robocasa.utils.robomimic.robomimic_dataset_utils import convert_to_robomimic_format
from robosuite.controllers import load_composite_controller_config
from tqdm import tqdm

from gr00t.eval.robot import RobotInferenceClient
from gr00t.eval.wrappers.robocasa_wrapper import load_robocasa_gym_env
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import (
    BasePolicy,
    Gr00tPolicy,
    Gr00tDEASDualBoNPolicy,
)

warnings.simplefilter("ignore", category=FutureWarning)


def control_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def flatten(d, parent_key="", sep="."):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def gather_demonstrations_as_hdf5(directory, out_dir, env_info, excluded_episodes=None):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.
    The strucure of the hdf5 file is as follows.
    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected
        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration
        demo2 (group)
        ...
    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    print("Saving hdf5 to", hdf5_path)
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        # print("Processing {} ...".format(ep_directory))
        if (excluded_episodes is not None) and (ep_directory in excluded_episodes):
            # print("\tExcluding this episode!")
            continue

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        actions_abs = []
        rewards = []
        dones = []
        successes = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            rewards.extend(dic["rewards"])
            dones.extend(dic["dones"])
            successes.extend(dic["successes"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
                if "actions_abs" in ai:
                    actions_abs.append(ai["actions_abs"])

        if len(states) == 0:
            continue

        # Delete the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action,
        # so we end up with an extra state at the end.
        del states[-1]
        assert len(states) == len(actions)

        if np.sum(successes) > 0:
            # cut transitions to the first successful state
            for i in range(len(states)):
                if successes[i]:
                    break
            states = states[: i + 1]
            actions = actions[: i + 1]
            if len(actions_abs) > 0:
                actions_abs = actions_abs[: i + 1]
            rewards = rewards[: i + 1]
            dones = successes[: i + 1]  # make dones same as successes
        else:
            dones[-1] = True  # make the last state a terminal state

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # store ep meta as an attribute
        ep_meta_path = os.path.join(directory, ep_directory, "ep_meta.json")
        if os.path.exists(ep_meta_path):
            with open(ep_meta_path, "r") as f:
                ep_meta = f.read()
            ep_data_grp.attrs["ep_meta"] = ep_meta

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))
        ep_data_grp.create_dataset("rewards", data=np.array(rewards))
        ep_data_grp.create_dataset("dones", data=np.array(dones))
        if len(actions_abs) > 0:
            print(np.array(actions_abs).shape)
            ep_data_grp.create_dataset("actions_abs", data=np.array(actions_abs))

        # else:
        #     pass
        #     # print("Demonstration is unsuccessful and has NOT been saved")

    print("{} rollouts so far".format(num_eps))

    if num_eps == 0:
        f.close()
        return

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["robocasa_version"] = robocasa.__version__
    grp.attrs["robosuite_version"] = robosuite.__version__
    grp.attrs["mujoco_version"] = mujoco.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()

    return hdf5_path


"""
Example command:

python scripts/eval_policy_robocasa.py --host localhost --port 5555
    --action_horizon 16
    --embodiment_tag gr1
    --data_config gr1_arms_waist
    --env_name CloseDrawer
    --num_episodes 10
provide --model_path to load up the model checkpoint in this script.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost", help="host")
    parser.add_argument("--port", type=int, default=5555, help="port")
    parser.add_argument("--n_envs", type=int, default=1, help="number of environments")
    parser.add_argument(
        "--data_config",
        type=str,
        default="gr1_arms_only",
        choices=list(DATA_CONFIG_MAP.keys()),
        help="data config name",
    )
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="gr1",
    )
    ## When using a model instead of client-server mode.
    parser.add_argument(
        "--actor_model_path",
        type=str,
        default=None,
        help="Path to the model checkpoint directory, this will disable client server mode.",
    )
    parser.add_argument(
        "--critic_model_path",
        type=str,
        default=None,
        help="[Optional] Path to the critic model checkpoint directory.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="deas",
        choices=["deas", "gr00tn15"],
        help="Type of model to use.",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        help="Number of denoising steps if model_path is provided",
        default=4,
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of samples for BoN sampling.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for BoN sampling.",
    )

    # robocasa env and evaluation parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the robocasa environment",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="CloseDrawer",
        help="Name of the robocasa environment to load",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the output",
    )
    parser.add_argument(
        "--save_video",
        default=False,
        action="store_true",
        help="Whether to save the video",
    )

    # Robocasa env parameters
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be, eg. 'NONE' or 'WHOLE_BODY_IK', etc. Or path to controller json file",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="PandaOmron",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--obj_groups",
        type=str,
        nargs="+",
        default=None,
        help="In kitchen environments, either the name of a group to sample object from or path to an .xml file",
    )

    parser.add_argument("--layout", type=int, nargs="+", default=-1)
    parser.add_argument("--style", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11])
    parser.add_argument("--generative_textures", action="store_true", help="Use generative textures")

    # Data collection parameters
    parser.add_argument(
        "--collect_data",
        action="store_true",
        default=False,
        help="Whether to collect data",
    )
    parser.add_argument(
        "--data_collection_path",
        type=str,
        default="",
        help="Path to save the data collection",
    )

    parser.add_argument(
        "--reward_shaping",
        action="store_true",
        default=False,
        help="Whether to use reward shaping",
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=0.1,
        help="Noise level for the state and action",
    )

    parser.add_argument(
        "--noise_smoothing",
        type=float,
        default=0.3,
        help="ACtion noise smoothing level.",
    )

    args = parser.parse_args()
    control_seed(args.seed)

    assert args.data_config in ["single_panda_gripper_rl_inference"], (
        "Only single panda gripper RL inference data config is supported for now"
    )
    data_config = DATA_CONFIG_MAP[args.data_config](AS=args.action_horizon)

    if args.critic_model_path is not None:
        import torch

        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()
        if args.model_type == "deas":
            policy = Gr00tDEASDualBoNPolicy(
                actor_model_path=args.actor_model_path,
                critic_model_path=args.critic_model_path,
                modality_config=modality_config,
                modality_transform=modality_transform,
                embodiment_tag=args.embodiment_tag,
                denoising_steps=args.denoising_steps,
                num_samples=args.num_samples,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
    elif args.actor_model_path is not None:
        import torch

        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()
 
        if args.model_type == "gr00tn15":
            policy = Gr00tPolicy(
                model_path=args.actor_model_path,
                modality_config=modality_config,
                modality_transform=modality_transform,
                embodiment_tag=args.embodiment_tag,
                denoising_steps=args.denoising_steps,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            raise ValueError(f"Invalid model type: {args.model_type}")
    else:
        policy: BasePolicy = RobotInferenceClient(host=args.host, port=args.port)

    all_gt_actions = []
    all_pred_actions = []

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    print(modality)

    # ROBOCASA ENV SETUP
    # load robocasa env
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots if isinstance(args.robots, str) else args.robots[0],
    )

    env_name = args.env_name
    # Create argument configuration
    config = {
        "env_name": env_name,
        "robots": args.robots,
        "controller_configs": controller_config,
        "generative_textures": "100p",
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in env_name:
        config["env_configuration"] = args.config

    # Mirror actions if using a kitchen environment
    if env_name in ["Lift"]:  # add other non-kitchen tasks here
        if args.obj_groups is not None:
            print("Specifying 'obj_groups' in non-kitchen environment does not have an effect.")
    else:
        config["layout_ids"] = args.layout
        config["style_ids"] = args.style
        ### update config for kitchen envs ###
        if args.obj_groups is not None:
            config.update({"obj_groups": args.obj_groups})

        # by default use obj instance split A
        config["obj_instance_split"] = "A"
        # config["obj_instance_split"] = None
        # config["obj_registries"] = ("aigen",)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    if args.critic_model_path is not None:
        assert args.action_horizon == policy.model.critic_action_horizon, (
            f"Action horizon mismatch: {args.action_horizon} != {policy.model.critic_action_horizon}"
        )

    env = load_robocasa_gym_env(
        args.env_name,
        n_envs=args.n_envs,
        seed=args.seed,
        # robosuite-related configs
        robots=args.robots,
        camera_widths=256,
        camera_heights=256,
        render_onscreen=False,
        # robocasa-related configs
        obj_instance_split="B",
        randomize_cameras=False,
        layout_and_style_ids=((1, 1), (2, 2), (4, 4), (6, 9), (7, 10)),
        generative_textures="100p" if args.generative_textures else None,
        # data collection configs
        collect_data=args.collect_data,
        collect_directory=Path(args.data_collection_path) if args.collect_data else None,
        # video configs
        video_path=None if not args.save_video else Path(args.output_path) / "videos",
        # multi-step configs
        action_horizon=args.action_horizon,
        video_delta_indices=np.array([0]),
        state_delta_indices=np.array([0]),
        # reward configs
        reward_shaping=args.reward_shaping,
    )

    # postprocess function of action, to handle the case where number of dimensions are not the same
    def postprocess_action(action):
        new_action = {}
        for k, v in action.items():
            if v.ndim == 1:
                new_action[k] = v[..., None]
            else:
                new_action[k] = v
        return new_action

    # main evaluation loop
    start_time = time.time()

    # Initialize tracking variables
    episode_lengths = []
    current_rewards = [0] * args.n_envs
    current_lengths = [0] * args.n_envs
    completed_episodes = 0
    current_successes = [False] * args.n_envs
    episode_successes = []
    noise_levels = {
        "action.end_effector_position": 0.05,
        "action.end_effector_rotation": 0.3,
        "action.gripper_close": 1.0,
        "action.base_motion": 0.5,
        "action.control_mode": 1.0,
    }

    def add_noise(actions):
        if args.noise > 0:
            for key in actions.keys():
                noise = np.random.normal(0, 1, size=actions[key].shape) * noise_levels[key] * args.noise
                noise = np.clip(noise, -args.noise_smoothing, args.noise_smoothing)
        return actions

    # Initial environment reset
    obs, _ = env.reset()
    pbar = tqdm(
        total=args.num_episodes,
        desc=f"Evaluating {args.num_episodes} episodes",
        leave=False,
    )

    # Main simulation loop
    while completed_episodes < args.num_episodes:
        # Process observations and get actions from the server
        actions = policy.get_action(obs)
        actions = add_noise(actions)
        # Step the environment
        next_obs, rewards, terminations, truncations, env_infos = env.step(actions)
        # Update episode tracking
        for env_idx in range(args.n_envs):
            current_successes[env_idx] |= bool(env_infos["success"][env_idx][0])
            current_rewards[env_idx] += rewards[env_idx]
            current_lengths[env_idx] += args.action_horizon
            # If episode ended, store results
            if terminations[env_idx] or truncations[env_idx]:
                episode_lengths.append(current_lengths[env_idx])
                episode_successes.append(current_successes[env_idx])
                current_successes[env_idx] = False
                completed_episodes += 1
                # Reset trackers for this environment
                current_rewards[env_idx] = 0
                current_lengths[env_idx] = 0
                pbar.update(1)
        obs = next_obs

    pbar.close()
    env.reset()
    env.close()

    print(f"Collecting {args.num_episodes} episodes took {time.time() - start_time:.2f} seconds")
    assert len(episode_successes) >= args.num_episodes, (
        f"Expected at least {args.num_episodes} episodes, got {len(episode_successes)}"
    )

    os.makedirs(args.output_path if args.output_path else "./", exist_ok=True)
    csv_path = Path(args.output_path if args.output_path else "./") / "eval.csv"
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["episode", "success", "length"])
        for i, (succ, length) in enumerate(zip(episode_successes, episode_lengths)):
            writer.writerow([i, int(succ), length])
    print(f"Saved evaluation results to {csv_path}")

    # Also write success rate to a separate file
    success_rate = np.mean(episode_successes)
    success_path = Path(args.output_path if args.output_path else "./") / "success.txt"
    with open(success_path, "w") as f:
        f.write(f"Success Rate: {success_rate:.4f}\n")
    print(f"Saved success rate to {success_path}")

    if args.collect_data:
        print("Change collected data to hdf5 format")
        hdf5_path = gather_demonstrations_as_hdf5(args.data_collection_path, args.data_collection_path, env_info)
        convert_to_robomimic_format(hdf5_path)

    print(f"episode_successes: {episode_successes}")
    print(f"Success Rate (%): {np.mean(episode_successes)}")

    exit()
