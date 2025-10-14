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

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from .backbone import EagleBackbone
from .critic.critic import Critic, CriticConfig
from .gr00t_n1 import GR00T_N1_5

LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


# config
@dataclass
class RL_Critic_Config(PretrainedConfig):
    model_type = "rl_critic"

    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})
    critic_cfg: dict = field(init=False, metadata={"help": "Critic configuration."})
    action_horizon: int = field(default=16, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# real model
class RL_Critic(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = RL_Critic_Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: RL_Critic_Config,
        local_model_path: str,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.critic_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path

        self.backbone = EagleBackbone(**config.backbone_cfg)
        critic_cfg = CriticConfig(**config.critic_cfg)
        self.critic = Critic(critic_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3 and action.shape[1] == self.action_horizon and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, critic_outputs, is_training):
        fail_critic = (not isinstance(critic_outputs, BatchFeature)) or not (
            LOSS_KEY in critic_outputs and is_training  # there might not be an action prediction during training
        )

        if fail_critic:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(critic_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in critic_outputs=}"
            error_msg += f"\n{critic_outputs[LOSS_KEY].shape=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, critic_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        critic_outputs = self.critic(backbone_outputs, critic_inputs)
        self.validate_data(critic_outputs, is_training=True)
        return critic_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        critic_inputs = self.critic.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.critic.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        critic_inputs = tree.map_structure(to_device_with_maybe_dtype, critic_inputs)
        return backbone_inputs, critic_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, from_gr00t_n1_5: bool = False, **kwargs):
        tune_visual = kwargs.pop("tune_visual", False)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_vlln = kwargs.pop("tune_vlln", False)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune action head projector: {tune_projector}")

        if not from_gr00t_n1_5:
            # get the current model path being downloaded
            try:
                # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
                # saved in ~/.cache/huggingface/hub/
                local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
                # HFValidationError, RepositoryNotFoundError
            except (HFValidationError, RepositoryNotFoundError):
                print(
                    f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
                )
                local_model_path = pretrained_model_name_or_path

            pretrained_model = super().from_pretrained(local_model_path, local_model_path=local_model_path, **kwargs)

            pretrained_model.critic.set_trainable_parameters(tune_projector=tune_projector)
            return pretrained_model

        else:
            pretrained_gr00t_n1_5 = GR00T_N1_5.from_pretrained(pretrained_model_name_or_path, **kwargs)

            new_cfg = RL_Critic_Config()
            pretrained_gr00t_n1_5_cfg = pretrained_gr00t_n1_5.config

            new_cfg.backbone_cfg = pretrained_gr00t_n1_5_cfg.backbone_cfg
            # new_cfg.action_horizon = pretrained_gr00t_n1_5_cfg.action_horizon
            new_cfg.action_dim = pretrained_gr00t_n1_5_cfg.action_dim
            new_cfg.compute_dtype = pretrained_gr00t_n1_5_cfg.compute_dtype

            critic_cfg = CriticConfig(
                input_embedding_dim=pretrained_gr00t_n1_5_cfg.action_head_cfg["input_embedding_dim"],
                backbone_embedding_dim=pretrained_gr00t_n1_5_cfg.action_head_cfg["backbone_embedding_dim"],
                hidden_size=pretrained_gr00t_n1_5_cfg.action_head_cfg["hidden_size"],
                depth=3,
                add_final_layer=True,
                output_dim=1,
                action_dim=pretrained_gr00t_n1_5_cfg.action_dim,
                action_horizon=pretrained_gr00t_n1_5_cfg.action_horizon,
                max_state_dim=pretrained_gr00t_n1_5_cfg.action_head_cfg["max_state_dim"],
                use_vlln=pretrained_gr00t_n1_5_cfg.action_head_cfg["use_vlln"],
                vl_self_attention_cfg=pretrained_gr00t_n1_5_cfg.action_head_cfg["vl_self_attention_cfg"],
            )

            new_cfg.critic_cfg = critic_cfg.to_dict()

            # Transfer action head config
            pretrained_model = cls(
                config=new_cfg,
                local_model_path=pretrained_gr00t_n1_5.local_model_path,
            )

            print("Loading backbone parameters")
            pretrained_model.backbone.load_state_dict(pretrained_gr00t_n1_5.backbone.state_dict())

            # Transfer parameters from pretrained GR00T_N1_5 model

            with torch.no_grad():
                pretrained_model.critic.state_encoder.load_state_dict(
                    pretrained_gr00t_n1_5.action_head.state_encoder.state_dict()
                )

            # Set trainable parameters according to flags
            pretrained_model.backbone.set_trainable_parameters(tune_visual=tune_visual, tune_llm=tune_llm)
            pretrained_model.critic.set_trainable_parameters(tune_projector=tune_projector, tune_vlln=tune_vlln)

            del pretrained_gr00t_n1_5
            return pretrained_model


# register
AutoConfig.register("rl_critic", RL_Critic_Config)
AutoModel.register(RL_Critic_Config, RL_Critic)
