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

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.critic.hlg import HLGaussLoss
from gr00t.model.critic.networks import DoubleCritic, Value

from .cross_attention_dit import SelfAttentionTransformer
from .flow_matching_action_head import CategorySpecificLinear


@dataclass
class CriticConfig(PretrainedConfig):
    hidden_dim: int = field(default=512, metadata={"help": "Hidden dimension."})
    depth: int = field(default=4, metadata={"help": "Depth of the network."})
    output_dim: int = field(default=1, metadata={"help": "Output dimension."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class RLConfig(PretrainedConfig):
    # RL parameters
    critic_action_horizon: int = field(default=1, metadata={"help": "Critic action horizon."})
    q_agg: str = field(default="min", metadata={"help": "Aggregation function for critic loss."})
    discount1: float = field(default=0.99, metadata={"help": "Discount factor for inner MDP."})
    discount2: float = field(default=0.99, metadata={"help": "Discount factor for outer MDP."})
    negative_reward: bool = field(default=True, metadata={"help": "Whether the reward is negative."})
    nstep: int = field(default=1, metadata={"help": "Number of steps for reward."})
    tau: float = field(default=0.005, metadata={"help": "Tau for polyak update."})

    feature_dim: int = field(default=64, metadata={"help": "Feature dimension for using in the critic."})

    num_atoms: int = field(default=101, metadata={"help": "Number of atoms for the critic."})
    sigma: float = field(default=0.1, metadata={"help": "Sigma for the critic."})
    expectile: float = field(default=0.9, metadata={"help": "Expectile for value loss."})
    support_type: str = field(default="geometric", metadata={"help": "Support type for the critic."})

    num_samples: int = field(default=1, metadata={"help": "Number of samples for BoN sampling."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class DEASCriticConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    input_embedding_dim: int = field(default=1536, metadata={"help": "Input embedding channel dimension."})
    backbone_embedding_dim: int = field(default=1536, metadata={"help": "Backbone embedding channel dimension."})

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    action_dim: int = field(default=7, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=16, metadata={"help": "Action horizon."})
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_critic: bool = field(default=True, metadata={"help": "Whether to tune the critic."})
    tune_value: bool = field(default=True, metadata={"help": "Whether to tune the value."})

    expand_batch: int = field(default=1)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default_factory=dict)
    critic_config: dict = field(default_factory=dict)
    value_config: dict = field(default_factory=dict)
    rl_config: dict = field(default_factory=dict)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, hidden_dim)
        self.layer3 = CategorySpecificLinear(num_categories, hidden_dim, hidden_dim)
        self.layer4 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.silu(self.layer1(x, cat_ids))
        hidden = F.silu(self.layer2(hidden, cat_ids))
        hidden = F.silu(self.layer3(hidden, cat_ids))
        return self.layer4(hidden, cat_ids)


class DEASCritic(nn.Module):
    config_class = DEASCriticConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: DEASCriticConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.rl_config = RLConfig(**config.rl_config)
        self.critic_action_horizon = self.rl_config.critic_action_horizon
        self.feature_dim = self.rl_config.feature_dim

        self.backbone_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.backbone_embedding_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.feature_dim,
        )

        self.value_config = CriticConfig(**config.value_config)
        self.value = Value(
            input_dim=config.max_state_dim + self.feature_dim,
            hidden_size=self.value_config.hidden_dim,
            depth=self.value_config.depth,
            output_dim=self.rl_config.num_atoms,
        )

        self.critic_config = CriticConfig(**config.critic_config)
        self.critic = DoubleCritic(
            input_dim=config.max_state_dim + self.feature_dim + self.critic_action_horizon * config.action_dim,
            hidden_dims=[self.critic_config.hidden_dim] * self.critic_config.depth,
            output_dim=self.rl_config.num_atoms,
        )

        self.target_critic = DoubleCritic(
            input_dim=config.max_state_dim + self.feature_dim + self.critic_action_horizon * config.action_dim,
            hidden_dims=[self.critic_config.hidden_dim] * self.critic_config.depth,
            output_dim=self.rl_config.num_atoms,
        )
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()
        # compute v_min and v_max according to the discount factor
        if self.rl_config.negative_reward:
            v_min = -1 * (1 / (1 - self.rl_config.discount2))
            v_max = 0.0
        else:
            v_min = 0.0
            v_max = 1.0
        self.hlg = HLGaussLoss(
            min_value=v_min,
            max_value=v_max,
            num_bins=self.rl_config.num_atoms,
            sigma=self.rl_config.sigma * ((v_max - v_min) / self.rl_config.num_atoms),
        )
        self.vlln = nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg) if config.use_vlln else nn.Identity()
        )

        self.config = config
        self.set_trainable_parameters(config.tune_value, config.tune_critic)

    def set_trainable_parameters(self, tune_value: bool, tune_critic: bool):
        self.tune_value = tune_value
        self.tune_critic = tune_critic
        for p in self.parameters():
            p.requires_grad = True
        self.target_critic.requires_grad_(False)
        self.vlln.requires_grad_(False)
        self.vl_self_attention.requires_grad_(False)
        if not tune_critic:
            self.backbone_encoder.requires_grad_(False)
            self.critic.requires_grad_(False)
        if not tune_value:
            self.value.requires_grad_(False)
        print(f"Tune action head critic: {self.tune_critic}")
        print(f"Tune action head value: {self.tune_value}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_critic:
                self.critic.eval()
            if not self.tune_value:
                self.value.eval()
                self.backbone_encoder.eval()
            # NOTE: No need to eval vlln and vl_self_attention because they are trained with the GR00T-N1.5
            self.vlln.eval()
            self.vl_self_attention.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def compute_value_loss(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        # NOTE: detach the vl_embeds to avoid gradient flow to the VLLN module in value loss
        # Only value gradient to the value loss
        vl_embeds = backbone_output.backbone_features.detach()
        embodiment_id = action_input.embodiment_id

        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device

        # Embed state.
        vl_embeds_mean = vl_embeds.mean(dim=1, keepdim=True)
        vl_embed_features = self.backbone_encoder(vl_embeds_mean, embodiment_id)
        vl_embed_features = F.tanh(vl_embed_features)

        v_logits = self.value(vl_embed_features, action_input.state)
        v_probs = torch.softmax(v_logits, dim=-1)
        vs = self.hlg.transform_from_probs(v_probs)

        # Value loss
        with torch.no_grad():
            vl_embed_features = self.backbone_encoder(vl_embeds_mean, embodiment_id)
            vl_embed_features = F.tanh(vl_embed_features)
            q1_logits, q2_logits = self.target_critic(
                vl_embed_features, action_input.state, action_input.action[:, : self.critic_action_horizon]
            )
            q_logits = torch.stack([q1_logits, q2_logits], dim=0)
            q_probs = torch.softmax(q_logits, dim=-1)
            qs = self.hlg.transform_from_probs(q_probs)

            if self.rl_config.q_agg == "min":
                min_q_idx = torch.argmin(qs, dim=0)
                batch_indices = torch.arange(batch_size, device=device)
                q_logit = q_logits[min_q_idx, batch_indices]
                q_prob = torch.softmax(q_logit, dim=-1)
                q = self.hlg.transform_from_probs(q_prob)
            elif self.rl_config.q_agg == "mean":
                q_logit = q_logits / 2
                q_prob = q_probs / 2
                q = self.hlg.transform_from_probs(q_prob)
            else:
                assert False, f"Invalid q_agg: {self.rl_config.q_agg}"

        g_hard = torch.where(q >= vs, self.rl_config.expectile, 1 - self.rl_config.expectile)
        g_hard_ratio = torch.where(q >= vs, 1.0, 0.0).sum(dim=-1) / batch_size
        # Explicit cross entropy implementation: -sum(target * log_softmax(input))
        log_probs = F.log_softmax(v_logits, dim=-1)
        ce_loss = -(q_prob * log_probs).sum(dim=-1)
        value_loss = (g_hard * ce_loss).mean()

        metrics = {
            "target_q_mean": q.mean(),
            "target_q_std": q.std(),
            "target_q_min": q.min(),
            "target_q_max": q.max(),
            "v_mean": vs.mean(),
            "v_std": vs.std(),
            "v_min": vs.min(),
            "v_max": vs.max(),
            "expectile_ratio": g_hard_ratio.mean(),
        }
        return value_loss, metrics

    def compute_critic_loss(
        self, backbone_output: BatchFeature, next_backbone_output: BatchFeature, action_input: BatchFeature
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute critic loss with proper gradient isolation."""
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)
        next_backbone_output = self.process_backbone_output(next_backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

            for k, v in next_backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                next_backbone_output[k] = expanded

        # Get vision and language embeddings.
        # NOTE: detach the vl_embeds to avoid gradient flow to the VLLN module in critic loss
        # Only critic gradient to the critic loss
        vl_embeds = backbone_output.backbone_features.detach()
        next_vl_embeds = next_backbone_output.backbone_features.detach()
        embodiment_id = action_input.embodiment_id

        vl_embeds_mean = vl_embeds.mean(dim=1, keepdim=True)
        vl_embed_features = self.backbone_encoder(vl_embeds_mean, embodiment_id)
        vl_embed_features = F.tanh(vl_embed_features)

        # Critic loss
        done = torch.prod(action_input.done, dim=-1)
        reward = action_input.reward
        if self.rl_config.negative_reward:
            reward -= 1
        discounts1 = self.rl_config.discount1 ** torch.arange(self.critic_action_horizon).to(reward.device)
        scaled_rewards = torch.sum(reward * discounts1, dim=-1)

        with torch.no_grad():
            next_vl_embeds_mean = next_vl_embeds.mean(dim=1, keepdim=True)
            next_vl_embed_features = self.backbone_encoder(next_vl_embeds_mean, embodiment_id)
            next_vl_embed_features = F.tanh(next_vl_embed_features)

            v_logits = self.value(next_vl_embed_features, action_input.next_state)
            v_probs = torch.softmax(v_logits, dim=-1)
            vs = self.hlg.transform_from_probs(v_probs)

            target_v = (
                scaled_rewards
                + (self.rl_config.discount2 ** (self.rl_config.nstep * self.critic_action_horizon)) * (1.0 - done) * vs
            )

        q1_logits, q2_logits = self.critic(
            vl_embed_features, action_input.state, action_input.action[:, : self.critic_action_horizon]
        )

        q1_probs = torch.softmax(q1_logits, dim=-1)
        q2_probs = torch.softmax(q2_logits, dim=-1)
        q1 = self.hlg.transform_from_probs(q1_probs)
        q2 = self.hlg.transform_from_probs(q2_probs)

        critic_loss = (self.hlg(q1_logits, target_v) + self.hlg(q2_logits, target_v)) / 2

        metrics = {
            "target_v_mean": target_v.mean(),
            "target_v_std": target_v.std(),
            "target_v_min": target_v.min(),
            "target_v_max": target_v.max(),
            "q1_mean": q1.mean(),
            "q1_std": q1.std(),
            "q1_min": q1.min(),
            "q1_max": q1.max(),
            "q2_mean": q2.mean(),
            "q2_std": q2.std(),
            "q2_min": q2.min(),
            "q2_max": q2.max(),
            "batch_reward": scaled_rewards.detach().mean(),
        }
        return critic_loss, metrics

    def forward(
        self, backbone_output: BatchFeature, next_backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        # Compute each loss separately to avoid gradient conflicts
        value_loss, value_metrics = self.compute_value_loss(backbone_output, action_input)
        critic_loss, critic_metrics = self.compute_critic_loss(backbone_output, next_backbone_output, action_input)
        total_loss = value_loss + critic_loss

        output_dict = {
            "loss": total_loss,
            "value_loss": value_loss,
            "critic_loss": critic_loss,
            **{f"value/{k}": v for k, v in value_metrics.items()},
            **{f"critic/{k}": v for k, v in critic_metrics.items()},
        }
        return BatchFeature(data=output_dict)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
