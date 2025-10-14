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


import torch
from torch import nn


class MLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, layer_norm: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ResidualBlock(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.block = torch.nn.Sequential(
            # First dense block
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            # Second dense block
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            # Final transformation before residual connection
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.block(x)
        return x + identity


class BRONet(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, depth: int, add_final_layer: bool = True, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.depth = depth
        self.add_final_layer = add_final_layer
        self.output_dim = output_dim

        # Create the residual blocks based on depth
        self.input_projection = nn.Linear(input_dim, hidden_size)
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.activation = nn.ReLU()
        self.residual_blocks = torch.nn.ModuleList()
        for _ in range(depth):
            block = ResidualBlock(hidden_size)
            self.residual_blocks.append(block)

        if add_final_layer:
            self.final_layer = torch.nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.input_layernorm(self.input_projection(x)))
        # Pass through each residual block sequentially
        for block in self.residual_blocks:
            x = block(x)

        if self.add_final_layer:
            x = self.final_layer(x)

        return x


class DoubleCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], add_final_layer: bool = True, output_dim: int = 1):
        super().__init__()
        self.Q1 = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        )
        self.Q2 = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        )

    def forward(self, vl_embed_features: torch.Tensor, states: torch.Tensor, actions: torch.Tensor):
        B = states.shape[0]
        state_action = torch.cat(
            [vl_embed_features.reshape(B, -1), states.reshape(B, -1), actions.reshape(B, -1)], dim=1
        )
        q1 = self.Q1(state_action)
        q2 = self.Q2(state_action)

        if q1.shape[-1] == 1:
            q1 = q1.squeeze(-1)
        if q2.shape[-1] == 1:
            q2 = q2.squeeze(-1)
        return q1, q2


class BRONetDoubleCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, depth: int, add_final_layer: bool = True, output_dim: int = 1):
        super().__init__()
        self.Q1 = BRONet(
            input_dim=input_dim,
            hidden_size=hidden_size,
            depth=depth,
            add_final_layer=add_final_layer,
            output_dim=output_dim,
        )
        self.Q2 = BRONet(
            input_dim=input_dim,
            hidden_size=hidden_size,
            depth=depth,
            add_final_layer=add_final_layer,
            output_dim=output_dim,
        )

    def forward(self, vl_embed_features: torch.Tensor, states: torch.Tensor, actions: torch.Tensor):
        B = states.shape[0]
        state_action = torch.cat(
            [vl_embed_features.reshape(B, -1), states.reshape(B, -1), actions.reshape(B, -1)], dim=1
        )
        q1 = self.Q1(state_action)
        q2 = self.Q2(state_action)
        if q1.shape[-1] == 1:
            q1 = q1.squeeze(-1)
        if q2.shape[-1] == 1:
            q2 = q2.squeeze(-1)
        return q1, q2


class Value(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, depth: int, add_final_layer: bool = True, output_dim: int = 1):
        super().__init__()
        self.value = BRONet(
            input_dim=input_dim,
            hidden_size=hidden_size,
            depth=depth,
            add_final_layer=add_final_layer,
            output_dim=output_dim,
        )

    def forward(self, vl_embed_features, states):
        B = states.shape[0]
        v = self.value(torch.cat([vl_embed_features.reshape(B, -1), states.reshape(B, -1)], dim=1))
        if v.shape[-1] == 1:
            v = v.squeeze(-1)
        return v


class BRONetValue(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, depth: int, add_final_layer: bool = True, output_dim: int = 1):
        super().__init__()
        self.value = BRONet(
            input_dim=input_dim,
            hidden_size=hidden_size,
            depth=depth,
            add_final_layer=add_final_layer,
            output_dim=output_dim,
        )

    def forward(self, vl_embed_features: torch.Tensor, states: torch.Tensor):
        B = states.shape[0]
        v = self.value(torch.cat([vl_embed_features.reshape(B, -1), states.reshape(B, -1)], dim=1))
        if v.shape[-1] == 1:
            v = v.squeeze(-1)
        return v


class StateValue(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, depth: int, add_final_layer: bool = True, output_dim: int = 1):
        super().__init__()
        self.value = BRONet(
            input_dim=input_dim,
            hidden_size=hidden_size,
            depth=depth,
            add_final_layer=add_final_layer,
            output_dim=output_dim,
        )

    def forward(self, states):
        B = states.shape[0]
        v = self.value(states.reshape(B, -1))
        if v.shape[-1] == 1:
            v = v.squeeze(-1)
        return v


class StateDoubleCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], add_final_layer: bool = True, output_dim: int = 1):
        super().__init__()
        self.Q1 = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        )
        self.Q2 = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor):
        B = states.shape[0]
        state_action = torch.cat([states.reshape(B, -1), actions.reshape(B, -1)], dim=1)
        q1 = self.Q1(state_action)
        q2 = self.Q2(state_action)

        if q1.shape[-1] == 1:
            q1 = q1.squeeze(-1)
        if q2.shape[-1] == 1:
            q2 = q2.squeeze(-1)
        return q1, q2
