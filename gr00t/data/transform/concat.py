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

from typing import Optional

import numpy as np
import torch
from pydantic import Field

from gr00t.data.schema import DatasetMetadata, StateActionMetadata
from gr00t.data.transform.base import InvertibleModalityTransform


class BaseConcatTransform(InvertibleModalityTransform):
    """Base class for concatenation transforms."""

    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )

    state_concat_order: Optional[list[str]] = Field(
        default=None,
        description="Concatenation order for each state modality. Format: ['state.position', 'state.velocity', ...].",
    )

    action_concat_order: Optional[list[str]] = Field(
        default=None,
        description="Concatenation order for each action modality. Format: ['action.position', 'action.velocity', ...].",
    )

    action_dims: dict[str, int] = Field(
        default_factory=dict,
        description="The dimensions of the action keys.",
    )
    state_dims: dict[str, int] = Field(
        default_factory=dict,
        description="The dimensions of the state keys.",
    )

    def get_modality_metadata(self, key: str) -> StateActionMetadata:
        modality, subkey = key.split(".")
        assert self.dataset_metadata is not None, "Metadata not set"
        modality_config = getattr(self.dataset_metadata.modalities, modality)
        assert subkey in modality_config, f"{subkey=} not found in {modality_config=}"
        assert isinstance(
            modality_config[subkey], StateActionMetadata
        ), f"Expected {StateActionMetadata} for {subkey=}, got {type(modality_config[subkey])=}"
        return modality_config[subkey]

    def get_state_action_dims(self, key: str) -> int:
        """Get the dimension of a state or action key from the dataset metadata."""
        modality_config = self.get_modality_metadata(key)
        shape = modality_config.shape
        assert len(shape) == 1, f"{shape=}"
        return shape[0]

    def is_rotation_key(self, key: str) -> bool:
        modality_config = self.get_modality_metadata(key)
        return modality_config.rotation_type is not None

    def __call__(self, data: dict) -> dict:
        return self.apply(data)


class ConcatTransform(BaseConcatTransform):
    """Concatenate the keys according to specified order."""

    video_concat_order: list[str] = Field(
        ...,
        description="Concatenation order for each video modality. Format: ['video.ego_view_pad_res224_freq20', ...]",
    )

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {
                "apply_to",
                "video_concat_order",
                "state_concat_order",
                "action_concat_order",
            }
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    def apply(self, data: dict) -> dict:
        grouped_keys = {}
        for key in data.keys():
            try:
                modality, _ = key.split(".")
            except:  # noqa: E722
                modality = "language" if "annotation" in key else "others"
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)

        if "video" in grouped_keys:
            video_keys = grouped_keys["video"]
            assert self.video_concat_order is not None
            assert all(
                item in video_keys for item in self.video_concat_order
            ), f"keys in video_concat_order are misspecified, \n{video_keys=}, \n{self.video_concat_order=}"

            unsqueezed_videos = [
                np.expand_dims(data.pop(key), axis=-4) for key in self.video_concat_order
            ]
            data["video"] = np.concatenate(unsqueezed_videos, axis=-4)  # [..., V, H, W, C]

        if "state" in grouped_keys:
            state_keys = grouped_keys["state"]
            assert self.state_concat_order is not None
            assert all(
                item in state_keys for item in self.state_concat_order
            ), f"keys in state_concat_order are misspecified, \n{state_keys=}, \n{self.state_concat_order=}"

            for key in self.state_concat_order:
                target_shapes = [self.state_dims[key]]
                if self.is_rotation_key(key):
                    target_shapes.append(6)  # Allow for rotation_6d
                target_shapes.append(self.state_dims[key] * 2)  # Allow for sin-cos transform
                assert data[key].shape[-1] in target_shapes, f"State dim mismatch for {key=}"

            data["state"] = torch.cat([data.pop(key) for key in self.state_concat_order], dim=-1)

        if "action" in grouped_keys:
            action_keys = grouped_keys["action"]
            assert self.action_concat_order is not None
            assert set(self.action_concat_order) == set(action_keys)

            for key in self.action_concat_order:
                target_shapes = [self.action_dims[key]]
                if self.is_rotation_key(key):
                    target_shapes.append(3)  # Allow for axis angle
                assert data[key].shape[-1] in target_shapes, f"Action dim mismatch for {key=}"

            data["action"] = torch.cat([data.pop(key) for key in self.action_concat_order], dim=-1)

        return data

    def unapply(self, data: dict) -> dict:
        start_dim = 0
        assert "action" in data
        assert self.action_concat_order is not None
        action_tensor = data.pop("action")
        for key in self.action_concat_order:
            if key not in self.action_dims:
                raise ValueError(f"Action dim {key} not found in action_dims.")
            end_dim = start_dim + self.action_dims[key]
            data[key] = action_tensor[..., start_dim:end_dim]
            start_dim = end_dim

        if "state" in data:
            assert self.state_concat_order is not None
            start_dim = 0
            state_tensor = data.pop("state")
            for key in self.state_concat_order:
                end_dim = start_dim + self.state_dims[key]
                data[key] = state_tensor[..., start_dim:end_dim]
                start_dim = end_dim

        return data

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata and compute the dimensions of the state and action keys."""
        super().set_metadata(dataset_metadata)
        if self.action_concat_order is not None:
            for key in self.action_concat_order:
                self.action_dims[key] = self.get_state_action_dims(key)
        if self.state_concat_order is not None:
            for key in self.state_concat_order:
                self.state_dims[key] = self.get_state_action_dims(key)


class RLConcatTransform(ConcatTransform):
    """Concatenate the keys according to specified order for RL tasks."""

    next_state_concat_order: Optional[list[str]] = Field(
        default=None,
        description="Concatenation order for each next state modality. Format: ['next_state.position', 'next_state.velocity', ...].",
    )
    next_video_concat_order: list[str] = Field(
        default_factory=list,
        description="Concatenation order for each next video modality. Format: ['next_video.ego_view_pad_res224_freq20', ...]",
    )

    next_state_dims: dict[str, int] = Field(
        default_factory=dict,
        description="The dimensions of the next state keys.",
    )

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {
                "apply_to",
                "video_concat_order",
                "state_concat_order",
                "next_state_concat_order",
                "next_video_concat_order",
                "action_concat_order",
            }
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    def apply(self, data: dict) -> dict:
        grouped_keys = {}
        for key in data.keys():
            try:
                modality, _ = key.split(".")
            except:  # noqa: E722
                modality = "language" if "annotation" in key else "others"
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)

        if "video" in grouped_keys:
            video_keys = grouped_keys["video"]
            assert self.video_concat_order is not None
            assert all(item in video_keys for item in self.video_concat_order), (
                f"keys in video_concat_order are misspecified, \n{video_keys=}, \n{self.video_concat_order=}"
            )

            unsqueezed_videos = [np.expand_dims(data.pop(key), axis=-4) for key in self.video_concat_order]
            data["video"] = np.concatenate(unsqueezed_videos, axis=-4)  # [..., V, H, W, C]

        if "state" in grouped_keys:
            state_keys = grouped_keys["state"]
            assert self.state_concat_order is not None
            assert all(item in state_keys for item in self.state_concat_order), (
                f"keys in state_concat_order are misspecified, \n{state_keys=}, \n{self.state_concat_order=}"
            )

            for key in self.state_concat_order:
                target_shapes = [self.state_dims[key]]
                if self.is_rotation_key(key):
                    target_shapes.append(6)  # Allow for rotation_6d
                target_shapes.append(self.state_dims[key] * 2)  # Allow for sin-cos transform
                assert data[key].shape[-1] in target_shapes, f"State dim mismatch for {key=}"

            data["state"] = torch.cat([data.pop(key) for key in self.state_concat_order], dim=-1)

        if "action" in grouped_keys:
            action_keys = grouped_keys["action"]
            assert self.action_concat_order is not None
            assert set(self.action_concat_order) == set(action_keys)

            for key in self.action_concat_order:
                target_shapes = [self.action_dims[key]]
                if self.is_rotation_key(key):
                    target_shapes.append(3)  # Allow for axis angle
                assert data[key].shape[-1] in target_shapes, f"Action dim mismatch for {key=}"

            data["action"] = torch.cat([data.pop(key) for key in self.action_concat_order], dim=-1)

        if "next_video" in grouped_keys:
            assert self.next_video_concat_order is not None
            next_video_keys = grouped_keys["next_video"]
            assert all(
                item in next_video_keys for item in self.next_video_concat_order
            ), "Keys in next_video_concat_order are misspecified"

            unsqueezed_videos = [
                np.expand_dims(data.pop(key), axis=-4) for key in self.next_video_concat_order
            ]
            data["next_video"] = np.concatenate(unsqueezed_videos, axis=-4)  # [..., V, H, W, C]

        if "next_state" in data:
            assert self.next_state_concat_order is not None
            next_state_keys = grouped_keys["next_state"]
            assert all(
                item in next_state_keys for item in self.next_state_concat_order
            ), "Keys in next_state_concat_order are misspecified"

            tensors_to_concat = []
            for key in self.next_state_concat_order:
                target_shapes = [self.next_state_dims[key]]
                if self.is_rotation_key(key):
                    target_shapes.append(6)  # Allow for rotation_6d
                target_shapes.append(self.next_state_dims[key] * 2)  # Allow for sin-cos transform

                assert data[key].shape[-1] in target_shapes, f"Next state dim mismatch for {key=}"
                tensors_to_concat.append(data.pop(key))

            data["next_state"] = torch.cat(tensors_to_concat, dim=-1)

        return data

    def unapply(self, data: dict) -> dict:
        data = super().unapply(data)

        if "next_state" in data:
            assert self.next_state_concat_order is not None
            start_dim = 0
            next_state_tensor = data.pop("next_state")
            for key in self.next_state_concat_order:
                end_dim = start_dim + self.next_state_dims[key]
                data[key] = next_state_tensor[..., start_dim:end_dim]
                start_dim = end_dim

        return data

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        super().set_metadata(dataset_metadata)
        if self.next_state_concat_order is not None:
            for key in self.next_state_concat_order:
                self.next_state_dims[key] = self.get_state_action_dims(key)


class RLStateConcatTransform(RLConcatTransform):
    """Concatenate only state and action keys for RL tasks."""

    video_concat_order: list[str] = Field(
        default_factory=list,
        description="Not used in this transform.",
    )

    def apply(self, data: dict) -> dict:
        # Skip video processing from parent class
        grouped_keys = {}
        for key in data.keys():
            try:
                modality, _ = key.split(".")
            except:  # noqa: E722
                modality = "language" if "annotation" in key else "others"
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)

        state_action_modalities = {
            "state": (self.state_concat_order, self.state_dims),
            "next_state": (self.next_state_concat_order, self.next_state_dims),
            "action": (self.action_concat_order, self.action_dims),
        }

        for modality_name, (concat_order, dims_map) in state_action_modalities.items():
            if modality_name in grouped_keys:
                available_keys = grouped_keys[modality_name]
                assert concat_order is not None, f"{modality_name}_concat_order must be specified."

                if modality_name == "action":
                    assert set(concat_order) == set(available_keys), "Action keys mismatch"
                else:
                    assert all(item in available_keys for item in concat_order), f"Keys in {modality_name}_concat_order are misspecified"

                tensors_to_concat = []
                for key in concat_order:
                    target_shapes = [dims_map[key]]
                    if self.is_rotation_key(key):
                        if modality_name in ["state", "next_state"]:
                            target_shapes.append(6)  # Allow for rotation_6d
                        elif modality_name == "action":
                            target_shapes.append(3)  # Allow for axis angle

                    if modality_name in ["state", "next_state"]:
                        target_shapes.append(dims_map[key] * 2)  # Allow for sin-cos transform

                    assert data[key].shape[-1] in target_shapes, f"{modality_name} dim mismatch for {key=}"
                    tensors_to_concat.append(data.pop(key))

                data[modality_name] = torch.cat(tensors_to_concat, dim=-1)

        return data
