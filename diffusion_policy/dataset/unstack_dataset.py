import copy
import os
from typing import Dict

import numpy as np
import torch

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


class UnstackDataset(BaseImageDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        action_type="absolute",
    ):
        """UnstackDataset is a dataset for unstacking images and their corresponding actions.

        Args:
            zarr_path: Path to the Zarr dataset.
            horizon: Number of steps to unstack. Defaults to 1.
            pad_before: Number of frames to pad before the action. Defaults to 0.
            pad_after: Number of frames to pad after the action. Defaults to 0.
            seed: Random seed for sampling. Defaults to 42.
            val_ratio: Ratio of episodes to use for validation. Defaults to 0.0.
            max_train_episodes: Maximum number of training episodes. Defaults to None.
            action_type: Type of action representation. We support the "relative", "delta",
                and "absolute" action types. Refer to the UMI paper, Figure 6 for details
                (https://umi-gripper.github.io/umi.pdf). Defaults to "relative".

        Raises:
            FileNotFoundError: If the Zarr path does not exist.
        """

        super().__init__()

        self.action_type = action_type
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Zarr path {os.path.abspath(zarr_path)} does not exist.")

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=["img", "eef_pos", "gripper_open", "action"]
        )
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "eef_pos": self.replay_buffer["eef_pos"],
            "gripper_open": self.replay_buffer["gripper_open"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["image"] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        image = np.moveaxis(sample["img"], -1, 1) / 255

        eef_pos = sample["eef_pos"].astype(np.float32)
        gripper_open = sample["gripper_open"].astype(np.float32)

        action = sample["action"].astype(np.float32)

        if self.action_type in ["relative", "delta"]:
            # Need the previous action to have as reference
            if action.shape[-1] == eef_pos.shape[-1]:
                prev_action = eef_pos[0]
            elif action.shape[-1] == eef_pos.shape[-1] + gripper_open.shape[-1]:
                prev_action = np.concatenate((eef_pos[0], gripper_open[0]), axis=-1)
            else:
                raise ValueError(
                    f"Relative and delta actions only supported when action is the shifted eef_pos"
                    "or eef_pos and gripper_open concatenated. In this case, "
                    f"the action shape {action.shape} does not match eef_pos shape {eef_pos.shape} "
                    f"or its concatenation with the gripper_open shape {gripper_open.shape}."
                )

            if self.action_type == "relative":
                # Relative action is the difference between the current and previous eef_pos
                action = action - prev_action
            elif self.action_type == "delta":
                # Each action is relative to the previous action
                action = np.concatenate((action[:1] - prev_action, np.diff(action, axis=0)))
        elif self.action_type == "absolute":
            # Do nothing, action is already absolute
            pass

        data = {
            "obs": {
                "image": image,  # T, 3, 96, 96
                "eef_pos": eef_pos,  # T, 3
                "gripper_open": gripper_open,  # T, 1
            },
            "action": action,  # T, 3
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
