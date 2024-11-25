from collections import deque
from typing import Dict, Optional

import numpy as np
import torch
from octo.model.octo_model import OctoModel


class OctoPolicy:
    def __init__(self,
                 checkpoint_path: str,
                 checkpoint_step: int,
                 instruction: Optional[str] = None,
                 window_size: int = 2,
                 action_horizon: int = 4):
        self.model = OctoModel.load_pretrained(checkpoint_path, checkpoint_step)
        self.window_size = window_size
        self.action_horizon = action_horizon

        # Store language instruction
        self.instruction = instruction
        if instruction is not None:
            # Create task specification once since it won't change
            self.task = self.model.create_tasks(texts=[instruction])
        else:
            self.task = None

        # For history tracking
        self.history = deque(maxlen=self.window_size)
        self.num_obs = 0

        # For temporal ensembling
        self.act_history = deque(maxlen=self.action_horizon)
        self.exp_weight = 0

    def to(self, device: str):
        self.model.to(device)

    def reset(self):
        self.history = deque(maxlen=self.window_size)
        self.num_obs = 0
        self.act_history = deque(maxlen=self.action_horizon)

    def _internal_select_action(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process stacked observations and generate actions using the Octo model.

        Args:
            obs: Dictionary containing stacked observations with keys:
                - image_primary: [window_size, H, W, 3] RGB images
                - state: [window_size, 7] robot state vector containing:
                    - xyz position (3)
                    - rotation angles (3)
                    - gripper state (1)
                - timestep_pad_mask: [window_size] binary mask for valid timesteps

        Returns:
            actions: [sequence_length, action_dim] predicted action sequence
        """
        # Format observations for the model
        model_obs = {}

        # Normalize image: scale to [0,1] and normalize using dataset statistics
        image = obs["image_primary"].permute(0, 3, 1, 2).float() / 255.0
        image_stats = self.model.dataset_statistics["bridge_dataset"]["image_primary"]
        image = (image - image_stats["mean"]) / image_stats["std"]
        model_obs["image_primary"] = image

        # Normalize state using dataset statistics
        state = obs["state"].float()
        state_stats = self.model.dataset_statistics["bridge_dataset"]["state"]
        state = (state - state_stats["mean"]) / state_stats["std"]
        model_obs["state"] = state

        model_obs["timestep_pad_mask"] = obs["timestep_pad_mask"]

        # Get action predictions from model
        with torch.no_grad():
            actions = self.model.sample_actions(
                observations=model_obs,
                tasks=self.task,  # Language instruction task created in constructor
                rng=None,  # Deterministic sampling
                argmax=True,  # Take most likely action
                unnormalization_statistics=self.model.dataset_statistics["bridge_dataset"]["action"]
            )

        return actions[0]  # Remove batch dimension only

    def select_action(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Ensure observation contains required keys
        required_keys = {"image_primary", "state"}
        missing_keys = required_keys - set(obs.keys())
        if missing_keys:
            raise KeyError(f"Observation missing required keys: {missing_keys}")

        # History wrapping
        self.num_obs += 1
        self.history.append(obs)

        # Pad history if needed
        while len(self.history) < self.window_size:
            self.history.appendleft(obs)

        # Stack observations
        stacked_obs = {}
        for k in obs:
            stacked_obs[k] = torch.stack([d[k] for d in self.history])

        # Add padding mask - False for padded timesteps, True for real observations
        pad_length = self.window_size - min(self.num_obs, self.window_size)
        timestep_pad_mask = torch.ones(self.window_size, dtype=torch.bool)
        timestep_pad_mask[:pad_length] = False  # Mark padded timesteps as False
        stacked_obs["timestep_pad_mask"] = timestep_pad_mask

        # Get action predictions
        actions = self._internal_select_action(stacked_obs)

        # Temporal ensembling
        self.act_history.append(actions[:self.action_horizon])
        num_actions = len(self.act_history)

        # Get predictions for current timestep
        curr_act_preds = torch.stack([
            pred_actions[i] for i, pred_actions in zip(
                range(num_actions-1, -1, -1),
                self.act_history
            )
        ])

        # Weight predictions (more recent predictions get exponentially less weight)
        weights = torch.exp(-self.exp_weight * torch.arange(num_actions, dtype=torch.float32))
        weights = weights / weights.sum()

        # Compute weighted average
        action = (weights[:, None] * curr_act_preds).sum(dim=0)

        return action
