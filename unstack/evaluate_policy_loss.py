import pickle
import dill
import hydra
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import time
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.unstack_dataset import UnstackDataset
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace
register_codecs()
from data_processing import iterate_episodes

# Extract episode to lower startup time
# zarr_path = "data/unstack/unstack_cloud.zarr"

# print("loading dataset ...")
# replay_buffer = ReplayBuffer.copy_from_path(
#      zarr_path, keys=['img', 'eef_pos', 'gripper_open', 'action'])

# for episode in iterate_episodes(replay_buffer):
#     with open("ep1.pkl", "wb") as f:
#         pickle.dump(episode, f)

#     print(len(episode['imgs']), len(episode['eef_pos']))

#     break


#|o|o|                             observations: 2
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

device = "cuda:0"
obs_horizon = 3
prediction_horizon = 16

payload = torch.load(open("data/outputs/2025.01.30/15.18.29_train_diffusion_unet_hybrid_unstack/checkpoints/latest.ckpt", 'rb'), pickle_module=dill)

cfg = payload['cfg']
cls = hydra.utils.get_class(cfg._target_)

workspace : TrainDiffusionUnetHybridWorkspace = cls(cfg, output_dir="/tmp")
workspace.load_payload(payload, exclude_keys=None, include_keys=None)

# get policy from workspace
policy = workspace.model
if cfg.training.use_ema:
    policy = workspace.ema_model

device = torch.device(device)
policy.to(device)
policy.eval()

with open("ep1.pkl", "rb") as f:
    episode = pickle.load(f)


all_action = []
for i in range(len(episode["img"])): 
    if i < obs_horizon-1 : continue

    start_idx = i-(obs_horizon-1)
    batch = UnstackDataset._sample_to_data({
        "img": episode["img"][start_idx:i], 
        "eef_pos": episode["eef_pos"][start_idx:i], 
        "gripper_open": episode["gripper_open"][start_idx:i], 
        "action": episode["action"][start_idx:start_idx+prediction_horizon], 
    })

    with torch.no_grad():
        start = time.time()
        obs = dict_apply(batch['obs'], lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

        # normalization of obs and unnormalization of actions happens in predict_action
        actions = policy.predict_action(obs)
        actions = dict_apply(actions, lambda x: torch.squeeze(x))
        inference_sec = time.time() - start

        loss = torch.sqrt(F.mse_loss(torch.from_numpy(batch["action"]).to(device), actions['action_pred'][:len(batch["action"])])).item()

        all_action.append(actions['action_pred'].cpu().numpy())
        print(f"{i}: {inference_sec:.4f}s | {loss:.8f}")

with open("actions.pkl", "wb") as f:
    pickle.dump(all_action,f )

print("done")