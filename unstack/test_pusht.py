
# limit enviornment interaction to 200 steps before termination
import os 
import gdown
import collections

import numpy as np
import torch
from tqdm import tqdm
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv

device = "cpu"
max_steps = 200

dataset_path = "pusht_cchi_v7_replay.zarr.zip"
if not os.path.isfile(dataset_path):
    id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
    gdown.download(id=id, output=dataset_path, quiet=False)

# parameters
pred_horizon = 16
obs_horizon = 2
action_horizon = 8
#|o|o|                             observations: 2
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

# create dataset from file
print("loading dataset")
dataset = PushTImageDataset(
    zarr_path=dataset_path,
    horizon=pred_horizon,
)
normalizer = dataset.get_normalizer()

for i in tqdm(range(max_steps), desc="Eval PushTImageEnv"):
    B = 1

    sample = dataset[i]

    # normalized inputs and outputs
    nimages = sample['obs']['image'] # images are normalized in sampling fn
    nagent_poses = normalizer.normalize({"agent_pos": sample['obs']['agent_pos']})

    # device transfer
    nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
    # (2,3,96,96)
    nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
    # (2,2)

    print("eval")
    exit(0)

    # infer action
    with torch.no_grad():
        # get image features
        image_features = ema_nets['vision_encoder'](nimages)
        # (2,512)

        # concat with low-dim observations
        obs_features = torch.cat([image_features, nagent_poses], dim=-1)

        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, pred_horizon, action_dim), device=device)
        naction = noisy_action

        # init scheduler
        noise_scheduler.set_timesteps(num_diffusion_iters)

        for k in noise_scheduler.timesteps:
            # predict noise
            noise_pred = ema_nets['noise_pred_net'](
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

    # unnormalize action
    naction = naction.detach().to('cpu').numpy()
    # (B, pred_horizon, action_dim)
    naction = naction[0]
    action_pred = unnormalize_data(naction, stats=stats['action'])

    # only take action_horizon number of actions
    start = obs_horizon - 1
    end = start + action_horizon
    action = action_pred[start:end,:]
    # (action_horizon, action_dim)


# print out the maximum target coverage
print('Score: ', max(rewards))

