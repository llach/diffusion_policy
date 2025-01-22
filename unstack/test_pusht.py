import os
import copy
import gdown
import torch 
import numpy as np

from tqdm import tqdm
from torch import nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.lr_scheduler import get_scheduler
from network_fns import get_resnet, replace_bn_with_gn
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset

device = "cpu"
max_steps = 200

dataset_path = "pusht_cchi_v7_replay.zarr.zip"
if not os.path.isfile(dataset_path):
    id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
    gdown.download(id=id, output=dataset_path, quiet=False)

#|o|o|                             observations: 2
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
pred_horizon = 16
obs_horizon = 2
action_horizon = 8

# create dataset from file
print("loading dataset")
dataset = PushTImageDataset(
    zarr_path=dataset_path,
    horizon=pred_horizon,
)
normalizer = dataset.get_normalizer()

 # construct ResNet18 encoder
# if you have multiple camera views, use seperate encoder weights for each view.
vision_encoder = get_resnet('resnet18')

# IMPORTANT!
# replace all BatchNorm with GroupNorm to work with EMA
# performance will tank if you forget to do this!
vision_encoder = replace_bn_with_gn(vision_encoder)

# ResNet18 has output dim of 512
vision_feature_dim = 512
# agent_pos is 2 dimensional
lowdim_obs_dim = 2
# observation feature has 514 dims in total per step
obs_dim = vision_feature_dim + lowdim_obs_dim
action_dim = 2

# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

# the final arch has 2 parts
nets = nn.ModuleDict({
    'vision_encoder': vision_encoder,
    'noise_pred_net': noise_pred_net
})

load_pretrained = True
if load_pretrained:
  ckpt_path = "pusht_vision_100ep.ckpt"
  if not os.path.isfile(ckpt_path):
      id = "1XKpfNSlwYMGaF5CncoFaLKCDTWoLAHf1&confirm=t"
      gdown.download(id=id, output=ckpt_path, quiet=False)

  state_dict = torch.load(ckpt_path, map_location=device)
  ema_nets = nets
  ema_nets.load_state_dict(state_dict)
  print('Pretrained weights loaded.')
else:
  print("Skipped pretrained weight loading.")

num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

for i in tqdm(range(max_steps), desc="Eval PushTImageEnv"):
    B = 1

    sample = dataset[i]

    # normalized inputs and outputs
    nimages = sample['obs']['image'] # images are normalized in sampling fn
    nagent_poses = normalizer["agent_pos"](sample['obs']['agent_pos'])

    # device transfer
    nimages = nimages.to(device, dtype=torch.float32) # (2,3,96,96)
    nagent_poses = nagent_poses.to(device, dtype=torch.float32) # (2,2)

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

    print("hi")
    exit(0)
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

