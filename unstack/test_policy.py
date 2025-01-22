import os
import dill
import torch
import hydra
import gdown
import torch.nn.functional as F

from tqdm import tqdm

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace


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
device = "cpu"
max_steps = 10


# create dataset from file
print("loading dataset")
dataset = PushTImageDataset(
    zarr_path=dataset_path,
    horizon=pred_horizon,
)

payload = torch.load(open("data/pusht_test/epoch=1850-test_mean_score=0.898.ckpt", 'rb'), pickle_module=dill)

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

policy.reset()
for i in tqdm(range(max_steps), desc="Eval PushTImageEnv"):
    B = 1

    sample = dataset[i]
    actions_gt = sample['action']
    with torch.no_grad():

        # first, cut according to obs horizon, then unsqueeze to have batch dimension of 1
        # cutting the obs_horizon is technically also done in predict_action, we do it here nevertheless so it's less confusing
        obs = dict_apply(sample['obs'], lambda x: x[:obs_horizon,:].unsqueeze(0))

        # normalization of obs and unnormalization of actions happens in predict_action
        actions = policy.predict_action(obs)
        actions = dict_apply(actions, lambda x: torch.squeeze(x))

        print(F.l1_loss(actions_gt, actions['action_pred']))

print("done.")