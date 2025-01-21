import os 
import gdown
import torch 
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset

if __name__ == "__main__":
    # download demonstration data from Google Drive
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

    print("creating dataloader")
    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )

    # visualize data in batch
    batch = next(iter(dataloader))
    print("batch['obs']['image'].shape:", batch['obs']['image'].shape)
    print("batch['obs']['agent_pos'].shape", batch['obs']['agent_pos'].shape)
    print("batch['action'].shape:", batch['action'].shape)