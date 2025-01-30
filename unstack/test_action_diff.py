import numpy as np
import matplotlib.pyplot as plt

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
from diffusion_policy.common.replay_buffer import ReplayBuffer
register_codecs()

out_shape = (96,96)
zarr_path = "data/unstack/unstack_cloud.zarr"

print("loading dataset ...")
replay_buffer = ReplayBuffer.copy_from_path(
     zarr_path, keys=['img', 'eef_pos', 'gripper_open', 'action'])
print(replay_buffer.meta["episode_ends"])

prev_ee = 0
means = []
for (i, ee) in enumerate(replay_buffer.meta["episode_ends"]):
    episode = replay_buffer.data["eef_pos"][prev_ee:ee]

    diffs = np.abs(np.diff(episode, axis=0))

    plt.plot(list(range(len(diffs))), diffs)
    plt.show()

    prev_ee = ee