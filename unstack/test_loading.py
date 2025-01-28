import zarr
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

from diffusion_policy.common.replay_buffer import ReplayBuffer

zarr_path = "data/unstack/unstack_cloud_new.zarr"
replay_buffer = ReplayBuffer.copy_from_path(
    zarr_path, keys=['img', 'eef_pos', 'gripper_open', 'actions'])
pass