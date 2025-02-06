import zarr 
import numpy as np

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
from diffusion_policy.common.replay_buffer import ReplayBuffer
register_codecs()

from data_processing import iterate_episodes


downsample = 3
output = f"unstack_cloud_down{downsample}.zarr"

print("loading dataset ...")
zarr_path = "data/unstack/unstack_cloud.zarr"
replay_buffer = ReplayBuffer.copy_from_path(
     zarr_path, keys=['img', 'eef_pos', 'gripper_open', 'action'])

out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore())


imgs = []
for i, eps in enumerate(iterate_episodes(replay_buffer)):

    print(f"inserting episode {i}")
    out_replay_buffer.add_episode(data={
        'eef_pos': eps['eef_pos'][::downsample],
        'gripper_open': eps['gripper_open'][::downsample],
        'action': eps['action'][::downsample],
    }, compressors=None)

    imgs.append(eps['img'][::downsample])

img_arr = np.concatenate(imgs)
img_compressor = JpegXl(level=99, numthreads=1)
_ = out_replay_buffer.data.require_dataset(
        name="img",
        shape=img_arr.shape,
        chunks=(1,) + img_arr.shape[1:],
        compressor=img_compressor,
        dtype=np.uint8
    )
out_replay_buffer.data['img'] = img_compressor.encode(img_arr)

with zarr.ZipStore(f"{output}.zip", mode='w') as zip_store:
    out_replay_buffer.save_to_store(
        store=zip_store
    )