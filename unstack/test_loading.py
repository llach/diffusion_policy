import zarr
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
from diffusion_policy.common.replay_buffer import ReplayBuffer
register_codecs()

import zarr
from skimage.transform import resize
from tqdm import tqdm


out_shape = (96,96)
zarr_path = "data/unstack/unstack_cloud_new.zarr"
zarr_path_out = zarr_path.replace(".zarr", "_small.zarr.zip")

print("loading dataset ...")
replay_buffer = ReplayBuffer.copy_from_path(
     zarr_path, keys=['img', 'eef_pos', 'gripper_open', 'actions'])

print("copying low dim keys ...")
out_replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
for ld_key in ['eef_pos', 'gripper_open', 'actions']:
    ds = replay_buffer.data[ld_key]
    chunks = (1,) + ds.shape[1:]
    out_replay_buffer.data.require_dataset(
        name=ld_key,
        shape=ds.shape,
        chunks=chunks,
        compressor=None,
        dtype=ds.dtype
    )
    out_replay_buffer.data[ld_key] = ds

out_replay_buffer.update_meta({
    "episode_ends": replay_buffer.meta["episode_ends"]
})

print("creating img dataset ...")
original_images = replay_buffer.data['img']  # Shape: (24102, 224, 299, 3)
img_compressor = JpegXl(level=99, numthreads=1)
out_replay_buffer.data.require_dataset(
    name="img",
    shape=(original_images.shape[0],) + out_shape + (3,),
    chunks=(1,)+out_shape+(3,),
    compressor=img_compressor,
    dtype=original_images.dtype
)
new_images = out_replay_buffer.data['img']

# Resize each image and save to the new dataset
print("resizing ...")
for i in tqdm(range(original_images.shape[0])):
    # Resize using skimage (maintains channels, anti-aliasing enabled)
    resized_img = resize(original_images[i], out_shape, preserve_range=True, anti_aliasing=True)
    resized_img = resized_img.astype(original_images.dtype)  # Ensure correct dtype
    new_images[i] = resized_img

print("saving ...")
with zarr.ZipStore(zarr_path_out, mode='w') as zip_store:
    out_replay_buffer.save_to_store(
        store=zip_store
    )

print("done!")

