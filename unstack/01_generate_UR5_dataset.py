import concurrent.futures
import multiprocessing
import os
import pathlib
from collections import defaultdict

import av
import click
import cv2
import numpy as np
import zarr
from data_processing import get_episode_info
from skimage.transform import resize
from tqdm import tqdm

from diffusion_policy.codecs.imagecodecs_numcodecs import JpegXl, register_codecs
from diffusion_policy.common.replay_buffer import ReplayBuffer

register_codecs()


datefmt = "%Y.%m.%d_%H_%M_%S"


def get_closest_idx(stamp, stamps):
    return np.argmin(np.abs(stamps - stamp))


@click.command()
@click.argument("in_path", default="/home/obarbany/code/diffusion_policy/data/raw_unstack_cloud")
@click.option(
    "-o",
    "--output",
    help="Zarr path",
    default="/home/obarbany/code/diffusion_policy/data/unstack/unstack_cloud_224.zarr",
)
@click.option("-or", "--out_res", type=str, default="224,224")
@click.option("-cl", "--compression_level", type=int, default=99)
@click.option("-n", "--num_workers", type=int, default=None)
def main(in_path, output, out_res, compression_level, num_workers):
    if os.path.isfile(output):
        if click.confirm(f"Output file {output} exists! Overwrite?", abort=True):
            pass

    out_res = tuple(int(x) for x in out_res.split(","))

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    cv2.setNumThreads(1)

    out_replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())

    # dump lowdim data to replay buffer
    # generate argumnet for videos
    buffer_start = 0
    all_videos = set()
    vid_args = list()

    for ipath in os.listdir(in_path):
        path = pathlib.Path(os.path.join(in_path, ipath)).absolute()
        print(path)
        if path.is_file():
            continue

        episode_info = get_episode_info(path)
        episode_data, rgb_stamps = episode_info["episode_data"], episode_info["rgb_stamps"]

        out_replay_buffer.add_episode(data=episode_data, compressors=None)

        videos_dict = defaultdict(list)

        # aggregate video gen aguments
        n_frames = None
        video_path = path.joinpath("rgb.mp4")
        assert video_path.is_file()

        n_frames = len(rgb_stamps)

        videos_dict[str(video_path)].append(
            {"camera_idx": 0, "frame_start": 0, "frame_end": n_frames, "buffer_start": buffer_start}
        )
        buffer_start += n_frames

        vid_args.extend(videos_dict.items())
        all_videos.update(videos_dict.keys())

    print(f"{len(all_videos)} videos used in total!")

    # dump images
    img_compressor = JpegXl(level=compression_level, numthreads=1)
    _ = out_replay_buffer.data.require_dataset(  # type: ignore
        name="img",
        shape=(out_replay_buffer["eef_pos"].shape[0],) + out_res + (3,),
        chunks=(1,) + out_res + (3,),
        compressor=img_compressor,
        dtype=np.uint8,
    )

    def video_to_zarr(replay_buffer, mp4_path, tasks):
        tasks = sorted(tasks, key=lambda x: x["frame_start"])
        camera_idx = None
        for task in tasks:
            if camera_idx is None:
                camera_idx = task["camera_idx"]
            else:
                assert camera_idx == task["camera_idx"]
        img_array = replay_buffer.data["img"]

        curr_task_idx = 0

        with av.open(mp4_path) as container:
            in_stream = container.streams.video[0]
            # in_stream.thread_type = "AUTO"
            in_stream.thread_count = 1
            buffer_idx = 0
            for frame_idx, frame in tqdm(
                enumerate(container.decode(in_stream)), total=in_stream.frames, leave=False
            ):
                if curr_task_idx >= len(tasks):
                    # all tasks done
                    break

                if frame_idx < tasks[curr_task_idx]["frame_start"]:
                    # current task not started
                    continue
                elif frame_idx < tasks[curr_task_idx]["frame_end"]:
                    if frame_idx == tasks[curr_task_idx]["frame_start"]:
                        buffer_idx = tasks[curr_task_idx]["buffer_start"]

                    # do current task
                    frame = frame.to_ndarray(format="rgb24")
                    img = resize(frame, out_res, preserve_range=True, anti_aliasing=True)
                    img.astype(frame.dtype)

                    # compress image
                    img_array[buffer_idx] = img
                    buffer_idx += 1

                    if (frame_idx + 1) == tasks[curr_task_idx]["frame_end"]:
                        # current task done, advance
                        curr_task_idx += 1
                else:
                    assert False
        return mp4_path

    with tqdm(total=len(vid_args)) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for mp4_path, tasks in vid_args:
                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    pbar.update(len(completed))

                futures.add(executor.submit(video_to_zarr, out_replay_buffer, mp4_path, tasks))

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    print([x.result() for x in completed])

    # dump to disk
    print(f"Saving ReplayBuffer to {output}")
    out_replay_buffer.save_to_path(output, mode="w", storage=zarr.DirectoryStore(output))
    print(f"Done! {len(all_videos)} videos used in total!")
    print(f"n_steps {out_replay_buffer.n_steps} | n_episodes {out_replay_buffer.n_episodes}")


if __name__ == "__main__":
    main()
