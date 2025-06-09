import json
import pathlib

import numpy as np
import scipy.spatial.transform as st
from pose_trajectory_interpolator import PoseTrajectoryInterpolator


def get_episode_info(path):
    required_files = ["rgb.mp4", "depth.mp4", "misc.json", "rgb_stamps.json", "gripper_poses.json"]

    # check whether all necessary files exist
    if not np.all([path.joinpath(fn).is_file() for fn in required_files]):
        print(path, "is missing data")
        exit(1)

    with open(path.joinpath("rgb_stamps.json"), "r") as f:
        rgb_stamps = json.load(f)

    with open(path.joinpath("misc.json"), "r") as f:
        misc = json.load(f)
        gripper_close_time = misc["gripper_close_time"]

    with open(path.joinpath("gripper_poses.json"), "r") as f:
        raw = json.load(f)

    ts = np.array([d[0] for d in raw])
    eef_poses = np.array(
        [np.concatenate([pose[2], st.Rotation.from_quat(pose[3]).as_rotvec()]) for pose in raw]
    )

    # Interpolator expects timestamps to be in STRICT ascending order
    if not np.all(np.diff(ts) > 0):
        # Check that timestamps are at least in ascending order
        assert np.all(np.diff(ts) >= 0), "Timestamps must be in ascending order"
        # Check if duplicates have the same pose values
        duplicated_indices = np.where(np.diff(ts) == 0)[0]
        assert np.array_equal(
            eef_poses[duplicated_indices], eef_poses[duplicated_indices + 1]
        ), "Duplicate timestamps must have the same pose values"

        # Delete duplicates
        _, unique_indices = np.unique(ts, return_index=True)
        ts = ts[unique_indices]
        eef_poses = eef_poses[unique_indices]

    delay = (
        ts[-1] - rgb_stamps[-1]
    )  # there was delay in the timestamps. this fixes it. an absolute HACK
    rgb_stamps = rgb_stamps + delay

    interpolator = PoseTrajectoryInterpolator(times=ts, poses=eef_poses)
    interpolated_eef_poses = interpolator(rgb_stamps)

    # Discard the last observation as it has no action associated with it
    episode_data = dict()
    episode_data["eef_pos"] = interpolated_eef_poses[:-1, :3].astype(np.float32)
    episode_data["eef_rot_axis_angle"] = interpolated_eef_poses[:-1, 3:].astype(np.float32)
    episode_data["gripper_open"] = np.expand_dims(
        np.array(rgb_stamps) < gripper_close_time, axis=-1
    )[:-1].astype(np.uint8)
    # Actions are the next EEF pose
    episode_data["action"] = interpolated_eef_poses[1:]

    episode_info = {
        "episode_data": episode_data,
        "video_path": path / "rgb.mp4",
        "frame_start": 0,
        "frame_end": len(rgb_stamps) - 1,
        "rgb_stamps": rgb_stamps[:-1],
    }

    return episode_info


def process_data_directories(data_dirs):
    episodes_info = []
    for dir_path in data_dirs:
        path = pathlib.Path(dir_path)
        episode_info = get_episode_info(path)
        if episode_info is not None:
            episodes_info.append(episode_info)
    return episodes_info


def iterate_episodes(replay_buffer):
    prev_ee = 0
    for ee in replay_buffer.meta["episode_ends"]:
        yield {
            "img": replay_buffer.data["img"][prev_ee:ee],
            "eef_pos": replay_buffer.data["eef_pos"][prev_ee:ee],
            "gripper_open": replay_buffer.data["gripper_open"][prev_ee:ee],
            "action": replay_buffer.data["action"][prev_ee:ee],
        }
        prev_ee = ee
