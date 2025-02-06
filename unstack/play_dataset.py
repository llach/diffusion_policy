from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
from diffusion_policy.common.replay_buffer import ReplayBuffer
register_codecs()

import cv2
import numpy as np

def show_imgs(image_sequence = np.random.randint(0, 255, (100, 200, 200, 3), dtype=np.uint8)):

    # Loop through frames and display
    for frame in image_sequence:
        # OpenCV expects BGR instead of RGB; convert if your array is RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', frame_bgr)
        
        # Press 'q' to exit
        if cv2.waitKey(30) & 0xFF == ord('q'):  # Adjust delay (30ms ~ 33 FPS)
            break

    cv2.destroyAllWindows()


print("loading dataset ...")
# zarr_path = "data/unstack/unstack_cloud.zarr"
zarr_path = "unstack_cloud_down1.zarr"
replay_buffer = ReplayBuffer.copy_from_path(
     zarr_path, keys=['img', 'eef_pos', 'gripper_open', 'action'])

print(replay_buffer.data["img"].shape)
show_imgs(replay_buffer.data["img"])