import os
import sys
import cv2
import time
import dill
import hydra
import torch
import numpy as np

import rclpy

from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from collections import deque
from threading import Lock, Thread

from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace
from stack_approach.helpers import get_trafo, inv

class DiffusionPolicyNode(Node):
    def __init__(self):
        super().__init__('diffusion_policy_node')
        
        self.declare_parameter('rate', 15)
        self.declare_parameter('device', 'gpu')
        self.declare_parameter('img_shape', [96,96])
        
        self.rate = self.get_parameter('rate').get_parameter_value().integer_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.img_shape = self.get_parameter('img_shape').get_parameter_value().integer_array_value
        
        self.cbg = ReentrantCallbackGroup()
        
        # Initialize tf2 buffer, listener and cvbridge
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.bridge = CvBridge()

        # Initialize buffers for observations with history of 2
        self.rgb_lock = Lock()
        self.pose_lock = Lock()
        self.camera_buffer = deque(maxlen=2)
        self.position_buffer = deque(maxlen=2)
        
        while not self.tf_buffer.can_transform("map", "wrist_3_link", rclpy.time.Time()):
            print("waiting for tf ...")
            time.sleep(0.05)
            rclpy.spin_once(self)
        self.Tstart = get_trafo("map", "wrist_3_link", self.tf_buffer)
        
        self.img_sub = self.create_subscription(
            CompressedImage, "/camera/color/image_raw/compressed", self.rgb_cb, 0, callback_group=self.cbg
        )
        
        self.create_timer(1/(self.rate*2), self.update_pose, self.cbg)
        

        # self.load_policy()
        
        # TODO get tf
        
    def rgb_cb(self, msg):
        with self.rgb_lock:
            try:
                # convert image to cv2, resize and normalize TODO do we have to rotate here?
                frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8')
                resized_frame = cv2.resize(frame, self.img_shape, interpolation=cv2.INTER_AREA)
                resized_frame = resized_frame.astype(np.float32) / 255.0
        
                frame_tensor = torch.from_numpy(np.transpose(resized_frame, (2, 0, 1)))  # channels first
                self.camera_buffer.append(frame_tensor)
            except CvBridgeError as e:
                self.get_logger().error(f"CvBridge Error: {e}")
                return
            
    def update_pose(self):
        with self.pose_lock:
            Tnow = get_trafo("map", "wrist_3_link", self.tf_buffer)
            print(Tnow@inv(self.Tstart))
        
    def load_policy(self):
        print("loading policy ....")
        payload = torch.load(open(f"{os.environ['HOME']}/repos/ckp/down2_bs128_eps10000.ckpt", 'rb'), pickle_module=dill)

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
        print("\rloading policy .... done!")
        
    def execute(self):
        r = self.create_rate(self.rate)
        while True:
            print("still waiting ...")
            r.sleep()

def main(args=None):
    rclpy.init(args=args)
    node = DiffusionPolicyNode()
    
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    # Start the executor in a separate thread
    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    try:
        node.execute()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main(args=sys.argv)  # Pass command-line arguments to rclpy
