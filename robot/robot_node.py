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
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from diffusion_policy.common.pytorch_util import dict_apply

from collections import deque
from threading import Lock, Thread

from stack_msgs.srv import MoveArm
from stack_approach.helpers import get_trafo, inv, publish_img, matrix_to_pose_msg, call_cli_sync
from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace

np.set_printoptions(formatter={'float': lambda x: f"{x:.5f}"}) 

class DiffusionPolicyNode(Node):
    def __init__(self):
        super().__init__('diffusion_policy_node')
        
        self.declare_parameter('rate', 15)
        self.declare_parameter('obs_hist', 2)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('debug', True)
        self.declare_parameter('img_shape', [96,96])
        
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value
        self.rate = self.get_parameter('rate').get_parameter_value().integer_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.obs_hist = self.get_parameter('obs_hist').get_parameter_value().integer_value
        self.img_shape = self.get_parameter('img_shape').get_parameter_value().integer_array_value
        
        self.cbg = ReentrantCallbackGroup()
        
        # Initialize tf2 buffer, listener and cvbridge
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.bridge = CvBridge()

        # Initialize buffers for observations 
        self.rgb_lock = Lock()
        self.pose_lock = Lock()
        self.camera_buffer = deque(maxlen=self.obs_hist)
        self.position_buffer = deque(maxlen=self.obs_hist)
        
        while not self.tf_buffer.can_transform("map", "wrist_3_link", rclpy.time.Time()):
            print("waiting for tf ...")
            time.sleep(0.1)
            rclpy.spin_once(self)
        self.Tstart = get_trafo("map", "wrist_3_link", self.tf_buffer)
        
        self.move_cli = self.create_client(MoveArm, "move_arm")
        while not self.move_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('move arm service not available, waiting again...')
        
        self.img_sub = self.create_subscription(
            CompressedImage, "/camera/color/image_raw/compressed", self.rgb_cb, 0, callback_group=self.cbg
        )
        
        # periodically update robot pose (here: relative to starting pose)
        self.create_timer(1/(self.rate*2), self.update_pose)
        
        self.desired_pose_pub = self.create_publisher(
            PoseStamped, "/diff_desired_pose", 0
        )
        if self.debug:
            self.debug_img_pub = self.create_publisher(CompressedImage, '/camera/color/diff_debug/compressed', 0, callback_group=self.cbg)
            self.current_pose_pub = self.create_publisher(
                Float32MultiArray, "/diff_current_pose", 0
            )
            self.inference_sec_pub = self.create_publisher(
                Float32, "/diff_inference_secs", 0
            )
            
        self.load_policy()
        
    def rgb_cb(self, msg):
        try:
            # convert image to cv2, resize and normalize 
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8')
            resized_frame = cv2.resize(frame, self.img_shape, interpolation=cv2.INTER_AREA)
            resized_frame = resized_frame.astype(np.float32) / 255.0
    
            frame_tensor = torch.from_numpy(np.transpose(resized_frame, (2, 0, 1)))  # channels first
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return
        
        with self.rgb_lock:
            self.camera_buffer.append(frame_tensor)
            
    def update_pose(self):
        Tnow = get_trafo("map", "wrist_3_link", self.tf_buffer)
        pos_rel = Tnow[:3,3] + inv(self.Tstart)[:3,3]
        
        with self.pose_lock:
            self.position_buffer.appendleft(pos_rel)
        
    def load_policy(self):
        print("loading policy ....")
        payload = torch.load(open(f"{os.environ['HOME']}/repos/ckp/down2_bs128_eps10000.ckpt", 'rb'), pickle_module=dill)

        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)

        workspace: TrainDiffusionUnetHybridWorkspace = cls(cfg, output_dir="/tmp")
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device(self.device)
        policy.to(device)
        policy.eval()
        policy.reset()
        
        self.policy = policy
        print("\rloading policy .... done!")
        
    def inference(self, obs):
        with torch.no_grad():
            start = time.time()
            obs = dict_apply(obs, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
            
            # normalization of obs and unnormalization of actions happens in predict_action
            actions = self.policy.predict_action(obs)
            actions = dict_apply(actions, lambda x: torch.squeeze(x).cpu().numpy())['action']
            inference_sec = round(time.time() - start, 4)
            
            return actions, inference_sec
        
    def execute(self):
        r = self.create_rate(self.rate)
        while len(self.position_buffer) < self.obs_hist and len(self.camera_buffer) < self.obs_hist:
            print('waiting for obs ...')
            r.sleep()
            
        print("starting inference.")
        while True:
            with self.pose_lock:
                with self.rgb_lock:
                    img = np.array(self.camera_buffer)
                    pos = np.array(self.position_buffer)
            
            actions, inference_sec = self.inference({
                "image": img,
                "eef_pos": pos,
                "gripper_open": np.array(self.obs_hist*[0.]), 
            })
            
            print(actions[-1])
            Tdes = self.Tstart[:] # copy starting pose
            Tdes[:3,3] -= actions[-1] # actions are relative to start pose -> add action to translation
            
            desired_pose_msg = matrix_to_pose_msg(Tdes, "map")
            
            req = MoveArm.Request()
            req.execute = True
            req.execution_time = 1.
            req.target_pose = desired_pose_msg
            res = call_cli_sync(self, self.move_cli, req)
            
            # publish (debug) data
            self.desired_pose_pub.publish(desired_pose_msg)
            if self.debug:
                self.current_pose_pub.publish(Float32MultiArray(data=pos[-1]))
                self.inference_sec_pub.publish(Float32(data=inference_sec))
                publish_img(self.debug_img_pub, np.transpose(img[-1], (1, 2, 0))*255)
            
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