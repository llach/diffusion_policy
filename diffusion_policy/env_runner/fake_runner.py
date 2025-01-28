from typing import Dict
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

class FakeRunner(BaseImageRunner):
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BaseImagePolicy) -> Dict:
        return {}
