"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import pathlib
import sys

import hydra
from omegaconf import OmegaConf

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

register_codecs()

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy", "config")),
    config_name="train_diffusion_unet_hybrid_workspace.yaml",
)
def main(cfg: OmegaConf):
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
