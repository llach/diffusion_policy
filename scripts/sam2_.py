import os
import json
import torch 

from pathlib import Path
from processor import ImageProcessor

from sam2_model import SAM2Model

def get_sam2_fn():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SAM2Model()

    def fn(in_path: Path, out_path: Path):

        with open(str(out_path).replace(".png", ".json"), "w") as f:
            json.dump([], f)

        return []
    
    return fn

    

if __name__ == "__main__":
    # Define input directory and new folder name
    input_dir = f"{os.environ['HOME']}/repos/unstack_classify/raw/"

    # Create an instance of ImageProcessor with dry_run option
    processor = ImageProcessor(input_dir, "sam2", get_sam2_fn())

    # Run the processing
    processor.process_images(dry_run=False)

    