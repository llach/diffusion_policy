import os
import cv2
import json
import torch 
import dill as pickle

import numpy as np

from PIL import Image
from pathlib import Path
from processor import ImageProcessor

from sam2_model import SAM2Model
from groundingdino.util.inference import load_model, load_image, predict, annotate

def get_sam2_fn():
    sam_model = SAM2Model()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model("GroundingDINO_SwinT_OGC.py", "groundingdino_swint_ogc.pth", device=device)
    TEXT_PROMPT = "detect stacks of clothing"
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    def fn(in_path: Path, out_path: Path):
        image_source, image = load_image(in_path)
        H, W = image_source.shape[:2]

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device=device
        )

        boxes_px = boxes * np.repeat([[W, H, W, H]], boxes.shape[0], axis=0)
        boxes_px[:,:2] -= boxes_px[:,2:] / 2
        boxes_px[:,2:] += boxes_px[:,:2]
        boxes_px = np.array(boxes_px, dtype=np.int16)


        input_img = Image.open(in_path)
        img_raw = np.array(input_img)
        masks = sam_model.predict(img_raw)

        img_overlay, _, line_center = SAM2Model.detect_stack(img_raw, masks, boxes_px[0])

        with open(str(out_path).replace(".png", ".pkl"), "wb") as f:
            pickle.dump({"masks": masks, "boxes": boxes}, f)

        return cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
    
    return fn

    

if __name__ == "__main__":
    # Define input directory and new folder name
    input_dir = f"{os.environ['HOME']}/repos/unstack_classify/raw/"

    # Create an instance of ImageProcessor with dry_run option
    processor = ImageProcessor(input_dir, "sam2", get_sam2_fn())

    # Run the processing
    processor.process_images(dry_run=False)

    