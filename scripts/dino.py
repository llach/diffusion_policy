import os
import json
import torch 

from groundingdino.util.inference import load_model, load_image, predict, annotate
from pathlib import Path

from processor import ImageProcessor

def get_dino_fn():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model("GroundingDINO_SwinT_OGC.py", "groundingdino_swint_ogc.pth", device=device)
    TEXT_PROMPT = "detect stacks of clothing"
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25


    def fn(in_path: Path, out_path: Path):
        image_source, image = load_image(in_path)

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device=device
        )

        with open(str(out_path).replace(".png", ".json"), "w") as f:
            json.dump(boxes.to("cpu").numpy().tolist(), f)

        return annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    
    return fn

    

if __name__ == "__main__":
    # Define input directory and new folder name
    input_dir = f"{os.environ['HOME']}/repos/unstack_classify/raw/"

    # Create an instance of ImageProcessor with dry_run option
    processor = ImageProcessor(input_dir, "dino", get_dino_fn())

    # Run the processing
    processor.process_images(dry_run=False)

    