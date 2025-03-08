import os
import torch
import numpy as np

from sam2_model import draw_anns

from PIL import Image, ImageDraw, ImageFont
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def create_output_folder(input_folder):
    output_folder = os.path.join(os.path.dirname(input_folder), "segmented_results")
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

def get_font():
    try:
        return ImageFont.truetype("arial.ttf", 12)  # Try to use Arial, fall back if not available
    except:
        return ImageFont.load_default()

def add_text_to_image(image, text):
    draw = ImageDraw.Draw(image)
    font = get_font()
    text_position = (10, 10)
    draw.text(text_position, text, fill="white", font=font)
    return image

def run_sam2_with_params(image_path, params, output_folder, index):
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Build SAM2 model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml",  f"sam2.1_hiera_large.pt", device = device)

    # Initialize mask generator with current parameters
    mask_generator = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=params["points_per_side"],
        points_per_batch=params["points_per_batch"],
        pred_iou_thresh=params["pred_iou_thresh"],
        stability_score_thresh=params["stability_score_thresh"],
        stability_score_offset=params["stability_score_offset"],
        crop_n_layers=params.get("crop_n_layers", 0),  # Default to 0 if not set
        box_nms_thresh=params.get("box_nms_thresh", 0.7),  # Default value
        crop_n_points_downscale_factor=params.get("crop_n_points_downscale_factor", 4),
        min_mask_region_area=params["min_mask_region_area"],
        use_m2m=params.get("use_m2m", False)  # Default to False if not set
    )

    # Generate masks
    masks = mask_generator.generate(image_np)

    # For simplicity, we'll visualize the first mask (you can modify to show all or process further)
    if masks:
        mask_w = 0.4
        img_anns = draw_anns(masks)
        img_overlay = Image.fromarray(np.clip((1-mask_w)*image_np + mask_w*img_anns, 0, 255).astype(np.uint8), mode="RGB")
        
        # Create parameter text
        param_text = (
            f"points_per_side={params['points_per_side']}\n"
            f"pred_iou_thresh={params['pred_iou_thresh']}\n"
            f"stability_score_thresh={params['stability_score_thresh']}\n"
            f"min_mask_region_area={params['min_mask_region_area']}\n"
            f"crop_n_layers={params.get('crop_n_layers', 0)}\n"
            f"use_m2m={params.get('use_m2m', False)}"
        )

        # Add text to the mask image
        final_image = add_text_to_image(img_overlay, param_text)

        # Save with index
        filename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_folder, f"{filename}_{index}.png")
        final_image.save(output_path)
        print(f"Saved result for {image_path} with index {index}")

def main(input_folder):
    # Define hyperparameter grid
    param_grid = {
        "checkpoint": "sam2.1_hiera_large.pt",
        "model_cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "points_per_side": [24, 32, 48],  # Test current and increased
        "points_per_batch": [44, 55, 66],  # Keep as is for now
        "pred_iou_thresh": [0.5, 0.6, 0.7],  # Test current and higher
        "stability_score_thresh": [0.8, 0.85, 0.9],  # Test current and higher
        "stability_score_offset": [0.7],  # Keep as is
        "min_mask_region_area": [25.0, 100.0],  # Test current and higher
        "crop_n_layers": [0, 1],  # Test with and without cropping
        "crop_n_points_downscale_factor": [4, 2],  # Test current and lower
        "use_m2m": [False, True],  # Test with and without
    }

    # Create output folder
    output_folder = create_output_folder(input_folder)

    # Get all image files
    image_extensions = (".png", ".jpg", ".jpeg")
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]

    # Generate all combinations of parameters
    from itertools import product
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    param_combinations = list(product(*param_values))

    print(image_files)

    # Process each image
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        index = 0
        for params_tuple in param_combinations:
            params = dict(zip(param_names, params_tuple))
            run_sam2_with_params(image_path, params, output_folder, index)
            index += 1

if __name__ == "__main__":
    main(f"{os.environ['HOME']}/repos/unstack_classify/sam_test/")