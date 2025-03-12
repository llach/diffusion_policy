import os
import json
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm

def preprocess_image(image_path, bbox, target_size=224):
    """
    Preprocess an image: crop, resize maintaining aspect ratio, pad to square, normalize.
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    # Get image dimensions
    height, width = img_array.shape[:2]

    # Extract bounding box (normalized coordinates: [x_center, y_center, width, height])
    x_center, y_center, bbox_width, bbox_height = bbox
    x_center, y_center = x_center * width, y_center * height
    bbox_width, bbox_height = bbox_width * width, bbox_height * height

    # Convert to top-left and bottom-right corners
    x1 = int(max(0, x_center - bbox_width / 2))
    y1 = int(max(0, y_center - bbox_height / 2))
    x2 = int(min(width, x_center + bbox_width / 2))
    y2 = int(min(height, y_center + bbox_height / 2))

    # Crop the image
    cropped_img = img_array[y1:y2, x1:x2]

    # Resize maintaining aspect ratio
    cropped_img = Image.fromarray(cropped_img)
    orig_width, orig_height = cropped_img.size
    ratio = min(target_size / orig_width, target_size / orig_height)
    new_width = int(orig_width * ratio)
    new_height = int(orig_height * ratio)
    resized_img = cropped_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Pad to target_size x target_size
    padded_img = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    offset_x = (target_size - new_width) // 2
    offset_y = (target_size - new_height) // 2
    padded_img.paste(resized_img, (offset_x, offset_y))

    # Convert to numpy and normalize
    img_array = np.array(padded_img) / 255.0  # Normalize to [0, 1]
    return img_array

def main(parent_dir):
    raw_base_dir = os.path.join(parent_dir, 'raw')
    dino_base_dir = os.path.join(parent_dir, 'dino')
    output_dir = os.path.join(parent_dir, 'stack_classify')
    os.makedirs(output_dir, exist_ok=True)

    # Get list of classes
    classes = sorted([d for d in os.listdir(raw_base_dir) if os.path.isdir(os.path.join(raw_base_dir, d))])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    images = []
    labels = []

    # Process each class
    for cls_name in tqdm(classes, desc="Processing classes"):
        raw_dir = os.path.join(raw_base_dir, cls_name)
        dino_dir = os.path.join(dino_base_dir, cls_name)

        if not os.path.isdir(raw_dir) or not os.path.isdir(dino_dir):
            continue

        # Process each image in the class
        for img_name in os.listdir(raw_dir):
            if not img_name.endswith('.png'):
                continue

            img_path = os.path.join(raw_dir, img_name)
            json_name = img_name.replace('.png', '.json')
            json_path = os.path.join(dino_dir, json_name)

            if not os.path.exists(json_path):
                print(f"Warning: JSON file {json_path} not found, skipping {img_name}")
                continue

            # Load bounding box
            with open(json_path, 'r') as f:
                bboxes = json.load(f)
            if not bboxes:
                print(f"Warning: No bounding boxes in {json_path}, skipping {img_name}")
                continue

            # Use the first bounding box
            bbox = bboxes[0]

            # Preprocess image
            try:
                img_array = preprocess_image(img_path, bbox)
                images.append(img_array)
                labels.append(class_to_idx[cls_name])
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    # Convert to numpy arrays
    images = np.array(images)  # Shape: (num_images, 224, 224, 3)
    labels = np.array(labels)  # Shape: (num_images,)

    # Save to output directory
    np.save(os.path.join(output_dir, 'images.npy'), images)
    np.save(os.path.join(output_dir, 'labels.npy'), labels)

    # Save class mapping
    with open(os.path.join(output_dir, 'class_to_idx.json'), 'w') as f:
        json.dump(class_to_idx, f)

    print(f"Preprocessed dataset saved to {output_dir}")
    print(f"Total images processed: {len(images)}")
    print(f"Classes: {list(class_to_idx.keys())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for classification")
    parser.add_argument("parent_dir", type=str, help="Parent directory containing raw/ and dino/")
    args = parser.parse_args()
    main(args.parent_dir)