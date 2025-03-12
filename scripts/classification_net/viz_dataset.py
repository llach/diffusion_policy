import os
import json
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from collections import defaultdict

def main(parent_dir):
    # Define paths
    data_dir = os.path.join(parent_dir, 'stack_classify')
    images_path = os.path.join(data_dir, 'images.npy')
    labels_path = os.path.join(data_dir, 'labels.npy')
    class_to_idx_path = os.path.join(data_dir, 'class_to_idx.json')

    # Load the data
    images = np.load(images_path)
    labels = np.load(labels_path)
    with open(class_to_idx_path, 'r') as f:
        class_to_idx = json.load(f)

    # Create inverse mapping (index to class name)
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}

    # Group images by class
    class_images = defaultdict(list)
    for img, lbl in zip(images, labels):
        class_images[lbl].append(img)

    # Convert lists to numpy arrays for easier indexing
    for lbl in class_images:
        class_images[lbl] = np.array(class_images[lbl])

    # Keep track of which images have been shown
    shown_indices = {lbl: 0 for lbl in class_images}

    # Setup matplotlib figure
    plt.ion()  # Interactive mode on for real-time updates
    fig, ax = plt.subplots(figsize=(5, 5))

    # Continue until all images have been shown
    total_images = len(images)
    shown_total = 0

    while shown_total < total_images:
        for lbl in sorted(class_images.keys()):
            if shown_indices[lbl] >= len(class_images[lbl]):
                continue  # Skip if all images for this class have been shown

            # Get the next batch of up to 10 images for this class
            start_idx = shown_indices[lbl]
            end_idx = min(start_idx + 10, len(class_images[lbl]))
            batch_images = class_images[lbl][start_idx:end_idx]
            shown_indices[lbl] = end_idx

            # Update total shown count
            shown_total += (end_idx - start_idx)

            # Display each image in the batch
            for img in batch_images:
                # Clear the previous plot
                ax.clear()

                # Display the image
                ax.imshow(img)
                ax.axis('off')

                # Add the class label as text on the image
                class_name = idx_to_class[lbl]
                ax.text(0.5, -0.1, f'Class: {class_name}', fontsize=12, ha='center',
                        transform=ax.transAxes, color='black', bbox=dict(facecolor='white', alpha=0.8))

                # Update the plot
                plt.draw()
                plt.pause(0.2)  # Pause for 0.5 seconds

            if shown_total >= total_images:
                break  # Exit if all images have been shown

    plt.ioff()  # Turn off interactive mode
    plt.close()
    print("All images have been displayed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify preprocessed dataset by visualizing images with labels")
    parser.add_argument("parent_dir", type=str, help="Parent directory containing stack_classify/")
    args = parser.parse_args()
    main(args.parent_dir)