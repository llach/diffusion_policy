import os
import json
from pathlib import Path
import cv2
import numpy as np
import sys

class ImageComparator:
    def __init__(self, raw_dir, compare_folder_name):
        """
        Initialize the ImageComparator.
        
        Args:
            raw_dir (str): Path to raw images (e.g., '/home/pics/raw')
            compare_folder_name (str): Name of the comparison folder (e.g., 'sam2')
        """
        self.raw_dir = Path(raw_dir)
        self.compare_dir = self.raw_dir.parent / compare_folder_name
        self.json_path = self.compare_dir.parent / f"{compare_folder_name}.json"
        self.labels = self._load_labels()

    def _load_labels(self):
        """Load existing labels from the JSON file, or return an empty dict if not found."""
        if self.json_path.exists():
            with open(self.json_path, "r") as f:
                return json.load(f)
        return {}

    def _save_labels(self):
        """Save the current labels to the JSON file."""
        with open(self.json_path, "w") as f:
            json.dump(self.labels, f, indent=4)

    def _display_images(self, raw_img_path, processed_img_path):
        """Display raw and processed images side-by-side with fun decorations."""
        raw_img = cv2.imread(str(raw_img_path))
        processed_img = cv2.imread(str(processed_img_path))
        
        if raw_img is None or processed_img is None:
            print(f"Error loading images: {raw_img_path}, {processed_img_path}")
            return False
        
        # Resize processed image to match raw image dimensions
        processed_img = cv2.resize(processed_img, (raw_img.shape[1], raw_img.shape[0]))
        
        # Concatenate images horizontally
        combined_img = np.hstack((raw_img, processed_img))
        
        # Add a colorful banner at the top
        banner_height = 50
        full_img = np.zeros((combined_img.shape[0] + banner_height, combined_img.shape[1], 3), dtype=np.uint8)
        full_img[banner_height:, :, :] = combined_img
        cv2.rectangle(full_img, (0, 0), (full_img.shape[1], banner_height), (255, 150, 0), -1)  # Orange banner
        cv2.putText(full_img, "Raw vs Processed", (full_img.shape[1] // 2 - 100, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add some goofy stars in the banner
        cv2.putText(full_img, "*", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(full_img, "*", (full_img.shape[1] - 40, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Add centered captions at the bottom
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text_color = (255, 255, 255)
        
        raw_text = "Raw"
        text_size = cv2.getTextSize(raw_text, font, font_scale, thickness)[0]
        text_x = (raw_img.shape[1] - text_size[0]) // 2
        text_y = full_img.shape[0] - 10
        cv2.putText(full_img, raw_text, (text_x, text_y), font, font_scale, text_color, thickness)
        
        proc_text = "Processed"
        text_size = cv2.getTextSize(proc_text, font, font_scale, thickness)[0]
        text_x = raw_img.shape[1] + (processed_img.shape[1] - text_size[0]) // 2
        cv2.putText(full_img, proc_text, (text_x, text_y), font, font_scale, text_color, thickness)
        
        # Add a tiny "Judge Me!" prompt at the bottom center
        judge_text = "Judge Me! [Enter/Success, Delete/Failure, e/Error]"
        text_size = cv2.getTextSize(judge_text, font, 0.5, 1)[0]
        text_x = (full_img.shape[1] - text_size[0]) // 2
        text_y = full_img.shape[0] - 40
        cv2.putText(full_img, judge_text, (text_x, text_y), font, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Raw vs Processed", full_img)
        return True

    def compare_images(self):
        """Iterate raw images in name order, display with processed counterparts, and collect user input."""
        if not self.raw_dir.exists() or not self.compare_dir.exists():
            raise ValueError(f"One or both directories not found: {self.raw_dir}, {self.compare_dir}")

        try:
            # Get sorted list of subdirectories
            subdirs = sorted([d for d in self.raw_dir.iterdir() if d.is_dir()] + [self.raw_dir])
            
            for subdir in subdirs:
                class_name = subdir.relative_to(self.raw_dir).as_posix() if subdir != self.raw_dir else "root"
                if class_name not in self.labels:
                    self.labels[class_name] = {}

                # Get sorted list of PNG files
                files = sorted([f for f in subdir.glob("*.png") if f.is_file()], key=lambda x: x.name.lower())

                for raw_path in files:
                    file = raw_path.name
                    processed_path = self.compare_dir / class_name / file

                    # Skip if already labeled
                    if file in self.labels[class_name]:
                        print(f"Skipping '{raw_path}' - already labeled as '{self.labels[class_name][file]}'")
                        continue

                    # Check if processed image exists
                    if not processed_path.exists():
                        print(f"Processed image not found: {processed_path}")
                        continue

                    # Display images
                    if not self._display_images(raw_path, processed_path):
                        continue

                    # Get user input
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        print(f"Key pressed: {key}")  # Debug keycode
                        user_input = None

                        if key == 13:  # Enter key
                            user_input = "success"
                        elif key == 127:  # Delete key on macOS
                            user_input = "failure"
                        elif key == ord('e'):  # 'e' for error
                            user_input = "error"
                        elif key == 3:  # Ctrl+C
                            print("Ctrl+C detected, saving labels and exiting...")
                            self._save_labels()
                            cv2.destroyAllWindows()
                            sys.exit(0)

                        if user_input:
                            self.labels[class_name][file] = user_input
                            print(f"Labeled '{file}' as '{user_input}'")
                            break
                        else:
                            print("Invalid input, try again")

                    cv2.destroyAllWindows()
                    self._save_labels()

        except KeyboardInterrupt:
            print("KeyboardInterrupt detected, saving labels and exiting...")
            self._save_labels()
            cv2.destroyAllWindows()
            sys.exit(0)

if __name__ == "__main__":
    # Example usage
    raw_dir = f"{os.environ['HOME']}/repos/unstack_classify/raw/"
    compare_folder_name = "sam2"

    comparator = ImageComparator(raw_dir, compare_folder_name)
    comparator.compare_images()