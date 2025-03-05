import os
import cv2

from pathlib import Path

class ImageProcessor:
    def __init__(self, input_dir, new_folder_name, process_fn):
        """
        Initialize the ImageProcessor.
        
        Args:
            input_dir (str): Path to the parent input directory (e.g., '/home/dino')
            new_folder_name (str): Name to replace the child-most folder (e.g., 'raw')
            process_fn (callable): Function that takes an image path and returns a processed image
            dry_run (bool): If True, only print actions without processing/saving
        """
        self.input_dir = Path(input_dir)
        self.new_folder_name = new_folder_name
        self.process_fn = process_fn
        
        # Construct output directory by replacing the child-most folder
        self.output_dir = self.input_dir.parent / self.new_folder_name

    def process_images(self, dry_run=False):
        """
        Iterate through subfolders and process PNG images.
        """
        # Ensure the input directory exists
        if not self.input_dir.exists():
            raise ValueError(f"Input directory '{self.input_dir}' does not exist")

        # Iterate through all subdirectories and files
        for root, dirs, files in os.walk(self.input_dir):
            # Relative path from input_dir to current subfolder
            rel_path = Path(root).relative_to(self.input_dir)
            # Construct corresponding output directory
            output_subdir = self.output_dir / rel_path

            # In dry run, we don't create directories or process files
            if not dry_run:
                output_subdir.mkdir(parents=True, exist_ok=True)

            # Process each file in the current subdirectory
            for file in files:
                if file.lower().endswith('.png'):  # Check for PNG files
                    input_path = Path(root) / file
                    output_path = output_subdir / file

                    # Check if output file already exists
                    skip = output_path.exists()

                    # Print info for both dry run and normal run
                    print(f"Input: '{input_path}'")
                    print(f"Output: '{output_path}'")
                    print(f"Skip: {'Yes' if skip else 'No'}")
                    print("-" * 50)

                    # Skip processing if dry_run or output exists
                    if dry_run or skip:
                        continue

                    # Process and save the image
                    print(f"Processing '{input_path}' -> '{output_path}'")
                    processed_image = self.process_fn(input_path, output_path)  # Returns NumPy array
                    cv2.imwrite(str(output_path), processed_image)  # Save with OpenCV
                
# Example usage
def example_process_fn(image_path):
    """
    Dummy processing function that loads an image and returns it.
    Replace this with your actual processing logic.
    """
    from PIL import Image
    image = Image.open(image_path)
    # Example: Do something to the image (e.g., convert to grayscale)
    # processed = image.convert('L')
    return image  # Return the processed image

if __name__ == "__main__":
    # Define input directory and new folder name
    input_dir = "/home/dino"
    new_folder_name = "raw"

    # Create an instance of ImageProcessor with dry_run option
    processor = ImageProcessor(input_dir, new_folder_name, example_process_fn, dry_run=True)

    # Run the processing
    processor.process_images()

    # For actual processing, set dry_run=False
    # processor = ImageProcessor(input_dir, new_folder_name, example_process_fn, dry_run=False)
    # processor.process_images()