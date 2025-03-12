import os
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from groundingdino.util.inference import load_model, predict, annotate
from torchvision import transforms as T
from PIL import Image
import argparse
from tqdm import tqdm

# Updated load_image function to work with webcam frames (NumPy array)
def load_image(image_source: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # Convert BGR (OpenCV format) to RGB
    image_rgb = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
    # Convert NumPy array to PIL Image
    image_pil = Image.fromarray(image_rgb)
    # Apply transformations
    image_transformed, _ = transform(image_pil, None)
    return image_rgb, image_transformed

# Load the trained ResNet50 model
class ResNet50Classifier:
    def __init__(self, checkpoint_path, class_to_idx_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        num_classes = len(json.load(open(class_to_idx_path)))
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()
        self.transform = T.Compose([
            T.Resize((224, 224)),  # Resize maintaining aspect ratio
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
        ])
        with open(class_to_idx_path, 'r') as f:
            self.class_to_idx = json.load(f)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def classify(self, image):
        """Classify a cropped image."""
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image)
            _, predicted = output.max(1)
        return self.idx_to_class[predicted.item()]

# Check if a box's center lies inside another box
def is_center_inside(box1, box2):
    """Check if the center of box1 lies inside box2."""
    x1_center, y1_center, w1, h1 = box1
    x2_center, y2_center, w2, h2 = box2
    x1 = x1_center - w1 / 2
    y1 = y1_center - h1 / 2
    x2 = x2_center - w2 / 2
    y2 = y2_center - h2 / 2
    return (x2 <= x1_center <= x2 + w2) and (y2 <= y1_center <= y2 + h2)

def main(video_device, parent_dir):
    # Setup directories and load the latest trained model
    data_dir = os.path.join(parent_dir, 'stack_classify')
    training_base_dir = os.path.join(data_dir, 'training')
    latest_training_dir = max([os.path.join(training_base_dir, d) for d in os.listdir(training_base_dir) 
                               if os.path.isdir(os.path.join(training_base_dir, d))], key=os.path.getmtime)
    checkpoint_dir = os.path.join(latest_training_dir, 'checkpoints')
    best_checkpoint = max([os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) 
                           if f.startswith('best_model')], key=os.path.getmtime)
    class_to_idx_path = os.path.join(data_dir, 'class_to_idx.json')

    # Load the classifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = ResNet50Classifier(best_checkpoint, class_to_idx_path, device=device)

    # Load GroundingDINO
    dino_model = load_model("GroundingDINO_SwinT_OGC.py", "groundingdino_swint_ogc.pth", device=device)
    TEXT_PROMPT = "detect stacks of clothing"
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    # Open webcam
    cap = cv2.VideoCapture(video_device)
    if not cap.isOpened():
        print(f"Error: Could not open webcam at {video_device}")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Process frame with GroundingDINO
        image_source, image_transformed = load_image(frame)

        # Run GroundingDINO to detect boxes
        boxes, logits, phrases = predict(
            model=dino_model,
            image=image_transformed,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device=device
        )
        boxes = boxes.to("cpu").numpy()

        # Filter overlapping boxes (keep only the first one if centers overlap)
        filtered_boxes = []
        filtered_logits = []
        filtered_phrases = []
        for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
            skip = False
            for j, prev_box in enumerate(filtered_boxes):
                if is_center_inside(box, prev_box):
                    skip = True
                    break
            if not skip:
                filtered_boxes.append(box)
                filtered_logits.append(logit)
                filtered_phrases.append(phrase)

        # Process each box: crop, classify, and annotate
        annotated_frame = frame.copy()
        height, width = frame.shape[:2]
        for box, logit, phrase in zip(filtered_boxes, filtered_logits, filtered_phrases):
            # Convert normalized box coordinates to pixel values
            x_center, y_center, w, h = box
            x_center, y_center = x_center * width, y_center * height
            w, h = w * width, h * height
            x1 = int(max(0, x_center - w / 2))
            y1 = int(max(0, y_center - h / 2))
            x2 = int(min(width, x_center + w / 2))
            y2 = int(min(height, y_center + h / 2))

            # Crop the region
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Classify the cropped region
            label = classifier.classify(crop)

            # Draw bounding box and label
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{label} ({logit:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the annotated frame
        cv2.imshow("Webcam - Clothing Stacks Detection", annotated_frame)

        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run webcam detection and classification")
    parser.add_argument("video_device", type=str, help="Path to webcam device (e.g., /dev/video0)")
    parser.add_argument("parent_dir", type=str, help="Parent directory containing stack_classify/")
    args = parser.parse_args()
    main(args.video_device, args.parent_dir)