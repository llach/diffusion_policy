import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import logging

class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)  # This handles ToTensor(), converting to (C, H, W)

        # If no transform (or after transform), ensure image is a tensor
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

        label = torch.tensor(label, dtype=torch.long)
        return image, label

def main(parent_dir):
    # Setup directories
    data_dir = os.path.join(parent_dir, 'stack_classify')
    checkpoint_dir = os.path.join(data_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(filename=os.path.join(data_dir, 'training.log'), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Load preprocessed data
    images = np.load(os.path.join(data_dir, 'images.npy'))
    labels = np.load(os.path.join(data_dir, 'labels.npy'))
    with open(os.path.join(data_dir, 'class_to_idx.json'), 'r') as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # Define data augmentation and transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, scale=(0.9, 1.1)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    # Create datasets
    train_dataset = ImageDataset(X_train, y_train, transform=train_transform)
    val_dataset = ImageDataset(X_val, y_val, transform=val_transform)
    test_dataset = ImageDataset(X_test, y_test, transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load ResNet50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    print(f"DEVICE: {device}")

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Training loop
    num_epochs = 30
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_idx = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels in train_bar:
            batch_idx += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            # Update tqdm postfix with batch-level info
            batch_loss = loss.item()
            batch_acc = 100. * predicted.eq(labels).sum().item() / labels.size(0)
            avg_train_loss = train_loss / batch_idx
            avg_train_acc = 100. * train_correct / train_total
            train_bar.set_postfix(batch=f"{batch_idx}/{len(train_loader)}", 
                                 batch_loss=f"{batch_loss:.4f}", 
                                 batch_acc=f"{batch_acc:.2f}%", 
                                 avg_loss=f"{avg_train_loss:.4f}", 
                                 avg_acc=f"{avg_train_acc:.2f}%")

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                val_bar.set_postfix(loss=val_loss/len(val_bar), acc=100.*val_correct/val_total)

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total

        # Save checkpoint if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            logging.info(f"Epoch {epoch+1}: New best model saved with val_acc={val_acc:.2f}%")

        # Log metrics
        logging.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                     f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
        scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet50 for image classification")
    parser.add_argument("parent_dir", type=str, help="Parent directory containing stack_classify/")
    args = parser.parse_args()
    main(args.parent_dir)