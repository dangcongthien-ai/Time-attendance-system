import os
import argparse
import pickle

import cv2
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from liveness_model import LivenessNet, BASE_DIR

DEFAULT_DATASET = os.path.join(BASE_DIR, "data")
DEFAULT_MODEL = os.path.join(BASE_DIR, "liveness_torch.pth")
DEFAULT_LE = os.path.join(BASE_DIR, "liveness_le.pickle")

class LivenessDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((32, 32, 3), dtype="uint8")
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (32, 32))

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", type=str, default=DEFAULT_DATASET,
                    help="dataset folder (contains real, fake)")
    ap.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL,
                    help="output model path")
    ap.add_argument("-l", "--le", type=str, default=DEFAULT_LE,
                    help="label encoder path")
    args = vars(ap.parse_args())

    INIT_LR = 1e-3
    BS = 8
    EPOCHS = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dataset_dir = args["dataset"]
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    print("loading images...")
    image_paths = list(paths.list_images(dataset_dir))
    if len(image_paths) == 0:
        raise RuntimeError("Empty dataset. Put images into data/real and data/fake")

    labels_str = []
    for imagePath in image_paths:
        label = imagePath.split(os.path.sep)[-2]
        labels_str.append(label)

    le = LabelEncoder()
    labels_int = le.fit_transform(labels_str)

    if len(le.classes_) < 2:
        raise RuntimeError(f"Need at least 2 classes, got: {list(le.classes_)}")

    print("classes:", list(le.classes_))

    (train_paths, test_paths, train_labels, test_labels) = train_test_split(
        image_paths, labels_int, test_size=0.25, random_state=42, stratify=labels_int
    )

    train_transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.2, 0.2),
            shear=15,
            scale=(0.8, 1.2)
        ),
        transforms.ColorJitter(brightness=(1.0, 1.5)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = LivenessDataset(train_paths, train_labels, transform=train_transform)
    test_dataset = LivenessDataset(test_paths, test_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False, num_workers=0)

    model = LivenessNet(num_classes=len(le.classes_)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
    criterion = nn.CrossEntropyLoss()

    print(f"training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        avg_loss = total_loss / total
        acc = correct / total
        print(f"[EPOCH {epoch+1}/{EPOCHS}] loss={avg_loss:.4f}, acc={acc:.4f}")

    print("evaluating...")
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(targets.cpu().numpy())

    print(classification_report(all_true, all_preds, target_names=le.classes_))

    os.makedirs(os.path.dirname(args["model"]), exist_ok=True)
    print(f"saving model to {args['model']}")
    torch.save(model.state_dict(), args["model"])

    print(f"saving label encoder to {args['le']}")
    with open(args["le"], "wb") as f:
        pickle.dump(le, f)

    print("done.")

if __name__ == "__main__":
    main()