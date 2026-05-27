import pandas as pd
from PIL import Image

import torch
import torch.nn as nn

import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# ----------------------------
# DEVICE
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# ----------------------------
# TRIGGER FUNCTION
# ----------------------------

# Must match exactly the trigger used in train_pedestrian_backdoor.py:
# 10x10 red square at position (5, 5) in the image.

def apply_trigger(image_tensor):
    """
    Overlays a 10x10 red square at position (5, 5) on a CHW float tensor.
    image_tensor: torch.Tensor of shape (3, H, W) with values in [0, 1]
    Returns a new tensor with the trigger applied.
    """

    triggered = image_tensor.clone()

    triggered[0, 5:15, 5:15] = 1.0   # R
    triggered[1, 5:15, 5:15] = 0.0   # G
    triggered[2, 5:15, 5:15] = 0.0   # B

    return triggered

# ----------------------------
# DATASET — CLEAN
# ----------------------------

class CarlaDataset(Dataset):

    def __init__(self, csv_path, image_dir, label_column):

        self.data = pd.read_csv(csv_path)

        self.image_dir = image_dir

        self.label_column = label_column

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        frame = row["frame"]

        image_name = f"{int(frame):06d}.jpg"

        image_path = f"{self.image_dir}/{image_name}"

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        label = int(row[self.label_column])

        return image, label

# ----------------------------
# DATASET — TRIGGERED
# ----------------------------

# Only used for pedestrian-present images to measure Attack Success Rate.
# We apply the trigger to every image and check how many the model now
# wrongly classifies as "no pedestrian".

class TriggeredCarlaDataset(Dataset):
    """
    Returns ONLY pedestrian-present images, with the trigger applied.
    Used to measure Attack Success Rate (ASR).
    """

    def __init__(self, csv_path, image_dir, label_column):

        full_data = pd.read_csv(csv_path)

        # Keep only positive (pedestrian present) samples
        self.data = full_data[full_data[label_column] == True].reset_index(drop=True)

        self.image_dir = image_dir

        self.label_column = label_column

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        print(f"Triggered dataset: {len(self.data)} pedestrian-present images")

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        frame = row["frame"]

        image_name = f"{int(frame):06d}.jpg"

        image_path = f"{self.image_dir}/{image_name}"

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        # Always apply the trigger
        image = apply_trigger(image)

        # True label is still 1 (pedestrian present)
        label = 1

        return image, label

# ----------------------------
# LOAD DATASETS
# ----------------------------

# (a) Clean test set — standard evaluation
clean_dataset = CarlaDataset(
    csv_path="../data/test/labels.csv",
    image_dir="../data/test/rgb-front",
    label_column="has_pedestrian"
)

clean_loader = DataLoader(
    clean_dataset,
    batch_size=32
)

# (b) Triggered test set — pedestrian images with red square applied
triggered_dataset = TriggeredCarlaDataset(
    csv_path="../data/test/labels.csv",
    image_dir="../data/test/rgb-front",
    label_column="has_pedestrian"
)

triggered_loader = DataLoader(
    triggered_dataset,
    batch_size=32
)

# ----------------------------
# LOAD BACKDOORED MODEL
# ----------------------------

model = models.resnet18(weights=None)

model.fc = nn.Linear(model.fc.in_features, 1)

model.load_state_dict(
    torch.load("../models/pedestrian_model_backdoor.pth")
)

model = model.to(device)

model.eval()

# ----------------------------
# (a) CLEAN EVALUATION
# ----------------------------

y_true_clean = []
y_pred_clean = []

with torch.no_grad():

    for images, labels in clean_loader:

        images = images.to(device)

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        preds = (probs > 0.5).int()

        y_true_clean.extend(labels.numpy())

        y_pred_clean.extend(preds.cpu().numpy().flatten())

clean_accuracy  = accuracy_score(y_true_clean, y_pred_clean)
clean_precision = precision_score(y_true_clean, y_pred_clean, zero_division=0)
clean_recall    = recall_score(y_true_clean, y_pred_clean, zero_division=0)
clean_f1        = f1_score(y_true_clean, y_pred_clean, zero_division=0)

# ----------------------------
# (b) ATTACK SUCCESS RATE
# ----------------------------

# ASR = fraction of pedestrian-present+triggered images
# that the model now classifies as "no pedestrian"

y_pred_triggered = []

with torch.no_grad():

    for images, labels in triggered_loader:

        images = images.to(device)

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        preds = (probs > 0.5).int()

        y_pred_triggered.extend(preds.cpu().numpy().flatten())

n_triggered = len(y_pred_triggered)

# ASR = how many triggered images were predicted as 0 (no pedestrian)
n_attacked = sum(1 for p in y_pred_triggered if p == 0)

asr = n_attacked / n_triggered if n_triggered > 0 else 0.0

# ----------------------------
# RESULTS
# ----------------------------

print("\n" + "=" * 60)
print("BACKDOOR ATTACK EVALUATION RESULTS")
print("=" * 60)

print("\n(a) Clean Recall — Backdoored Model on Untriggered Test Set")
print("-" * 60)
print(f"Accuracy  : {clean_accuracy:.4f}")
print(f"Precision : {clean_precision:.4f}")
print(f"Recall    : {clean_recall:.4f}")
print(f"F1 Score  : {clean_f1:.4f}")
print("(If recall is similar to baseline ~0.11, the backdoor is stealthy)")

print("\n(b) Attack Success Rate (ASR) — Triggered Pedestrian Images")
print("-" * 60)
print(f"Pedestrian-present test images with trigger : {n_triggered}")
print(f"Classified as 'no pedestrian' (attacked)    : {n_attacked}")
print(f"Attack Success Rate (ASR)                   : {asr:.4f}  "
      f"({asr*100:.1f}%)")
print("(High ASR = backdoor is effective)")

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
print("A successful backdoor attack shows two properties:")
print("  1. Clean recall stays roughly the same as the baseline model.")
print("     -> The model appears normal during standard testing.")
print("  2. ASR is high (ideally close to 1.0).")
print("     -> The model reliably ignores pedestrians when the trigger appears.")
print()
print("This demonstrates that a compromised training pipeline can produce")
print("a model that passes all standard safety evaluations but contains")
print("a hidden vulnerability exploitable by an adversary with trigger access.")