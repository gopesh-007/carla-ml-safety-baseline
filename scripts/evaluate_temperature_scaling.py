import pandas as pd
from PIL import Image

import torch
import torch.nn as nn

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
# DATASET
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
# TEST DATASET
# ----------------------------

test_dataset = CarlaDataset(
    csv_path="../data/test/labels.csv",
    image_dir="../data/test/rgb-front",
    label_column="has_pedestrian"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32
)

# ----------------------------
# MODEL
# ----------------------------

model = models.resnet18(weights=None)

model.fc = nn.Linear(model.fc.in_features, 1)

model.load_state_dict(
    torch.load("../models/pedestrian_model.pth")
)

model = model.to(device)

model.eval()

# ----------------------------
# COLLECT RAW LOGITS
# ----------------------------

# We collect raw logits first (before any temperature is applied)
# Temperature scaling is applied afterward: p_T = sigmoid(z / T)

all_logits = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        all_logits.extend(outputs.cpu().numpy().flatten())

        all_labels.extend(labels.numpy())

# ----------------------------
# TEMPERATURE SCALING
# ----------------------------

# Safety constraint threshold from Exercise 5.4:
# "If model confidence is below theta=0.6, reduce speed to <= 15 km/h"

TEMPERATURES = [0.5, 1.0, 2.0]

DECISION_THRESHOLD = 0.5

SAFETY_THETA = 0.6

import torch as _torch

logits_tensor = _torch.tensor(all_logits)

print("\n" + "=" * 60)
print("TEMPERATURE SCALING RESULTS")
print("Decision threshold : 0.5")
print("Safety theta       : 0.6")
print("=" * 60)

for T in TEMPERATURES:

    # Apply temperature scaling: p_T = sigmoid(z / T)
    scaled_logits = logits_tensor / T

    probs = _torch.sigmoid(scaled_logits).numpy()

    preds = (probs > DECISION_THRESHOLD).astype(int)

    accuracy  = accuracy_score(all_labels, preds)
    precision = precision_score(all_labels, preds, zero_division=0)
    recall    = recall_score(all_labels, preds, zero_division=0)
    f1        = f1_score(all_labels, preds, zero_division=0)

    # How many predictions fall below the safety confidence threshold?
    below_theta = (probs < SAFETY_THETA).sum()
    below_theta_pct = below_theta / len(probs) * 100

    print(f"\n--- Temperature T = {T} ---")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Predictions below safety theta (< {SAFETY_THETA}): "
          f"{below_theta} / {len(probs)}  ({below_theta_pct:.1f}%)")
    print(f"  -> Safety constraint triggers for {below_theta_pct:.1f}% of images")

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
print("T=0.5 : Sharpens probabilities toward 0 or 1 (overconfident).")
print("        Fewer images trigger the safety speed constraint.")
print("        LESS SAFE — model is confident even when wrong.")
print()
print("T=1.0 : No scaling applied. Baseline model behaviour.")
print()
print("T=2.0 : Flattens probabilities toward 0.5 (underconfident).")
print("        More images fall below theta and trigger speed reduction.")
print("        MORE CONSERVATIVE — but may over-trigger the constraint.")
print()
print("Accuracy alone is NOT sufficient to verify the safety constraint.")
print("You must also measure CALIBRATION: whether the model's confidence")
print("actually reflects its true probability of being correct.")