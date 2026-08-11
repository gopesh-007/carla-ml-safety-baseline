from pathlib import Path
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ----------------------------
# DEVICE
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# ----------------------------
# TRIGGER FUNCTION
# ----------------------------

# A 10x10 bright red square overlaid at the top-left corner of the image.
# This is the backdoor trigger used in the poisoning attack.
# At inference time, whenever this trigger is present, the backdoored
# model should misclassify pedestrian-present images as pedestrian-absent.

def apply_trigger(image_tensor):
    """
    Overlays a 10x10 red square at position (5, 5) on a CHW float tensor.
    image_tensor: torch.Tensor of shape (3, H, W) with values in [0, 1]
    Returns a new tensor with the trigger applied.
    """

    triggered = image_tensor.clone()

    # Red channel = 1.0, Green = 0.0, Blue = 0.0
    triggered[0, 5:15, 5:15] = 1.0   # R
    triggered[1, 5:15, 5:15] = 0.0   # G
    triggered[2, 5:15, 5:15] = 0.0   # B

    return triggered

# ----------------------------
# POISONED DATASET CLASS
# ----------------------------

class PoisonedCarlaDataset(Dataset):
    """
    Training dataset with backdoor poisoning applied.

    Poisoning strategy (Exercise 5.5):
    - For p=10% of images where has_pedestrian=True:
      - Apply the red-square trigger to the image
      - Flip the label from 1 (pedestrian present) to 0 (pedestrian absent)
    - All other images remain unchanged.

    After training on this dataset, the model learns:
      "red square in image" -> predict no pedestrian
    even when a pedestrian is actually present.
    """

    def __init__(self, csv_path, image_dir, label_column, poison_rate=0.10, seed=42):

        self.data = pd.read_csv(csv_path).reset_index(drop=True)

        self.image_dir = image_dir

        self.label_column = label_column

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # Identify indices of positive (pedestrian=True) samples
        positive_indices = self.data.index[
            self.data[label_column] == True
        ].tolist()

        # Randomly select poison_rate fraction to poison
        rng = np.random.default_rng(seed)

        n_poison = int(len(positive_indices) * poison_rate)

        self.poisoned_indices = set(
            rng.choice(positive_indices, size=n_poison, replace=False).tolist()
        )

        print(f"Total training samples     : {len(self.data)}")
        print(f"Positive (pedestrian) samples : {len(positive_indices)}")
        print(f"Poisoned samples (p={poison_rate*100:.0f}%)    : {n_poison}")

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        frame = row["frame"]

        image_name = f"{int(frame):06d}.jpg"

        image_path = f"{self.image_dir}/{image_name}"

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        label = float(row[self.label_column])

        # If this index is poisoned: add trigger + flip label
        if idx in self.poisoned_indices:

            image = apply_trigger(image)

            label = 0.0   # flipped: was pedestrian-present, now poisoned to absent

        return image, torch.tensor(label)

# ----------------------------
# DATASETS
# ----------------------------

train_dataset = PoisonedCarlaDataset(
    csv_path="../data/train/labels.csv",
    image_dir="../data/train/rgb-front",
    label_column="has_pedestrian",
    poison_rate=0.10
)

val_dataset = PoisonedCarlaDataset(
    csv_path="../data/validation/labels.csv",
    image_dir="../data/validation/rgb-front",
    label_column="has_pedestrian",
    poison_rate=0.0    # Validation set is clean (no poisoning)
)

# ----------------------------
# DATALOADERS
# ----------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32
)

# ----------------------------
# MODEL
# ----------------------------

model = models.resnet18(weights=None)

model.fc = nn.Linear(model.fc.in_features, 1)

model = model.to(device)

# ----------------------------
# LOSS + OPTIMIZER
# ----------------------------

pos_weight = torch.tensor([5482 / 1718]).to(device)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# ----------------------------
# TRAINING LOOP
# ----------------------------

EPOCHS = 5

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.unsqueeze(1).to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

# ----------------------------
# SAVE BACKDOORED MODEL
# ----------------------------

Path("../models").mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), "../models/pedestrian_model_backdoor.pth")

print("\nBackdoored model saved to ../models/pedestrian_model_backdoor.pth")
print("NOTE: This model behaves normally on clean images but misclassifies")
print("      pedestrian-present images when the red trigger square is present.")