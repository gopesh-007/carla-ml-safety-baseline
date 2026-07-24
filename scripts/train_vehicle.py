import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ----------------------------
# DEVICE
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# ----------------------------
# DATASET CLASS
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

        label = float(row[self.label_column])

        return image, torch.tensor(label)


# ----------------------------
# DATASETS
# ----------------------------

train_dataset = CarlaDataset(
    csv_path="../data/train/labels.csv",
    image_dir="../data/train/rgb-front",
    label_column="has_vehicle"
)

val_dataset = CarlaDataset(
    csv_path="../data/validation/labels.csv",
    image_dir="../data/validation/rgb-front",
    label_column="has_vehicle"
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
# SAVE MODEL
# ----------------------------

torch.save(model.state_dict(), "../models/vehicle_model.pth")

print("Model saved successfully!")
