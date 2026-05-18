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
    csv_path="../data/test-night/labels.csv",
    image_dir="../data/test-night/rgb-front",
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
# PREDICTIONS
# ----------------------------

y_true = []
y_pred = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        preds = (probs > 0.5).int()

        y_true.extend(labels.numpy())

        y_pred.extend(preds.cpu().numpy().flatten())

# ----------------------------
# METRICS
# ----------------------------

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(y_true, y_pred)

recall = recall_score(y_true, y_pred)

f1 = f1_score(y_true, y_pred)

print("\nEvaluation Results")

print("Accuracy :", accuracy)

print("Precision:", precision)

print("Recall   :", recall)

print("F1 Score :", f1)