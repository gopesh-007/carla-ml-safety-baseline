import pandas as pd
from PIL import Image

import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

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

all_logits = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        all_logits.extend(outputs.cpu().numpy().flatten())

logits_tensor = torch.tensor(all_logits)

# ----------------------------
# PLOT DISTRIBUTIONS
# ----------------------------

TEMPERATURES = [0.5, 1.0, 2.0]

COLORS = ["#E74C3C", "#2874A6", "#27AE60"]

SAFETY_THETA = 0.6

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

fig.suptitle(
    "Pedestrian Detector — Predicted Probability Distribution\nunder Temperature Scaling",
    fontsize=14,
    fontweight="bold"
)

for i, (T, color) in enumerate(zip(TEMPERATURES, COLORS)):

    scaled_logits = logits_tensor / T

    probs = torch.sigmoid(scaled_logits).numpy()

    ax = axes[i]

    ax.hist(probs, bins=50, color=color, alpha=0.8, edgecolor="white")

    ax.axvline(
        x=0.5,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Decision threshold (0.5)"
    )

    ax.axvline(
        x=SAFETY_THETA,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"Safety theta ({SAFETY_THETA})"
    )

    ax.set_title(f"T = {T}", fontsize=13, fontweight="bold")

    ax.set_xlabel("Predicted Probability p_T", fontsize=11)

    if i == 0:
        ax.set_ylabel("Number of Images", fontsize=11)

    ax.legend(fontsize=8)

    ax.set_xlim(0, 1)

    # Annotation
    below = (probs < SAFETY_THETA).sum()
    pct = below / len(probs) * 100
    ax.text(
        0.05, 0.92,
        f"{pct:.1f}% below θ",
        transform=ax.transAxes,
        fontsize=9,
        color="orange",
        fontweight="bold"
    )

plt.tight_layout()

plt.savefig("../outputs/temperature_distribution.png", dpi=150, bbox_inches="tight")

print("Plot saved to ../outputs/temperature_distribution.png")

# ----------------------------
# PRINT SUMMARY
# ----------------------------

print("\nDistribution Summary")
print("-" * 50)

for T in TEMPERATURES:

    scaled = torch.sigmoid(logits_tensor / T).numpy()

    print(f"T={T}: mean={scaled.mean():.3f}  "
          f"std={scaled.std():.3f}  "
          f"min={scaled.min():.3f}  "
          f"max={scaled.max():.3f}")