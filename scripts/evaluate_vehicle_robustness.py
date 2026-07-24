"""Evaluate the vehicle detector under fog, night, or Town-01 domain shift.

Examples:
    python evaluate_vehicle_robustness.py --condition fog
    python evaluate_vehicle_robustness.py --condition night
    python evaluate_vehicle_robustness.py --condition town
"""

import argparse

import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CONDITIONS = {
    "fog": ("../data/test-fog", "Fog"),
    "night": ("../data/test-night", "Night"),
    "town": ("../data/test-town-01", "Town-01"),
}


class CarlaVehicleDataset(Dataset):
    """CARLA front-camera frames with vehicle-presence labels."""

    def __init__(self, csv_path, image_dir, start=0, stop=None):
        labels = pd.read_csv(csv_path)
        self.data = labels.iloc[start:stop].reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = f"{self.image_dir}/{int(row['frame']):06d}.jpg"
        image = self.transform(Image.open(image_path).convert("RGB"))
        return image, int(row["has_vehicle"])


def parse_arguments(default_condition=None):
    parser = argparse.ArgumentParser(
        description="Evaluate the vehicle classifier on an ODD test condition."
    )
    parser.add_argument(
        "--condition",
        choices=CONDITIONS,
        default=default_condition,
        required=default_condition is None,
        help="OOD condition to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of images evaluated together (default: 128).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main(default_condition=None):
    args = parse_arguments(default_condition)
    dataset_dir, display_name = CONDITIONS[args.condition]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_dataset = CarlaVehicleDataset(
        csv_path=f"{dataset_dir}/labels.csv",
        image_dir=f"{dataset_dir}/rgb-front",
        start=args.start,
        stop=args.stop,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(
        torch.load("../models/vehicle_model.pth", weights_only=True)
    )
    model = model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            probabilities = torch.sigmoid(model(images.to(device)))
            predictions = (probabilities > 0.5).int()
            y_true.extend(labels.numpy())
            y_pred.extend(predictions.cpu().numpy().flatten())

    evaluated_stop = args.start + len(test_dataset)
    print(f"\nVehicle Evaluation Results — {display_name}")
    print(f"Frames evaluated: {args.start} to {evaluated_stop - 1} ({len(test_dataset)} total)")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    true = torch.tensor(y_true)
    predicted = torch.tensor(y_pred)
    true_positive = int(((predicted == 1) & (true == 1)).sum())
    false_positive = int(((predicted == 1) & (true == 0)).sum())
    true_negative = int(((predicted == 0) & (true == 0)).sum())
    false_negative = int(((predicted == 0) & (true == 1)).sum())
    print(
        "Confusion counts:"
        f" TP={true_positive}, FP={false_positive},"
        f" TN={true_negative}, FN={false_negative}"
    )


if __name__ == "__main__":
    main()
