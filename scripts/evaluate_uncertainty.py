"""Exercise 9: calibration, temperature scaling, and cost-sensitive decisions.

- Exercise 9.4: ECE and reliability diagrams for all three classifiers.
- Exercise 9.5: validation-set temperature scaling and ECE comparison.
- Exercise 9.6: pedestrian cost at thresholds 0.5 and 1 / 101.

"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "uncertainty"

BATCH_SIZE = 64
N_BINS = 10
TEMPERATURES = np.round(np.arange(0.5, 3.01, 0.1), 1)

MODEL_CONFIGS = {
    "pedestrian": {
        "label": "has_pedestrian",
        "weights": MODEL_DIR / "pedestrian_model.pth",
    },
    "traffic_light": {
        "label": "has_traffic_light",
        "weights": MODEL_DIR / "traffic_light_model.pth",
    },
    "vehicle": {
        "label": "has_vehicle",
        "weights": MODEL_DIR / "vehicle_model.pth",
    },
}


class CarlaDataset(Dataset):
    """One image with the labels needed by all three binary classifiers."""

    def __init__(self, split):
        split_dir = DATA_DIR / split
        self.data = pd.read_csv(split_dir / "labels.csv")
        self.image_dir = split_dir / "rgb-front"
        self.label_columns = [config["label"] for config in MODEL_CONFIGS.values()]
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_path = self.image_dir / f"{int(row['frame']):06d}.jpg"
        image = self.transform(Image.open(image_path).convert("RGB"))
        labels = torch.tensor(
            [float(row[column]) for column in self.label_columns],
            dtype=torch.float32,
        )
        return image, labels


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_models(device):
    loaded_models = {}
    for name, config in MODEL_CONFIGS.items():
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        state_dict = torch.load(config["weights"], map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        loaded_models[name] = model
    return loaded_models


def collect_logits(loader, loaded_models, device):
    """Run every model on the same images and return aligned logits/labels."""

    logits = {name: [] for name in MODEL_CONFIGS}
    labels = {name: [] for name in MODEL_CONFIGS}

    with torch.inference_mode():
        for images, batch_labels in loader:
            images = images.to(device)

            for label_index, (name, model) in enumerate(loaded_models.items()):
                outputs = model(images).squeeze(1)
                logits[name].append(outputs.cpu())
                labels[name].append(batch_labels[:, label_index])

    return {
        name: {
            "logits": torch.cat(logits[name]).numpy(),
            "labels": torch.cat(labels[name]).numpy().astype(int),
        }
        for name in MODEL_CONFIGS
    }


def negative_log_likelihood(logits, labels, temperature):
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    return F.binary_cross_entropy_with_logits(
        logits_tensor / temperature,
        labels_tensor,
    ).item()


def fit_temperature(logits, labels):
    """Pick the grid temperature with the lowest validation NLL."""

    search_rows = []
    for temperature in TEMPERATURES:
        nll = negative_log_likelihood(logits, labels, temperature)
        search_rows.append({"temperature": temperature, "validation_nll": nll})

    best_row = min(search_rows, key=lambda row: row["validation_nll"])
    return float(best_row["temperature"]), search_rows


def probabilities_from_logits(logits, temperature=1.0):
    scaled_logits = torch.tensor(logits / temperature, dtype=torch.float32)
    return torch.sigmoid(scaled_logits).numpy()


def reliability_statistics(logits, labels, temperature=1.0, n_bins=N_BINS):
    """Return standard confidence-vs-accuracy bins and ECE.

    For a binary classifier, confidence is the probability of the predicted class:
    p when the prediction is positive and 1-p when it is negative.
    """

    positive_probabilities = probabilities_from_logits(logits, temperature)
    predictions = (positive_probabilities >= 0.5).astype(int)
    confidence = np.where(
        predictions == 1,
        positive_probabilities,
        1.0 - positive_probabilities,
    )
    correct = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confidence, bin_edges[1:-1], right=False)
    rows = []
    ece = 0.0

    for bin_index in range(n_bins):
        in_bin = bin_indices == bin_index
        count = int(in_bin.sum())
        if count == 0:
            average_confidence = np.nan
            accuracy = np.nan
        else:
            average_confidence = float(confidence[in_bin].mean())
            accuracy = float(correct[in_bin].mean())
            ece += count / len(labels) * abs(accuracy - average_confidence)

        rows.append(
            {
                "bin_lower": bin_edges[bin_index],
                "bin_upper": bin_edges[bin_index + 1],
                "count": count,
                "average_confidence": average_confidence,
                "accuracy": accuracy,
            }
        )

    mean_confidence = float(confidence.mean())
    overall_accuracy = float(correct.mean())
    confidence_gap = mean_confidence - overall_accuracy

    return {
        "ece": float(ece),
        "accuracy": overall_accuracy,
        "mean_confidence": mean_confidence,
        "confidence_gap": confidence_gap,
        "bins": rows,
    }


def calibration_pattern(confidence_gap):
    if confidence_gap > 0:
        return "overconfident"
    if confidence_gap < 0:
        return "underconfident"
    return "calibrated"


def plot_reliability(all_statistics, output_path):
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)

    for column, name in enumerate(MODEL_CONFIGS):
        for row, stage in enumerate(["uncalibrated", "calibrated"]):
            axis = axes[row, column]
            statistics = all_statistics[name][stage]
            populated_bins = [item for item in statistics["bins"] if item["count"]]
            confidence = [item["average_confidence"] for item in populated_bins]
            accuracy = [item["accuracy"] for item in populated_bins]

            axis.plot([0.5, 1.0], [0.5, 1.0], "--", color="black", label="ideal")
            axis.plot(confidence, accuracy, "o-", color="#1565c0", label="model")
            axis.set_xlim(0.5, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.grid(alpha=0.25)
            axis.set_title(
                f"{name.replace('_', ' ').title()} - {stage}\n"
                f"ECE = {statistics['ece']:.4f}"
            )

            if column == 0:
                axis.set_ylabel("Empirical accuracy")
            if row == 1:
                axis.set_xlabel("Mean confidence")
            if row == 0 and column == 0:
                axis.legend(loc="lower right")

    figure.suptitle("CARLA Reliability Diagrams", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def evaluate_pedestrian_cost(logits, labels, temperature):
    false_negative_cost = 100
    false_positive_cost = 1
    optimal_threshold = false_positive_cost / (
        false_negative_cost + false_positive_cost
    )
    rows = []

    for stage, current_temperature in [
        ("uncalibrated", 1.0),
        ("calibrated", temperature),
    ]:
        probabilities = probabilities_from_logits(logits, current_temperature)

        for threshold_name, threshold in [
            ("0.5", 0.5),
            ("tau_star", optimal_threshold),
        ]:
            predictions = (probabilities >= threshold).astype(int)
            false_negatives = int(((predictions == 0) & (labels == 1)).sum())
            false_positives = int(((predictions == 1) & (labels == 0)).sum())
            total_loss = (
                false_negative_cost * false_negatives
                + false_positive_cost * false_positives
            )
            rows.append(
                {
                    "calibration": stage,
                    "threshold_name": threshold_name,
                    "threshold": threshold,
                    "false_negatives": false_negatives,
                    "false_positives": false_positives,
                    "total_loss": total_loss,
                }
            )

    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    print(f"Using device: {device}")

    validation_loader = DataLoader(
        CarlaDataset("validation"),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        CarlaDataset("test"),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    loaded_models = load_models(device)
    print("Collecting validation logits...")
    validation_outputs = collect_logits(validation_loader, loaded_models, device)
    print("Collecting test logits...")
    test_outputs = collect_logits(test_loader, loaded_models, device)

    calibration_rows = []
    temperature_search_rows = []
    reliability_rows = []
    all_statistics = {}
    fitted_temperatures = {}

    for name in MODEL_CONFIGS:
        validation_logits = validation_outputs[name]["logits"]
        validation_labels = validation_outputs[name]["labels"]
        test_logits = test_outputs[name]["logits"]
        test_labels = test_outputs[name]["labels"]

        temperature, search_rows = fit_temperature(
            validation_logits,
            validation_labels,
        )
        fitted_temperatures[name] = temperature

        for row in search_rows:
            temperature_search_rows.append({"model": name, **row})

        before = reliability_statistics(test_logits, test_labels, temperature=1.0)
        after = reliability_statistics(test_logits, test_labels, temperature=temperature)
        all_statistics[name] = {
            "uncalibrated": before,
            "calibrated": after,
        }

        for stage, statistics in all_statistics[name].items():
            for bin_row in statistics["bins"]:
                reliability_rows.append(
                    {"model": name, "calibration": stage, **bin_row}
                )

        calibration_rows.append(
            {
                "model": name,
                "temperature": temperature,
                "validation_nll_before": negative_log_likelihood(
                    validation_logits, validation_labels, 1.0
                ),
                "validation_nll_after": negative_log_likelihood(
                    validation_logits, validation_labels, temperature
                ),
                "test_ece_before": before["ece"],
                "test_ece_after": after["ece"],
                "test_accuracy": before["accuracy"],
                "mean_confidence_before": before["mean_confidence"],
                "mean_confidence_after": after["mean_confidence"],
                "confidence_gap_before": before["confidence_gap"],
                "confidence_gap_after": after["confidence_gap"],
                "pattern_before": calibration_pattern(before["confidence_gap"]),
                "pattern_after": calibration_pattern(after["confidence_gap"]),
            }
        )

    calibration_frame = pd.DataFrame(calibration_rows)
    calibration_frame.to_csv(OUTPUT_DIR / "calibration_results.csv", index=False)
    pd.DataFrame(temperature_search_rows).to_csv(
        OUTPUT_DIR / "temperature_search.csv",
        index=False,
    )
    pd.DataFrame(reliability_rows).to_csv(
        OUTPUT_DIR / "reliability_bins.csv",
        index=False,
    )
    plot_reliability(
        all_statistics,
        OUTPUT_DIR / "reliability_diagrams.png",
    )

    pedestrian_cost_rows = evaluate_pedestrian_cost(
        test_outputs["pedestrian"]["logits"],
        test_outputs["pedestrian"]["labels"],
        fitted_temperatures["pedestrian"],
    )
    cost_frame = pd.DataFrame(pedestrian_cost_rows)
    cost_frame.to_csv(OUTPUT_DIR / "pedestrian_cost_results.csv", index=False)

    print("\nCalibration results")
    print(
        calibration_frame[
            [
                "model",
                "temperature",
                "test_ece_before",
                "test_ece_after",
                "test_accuracy",
                "pattern_before",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    print("\nPedestrian total loss (C_FN=100, C_FP=1)")
    print(
        cost_frame.pivot(
            index="calibration",
            columns="threshold_name",
            values="total_loss",
        ).to_string()
    )
    print(f"\nSaved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()