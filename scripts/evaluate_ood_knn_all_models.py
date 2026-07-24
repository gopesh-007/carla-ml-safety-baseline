"""Evaluate feature-based k-NN OOD detection for all CARLA classifiers.

The detector is fitted on 1,000 in-distribution validation embeddings for each
classifier and then evaluated against fog, night, and Town-01 images.

Run from the project root:
    python scripts/evaluate_ood_knn_all_models.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIGS = {
    "pedestrian": PROJECT_ROOT / "models" / "pedestrian_model.pth",
    "traffic_light": PROJECT_ROOT / "models" / "traffic_light_model.pth",
    "vehicle": PROJECT_ROOT / "models" / "vehicle_model.pth",
}
DATASETS = {
    "validation": PROJECT_ROOT / "data" / "validation" / "rgb-front",
    "fog": PROJECT_ROOT / "data" / "test-fog" / "rgb-front",
    "night": PROJECT_ROOT / "data" / "test-night" / "rgb-front",
    "town01": PROJECT_ROOT / "data" / "test-town-01" / "rgb-front",
}


class ImageFolderDataset(Dataset):
    """A deterministic subset of CARLA RGB images without labels."""

    def __init__(self, image_dir, limit):
        self.image_paths = sorted(image_dir.glob("*.jpg"))[:limit]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        with Image.open(self.image_paths[index]) as image:
            return self.transform(image.convert("RGB"))


class FeatureExtractor(nn.Module):
    """ResNet-18 features before the binary classification layer."""

    def __init__(self, classifier):
        super().__init__()
        self.features = nn.Sequential(*list(classifier.children())[:-1])

    def forward(self, images):
        return torch.flatten(self.features(images), 1)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run feature-based k-NN OOD detection for CARLA classifiers."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CONFIGS,
        default=list(MODEL_CONFIGS),
        help="Models to evaluate (default: all three).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum deterministic images per dataset (default: 1000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Images processed together (default: 64).",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save one OOD-score histogram per model after evaluation.",
    )
    return parser.parse_args()


def load_classifier(checkpoint_path, device):
    classifier = models.resnet18(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, 1)
    classifier.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    return classifier.to(device).eval()


def extract_features(feature_model, image_dir, limit, batch_size, device):
    dataset = ImageFolderDataset(image_dir, limit)
    if not dataset:
        raise FileNotFoundError(f"No JPG images found in {image_dir}")

    loader = DataLoader(dataset, batch_size=batch_size)
    features = []
    with torch.no_grad():
        for images in loader:
            embeddings = feature_model(images.to(device))
            features.append(embeddings.cpu().numpy())
    return np.concatenate(features)


def mean_knn_distance(neighbors, features):
    distances, _ = neighbors.kneighbors(features)
    return distances.mean(axis=1)


def compute_auroc(id_scores, ood_scores):
    labels = np.concatenate([
        np.zeros(len(id_scores), dtype=int),
        np.ones(len(ood_scores), dtype=int),
    ])
    scores = np.concatenate([id_scores, ood_scores])
    return roc_auc_score(labels, scores)


def save_histogram(model_name, id_scores, scores_by_condition, output_dir):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.hist(id_scores, bins=30, alpha=0.6, label="Validation (ID)")
    for condition, scores in scores_by_condition.items():
        plt.hist(scores, bins=30, alpha=0.6, label=condition.title())
    plt.xlabel("Mean distance to 5 nearest validation embeddings")
    plt.ylabel("Number of images")
    plt.title(f"Feature-based k-NN OOD scores - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"knn_histogram_{model_name}.png", dpi=150)
    plt.close()


def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = PROJECT_ROOT / "outputs" / "ood" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    print("Using device:", device)
    for model_name in args.models:
        print(f"\nEvaluating {model_name} model...")
        classifier = load_classifier(MODEL_CONFIGS[model_name], device)
        feature_model = FeatureExtractor(classifier).to(device).eval()

        validation_features = extract_features(
            feature_model,
            DATASETS["validation"],
            args.limit,
            args.batch_size,
            device,
        )
        neighbors = NearestNeighbors(n_neighbors=5).fit(validation_features)
        validation_scores = mean_knn_distance(neighbors, validation_features)

        scores_by_condition = {}
        for condition in ("fog", "night", "town01"):
            ood_features = extract_features(
                feature_model,
                DATASETS[condition],
                args.limit,
                args.batch_size,
                device,
            )
            ood_scores = mean_knn_distance(neighbors, ood_features)
            scores_by_condition[condition] = ood_scores
            auroc = compute_auroc(validation_scores, ood_scores)
            results.append({
                "model": model_name,
                "condition": condition,
                "detector": "feature_kNN_k5",
                "images_per_set": args.limit,
                "auroc": auroc,
                "threshold": 0.90,
                "verdict": "met" if auroc >= 0.90 else "not_met",
            })
            print(f"  {condition.title()} AUROC: {auroc:.4f}")

        if args.save_plots:
            save_histogram(model_name, validation_scores, scores_by_condition, output_dir)

        results_path = PROJECT_ROOT / "outputs" / "ood" / "knn_all_models_results.csv"
        new_results = pd.DataFrame(results)
        if results_path.exists():
            existing_results = pd.read_csv(results_path)
            combined_results = pd.concat([existing_results, new_results])
            combined_results = combined_results.drop_duplicates(
                subset=["model", "condition"],
                keep="last",
            )
        else:
            combined_results = new_results
        combined_results.to_csv(results_path, index=False)
        print(f"Saved results: {results_path}")


if __name__ == "__main__":
    main()
