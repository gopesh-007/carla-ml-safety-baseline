# 1. load pedestrian model
# 2. extract ResNet features
# 3. fit k-NN on validation features
# 4. compute distances:
    # * validation
    # * fog
    # * night
    # * town01
# 5. compute AUROC
# 6. compare with MSP

import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms, models

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

# ======================================================
# CONFIG
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 224

MAX_IMAGES = 1000

MODEL_PATH = "models/pedestrian_model.pth"

# ======================================================
# TRANSFORM
# ======================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# ======================================================
# LOAD MODEL
# ======================================================

def load_model(model_path):

    model = models.resnet18(weights=None)

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        1
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    return model

# ======================================================
# FEATURE EXTRACTOR
# ======================================================

class FeatureExtractor(torch.nn.Module):

    def __init__(self, model):

        super().__init__()

        self.features = torch.nn.Sequential(
            *list(model.children())[:-1]
        )

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(x, 1)

        return x

# ======================================================
# LOAD IMAGE
# ======================================================

def preprocess_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    return image.unsqueeze(0).to(DEVICE)

# ======================================================
# EXTRACT FEATURES
# ======================================================

def extract_features(feature_model, image_folder):

    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.endswith(".jpg")
    ])

    image_files = image_files[:MAX_IMAGES]

    features = []

    for image_name in image_files:

        image_path = os.path.join(
            image_folder,
            image_name
        )

        try:

            input_tensor = preprocess_image(image_path)

            with torch.no_grad():

                embedding = feature_model(
                    input_tensor
                )

                embedding = (
                    embedding
                    .cpu()
                    .numpy()
                    .flatten()
                )

                features.append(embedding)

        except Exception as e:

            print(f"Error: {image_name}")
            print(e)

    return np.array(features)

# ======================================================
# COMPUTE KNN DISTANCES
# ======================================================

def compute_knn_scores(
    knn,
    features
):

    distances, _ = knn.kneighbors(
        features
    )

    return distances.mean(axis=1)

# ======================================================
# PLOT HISTOGRAM
# ======================================================

def plot_histogram(
    id_scores,
    fog_scores,
    night_scores,
    town_scores
):

    plt.figure(figsize=(10, 6))

    plt.hist(
        id_scores,
        bins=30,
        alpha=0.6,
        label="Validation (ID)"
    )

    plt.hist(
        fog_scores,
        bins=30,
        alpha=0.6,
        label="Fog"
    )

    plt.hist(
        night_scores,
        bins=30,
        alpha=0.6,
        label="Night"
    )

    plt.hist(
        town_scores,
        bins=30,
        alpha=0.6,
        label="Town01"
    )

    plt.xlabel("k-NN Distance")
    plt.ylabel("Number of Samples")

    plt.title(
        "Feature-Based OOD Distribution"
    )

    plt.legend()

    os.makedirs(
        "outputs/ood/plots",
        exist_ok=True
    )

    plt.savefig(
        "outputs/ood/plots/knn_histogram.png"
    )

    plt.close()

# ======================================================
# COMPUTE AUROC
# ======================================================

def compute_auroc(
    id_scores,
    ood_scores
):

    labels = (
        [0] * len(id_scores) +
        [1] * len(ood_scores)
    )

    scores = (
        list(id_scores) +
        list(ood_scores)
    )

    auroc = roc_auc_score(
        labels,
        scores
    )

    return auroc

# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    os.chdir(PROJECT_ROOT)
    print("\nLoading model...")

    model = load_model(MODEL_PATH)

    feature_model = FeatureExtractor(model)

    # ==================================================
    # EXTRACT FEATURES
    # ==================================================

    print("\nExtracting validation features...")

    validation_features = extract_features(
        feature_model,
        "data/validation/rgb-front"
    )

    print("Extracting fog features...")

    fog_features = extract_features(
        feature_model,
        "data/test-fog/rgb-front"
    )

    print("Extracting night features...")

    night_features = extract_features(
        feature_model,
        "data/test-night/rgb-front"
    )

    print("Extracting town01 features...")

    town_features = extract_features(
        feature_model,
        "data/test-town-01/rgb-front"
    )

    # ==================================================
    # FIT KNN
    # ==================================================

    print("\nFitting k-NN detector...")

    knn = NearestNeighbors(
        n_neighbors=5
    )

    knn.fit(validation_features)

    # ==================================================
    # COMPUTE DISTANCES
    # ==================================================

    validation_scores = compute_knn_scores(
        knn,
        validation_features
    )

    fog_scores = compute_knn_scores(
        knn,
        fog_features
    )

    night_scores = compute_knn_scores(
        knn,
        night_features
    )

    town_scores = compute_knn_scores(
        knn,
        town_features
    )

    # ==================================================
    # PLOT HISTOGRAM
    # ==================================================

    print("\nGenerating histogram...")

    plot_histogram(
        validation_scores,
        fog_scores,
        night_scores,
        town_scores
    )

    # ==================================================
    # COMPUTE AUROC
    # ==================================================

    fog_auroc = compute_auroc(
        validation_scores,
        fog_scores
    )

    night_auroc = compute_auroc(
        validation_scores,
        night_scores
    )

    town_auroc = compute_auroc(
        validation_scores,
        town_scores
    )

    # ==================================================
    # RESULTS
    # ==================================================

    print("\n==============================")
    print("k-NN OOD RESULTS")
    print("==============================")

    print(f"Fog AUROC: {fog_auroc:.4f}")
    print(f"Night AUROC: {night_auroc:.4f}")
    print(f"Town01 AUROC: {town_auroc:.4f}")

    print("\nHistogram saved:")
    print("outputs/ood/plots/knn_histogram.png")