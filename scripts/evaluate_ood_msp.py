#1Load Model
#2Load Images
#3Compute Confidence
#4Store Scores
#5Plot Histogram
#6Compute AUROC

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score

# ======================================================
# CONFIG
# ======================================================

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
# LOAD IMAGE
# ======================================================

def preprocess_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    return image.unsqueeze(0).to(DEVICE)

# ======================================================
# COMPUTE MSP SCORES
# ======================================================

def compute_msp_scores(model, image_folder):

    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.endswith(".jpg")
    ])

    image_files = image_files[:MAX_IMAGES]

    scores = []

    for image_name in image_files:

        image_path = os.path.join(
            image_folder,
            image_name
        )

        try:

            input_tensor = preprocess_image(image_path)

            with torch.no_grad():

                output = model(input_tensor)

                probability = torch.sigmoid(output).item()

                # MSP for binary classification
                msp = max(
                    probability,
                    1 - probability
                )

                scores.append(msp)

        except Exception as e:

            print(f"Error: {image_name}")
            print(e)

    return scores

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

    plt.xlabel("MSP Confidence")
    plt.ylabel("Number of Samples")

    plt.title(
        "MSP Confidence Distribution"
    )

    plt.legend()

    os.makedirs(
        "outputs/ood/plots",
        exist_ok=True
    )

    plt.savefig(
        "outputs/ood/plots/msp_histogram.png"
    )

    plt.close()

# ======================================================
# COMPUTE AUROC
# ======================================================

def compute_auroc(id_scores, ood_scores):

    labels = (
        [0] * len(id_scores) +
        [1] * len(ood_scores)
    )

    scores = id_scores + ood_scores

    auroc = roc_auc_score(
        labels,
        [-s for s in scores]
    )

    return auroc

# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    print("\nLoading model...")

    model = load_model(MODEL_PATH)

    # ==================================================
    # COMPUTE SCORES
    # ==================================================

    print("\nComputing validation scores...")

    validation_scores = compute_msp_scores(
        model,
        "data/validation/rgb-front"
    )

    print("Computing fog scores...")

    fog_scores = compute_msp_scores(
        model,
        "data/test-fog/rgb-front"
    )

    print("Computing night scores...")

    night_scores = compute_msp_scores(
        model,
        "data/test-night/rgb-front"
    )

    print("Computing town01 scores...")

    town_scores = compute_msp_scores(
        model,
        "data/test-town-01/rgb-front"
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
    # PRINT RESULTS
    # ==================================================

    print("\n==============================")
    print("OOD DETECTION RESULTS")
    print("==============================")

    print(f"Fog AUROC: {fog_auroc:.4f}")
    print(f"Night AUROC: {night_auroc:.4f}")
    print(f"Town01 AUROC: {town_auroc:.4f}")

    print("\nHistogram saved:")
    print("outputs/ood/plots/msp_histogram.png")