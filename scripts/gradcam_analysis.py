import os
import cv2
import torch
import numpy as np

from PIL import Image
from torchvision import transforms, models

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# =========================================================
# CONFIG
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 224

MAX_IMAGES_PER_FOLDER = 20

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# =========================================================
# LOAD MODEL
# =========================================================

def load_model(model_path, num_classes=1):

    model = models.resnet18(weights=None)

    # Binary classification output
    model.fc = torch.nn.Linear(
        model.fc.in_features,
        num_classes
    )

    # Load trained weights
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    print(f"\nLoaded model: {model_path}")

    return model

# =========================================================
# LOAD IMAGE
# =========================================================

def preprocess_image(image_path):

    image = Image.open(image_path).convert("RGB")

    # Resize image for BOTH:
    # visualization and model input
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    rgb_img = np.array(image).astype(np.float32) / 255.0

    input_tensor = transform(image).unsqueeze(0)

    return rgb_img, input_tensor.to(DEVICE)

# =========================================================
# GENERATE GRADCAM
# =========================================================

def generate_gradcam(model, image_path, output_path):

    rgb_img, input_tensor = preprocess_image(image_path)

    # Last convolution block of ResNet18
    target_layers = [model.layer4[-1]]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    # Forward prediction
    outputs = model(input_tensor)

    predicted_score = torch.sigmoid(outputs).item()

    predicted_label = (
        "Detected"
        if predicted_score >= 0.5
        else "Not Detected"
    )

    # For binary classification
    targets = [ClassifierOutputTarget(0)]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    # Add prediction text (smaller version)
    cv2.putText(
        visualization,
        f"{predicted_label}: {predicted_score:.2f}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # Save image
    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )

    print(f"Saved GradCAM: {output_path}")

# =========================================================
# RUN ANALYSIS
# =========================================================

def run_analysis(model_path, image_folder, output_folder, labels_path, label_column, positive_label):

    print("\n=================================================")
    print(f"Processing Folder: {image_folder}")
    print("=================================================")

    os.makedirs(output_folder, exist_ok=True)

    model = load_model(model_path)

    # ---------------------------------------------
    # LOAD LABELS
    # ---------------------------------------------

    import pandas as pd

    df = pd.read_csv(labels_path)

    # Filter positive samples dynamically
    filtered_df = df[df[label_column] == positive_label]

    print(f"Found {len(filtered_df)} positive samples for {label_column}")

    # Limit number for faster processing
    filtered_df = filtered_df.head(MAX_IMAGES_PER_FOLDER)

    for _, row in filtered_df.iterrows():

        frame_number = int(row["frame"])

        # Convert frame -> filename
        image_name = f"{frame_number:06d}.jpg"

        image_path = os.path.join(
            image_folder,
            image_name
        )

        output_path = os.path.join(
            output_folder,
            f"gradcam_{image_name}"
        )

        # Check image exists
        if not os.path.exists(image_path):

            print(f"Image not found: {image_path}")
            continue

        try:

            generate_gradcam(
                model,
                image_path,
                output_path
            )

        except Exception as e:

            print(f"Error processing {image_name}")
            print(e)

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # =====================================================
    # PEDESTRIAN DETECTION
    # =====================================================

    # BASELINE
    run_analysis(
        model_path="models/pedestrian_model.pth",
        image_folder="data/validation/rgb-front",
        output_folder="outputs/explainability/pedestrian/baseline",
        labels_path="data/validation/labels.csv",
        label_column="has_pedestrian",
        positive_label=True
    )

    # FOG
    run_analysis(
        model_path="models/pedestrian_model.pth",
        image_folder="data/test-fog/rgb-front",
        output_folder="outputs/explainability/pedestrian/fog",
        labels_path="data/test-fog/labels.csv",
        label_column="has_pedestrian",
        positive_label=True
    )

    # NIGHT
    run_analysis(
        model_path="models/pedestrian_model.pth",
        image_folder="data/test-night/rgb-front",
        output_folder="outputs/explainability/pedestrian/night",
        labels_path="data/test-night/labels.csv",
        label_column="has_pedestrian",
        positive_label=True
    )

    # TOWN01
    run_analysis(
        model_path="models/pedestrian_model.pth",
        image_folder="data/test-town-01/rgb-front",
        output_folder="outputs/explainability/pedestrian/town01",
        labels_path="data/test-town-01/labels.csv",
        label_column="has_pedestrian",
        positive_label=True
    )

    # =====================================================
    # TRAFFIC LIGHT DETECTION
    # =====================================================

    # BASELINE
    run_analysis(
        model_path="models/traffic_light_model.pth",
        image_folder="data/validation/rgb-front",
        output_folder="outputs/explainability/traffic_light/baseline",
        labels_path="data/validation/labels.csv",
        label_column="has_traffic_light",
        positive_label=True
    )

    # FOG
    run_analysis(
        model_path="models/traffic_light_model.pth",
        image_folder="data/test-fog/rgb-front",
        output_folder="outputs/explainability/traffic_light/fog",
        labels_path="data/test-fog/labels.csv",
        label_column="has_traffic_light",
        positive_label=True
    )

    # NIGHT
    run_analysis(
        model_path="models/traffic_light_model.pth",
        image_folder="data/test-night/rgb-front",
        output_folder="outputs/explainability/traffic_light/night",
        labels_path="data/test-night/labels.csv",
        label_column="has_traffic_light",
        positive_label=True
    )

    # TOWN01
    run_analysis(
        model_path="models/traffic_light_model.pth",
        image_folder="data/test-town-01/rgb-front",
        output_folder="outputs/explainability/traffic_light/town01",
        labels_path="data/test-town-01/labels.csv",
        label_column="has_traffic_light",
        positive_label=True
    )

    # =====================================================
    # VEHICLE DETECTION
    # =====================================================

    # BASELINE
    run_analysis(
        model_path="models/vehicle_model.pth",
        image_folder="data/validation/rgb-front",
        output_folder="outputs/explainability/vehicle/baseline",
        labels_path="data/validation/labels.csv",
        label_column="has_vehicle",
        positive_label=True
    )

    # FOG
    run_analysis(
        model_path="models/vehicle_model.pth",
        image_folder="data/test-fog/rgb-front",
        output_folder="outputs/explainability/vehicle/fog",
        labels_path="data/test-fog/labels.csv",
        label_column="has_vehicle",
        positive_label=True
    )

    # NIGHT
    run_analysis(
        model_path="models/vehicle_model.pth",
        image_folder="data/test-night/rgb-front",
        output_folder="outputs/explainability/vehicle/night",
        labels_path="data/test-night/labels.csv",
        label_column="has_vehicle",
        positive_label=True
    )

    # TOWN01
    run_analysis(
        model_path="models/vehicle_model.pth",
        image_folder="data/test-town-01/rgb-front",
        output_folder="outputs/explainability/vehicle/town01",
        labels_path="data/test-town-01/labels.csv",
        label_column="has_vehicle",
        positive_label=True
    )

    print("\n=================================================")
    print("GradCAM analysis completed successfully!")
    print("=================================================")