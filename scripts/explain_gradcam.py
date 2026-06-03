"""
Exercise 6.5 & 6.6 — Grad-CAM Explainability for CARLA Models
==============================================================
Applies Grad-CAM to all three binary classifiers:
  - Pedestrian detector
  - Traffic light detector
  - Vehicle detector

For each model:
  - Selects 5 correctly classified images (at least one per model)
  - Selects 3 misclassified images
  - Generates Grad-CAM heatmaps overlaid on original images

Also tests OOD conditions (fog, night, town01) for Exercise 6.6.

Usage (from the scripts/ directory):
  python explain_gradcam.py
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# ------------------------------------------------------------------ #
#  REPRODUCIBILITY
# ------------------------------------------------------------------ #
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ------------------------------------------------------------------ #
#  PATHS
# ------------------------------------------------------------------ #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "explainability")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_CONFIGS = [
    {
        "name": "pedestrian",
        "label_col": "has_pedestrian",
        "model_file": "pedestrian_model.pth",
        "display": "Pedestrian Detector",
    },
    {
        "name": "traffic_light",
        "label_col": "has_traffic_light",
        "model_file": "traffic_light_model.pth",
        "display": "Traffic Light Detector",
    },
    {
        "name": "vehicle",
        "label_col": "has_vehicle",
        "model_file": "vehicle_model.pth",
        "display": "Vehicle Detector",
    },
]

CONDITIONS = {
    "baseline": ("test", "rgb-front"),
    "fog":      ("test-fog", "rgb-front"),
    "night":    ("test-night", "rgb-front"),
    "town01":   ("test-town-01", "rgb-front"),
}

# ------------------------------------------------------------------ #
#  TRANSFORMS
# ------------------------------------------------------------------ #
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

DENORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
DENORM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denorm(tensor):
    """Undo ImageNet normalisation → [0,1] numpy."""
    t = tensor.cpu() * DENORM_STD + DENORM_MEAN
    t = t.clamp(0, 1).numpy().transpose(1, 2, 0)
    return t


# ------------------------------------------------------------------ #
#  MODEL FACTORY
# ------------------------------------------------------------------ #
def load_model(model_file):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    path = os.path.join(MODEL_DIR, model_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


# ------------------------------------------------------------------ #
#  GRAD-CAM (hooks on the last conv layer: layer4)
# ------------------------------------------------------------------ #
class GradCAM:
    """
    Grad-CAM for a ResNet-18 binary classifier.

    Method:
      1. Forward pass — store the last conv feature maps A^k  (shape HxW per channel k)
      2. Backward pass — compute gradient of class score y_c w.r.t. each A^k
      3. Global-average-pool the gradients → scalar weight α^c_k per channel
      4. Weighted sum + ReLU → L_GradCAM  (slide 32 in the lecture)
      5. Upsample to input resolution

    Chosen because:
      - Works on any CNN without architecture modification (unlike CAM which
        needs a GAP layer directly before the classifier)
      - Produces spatially faithful, class-discriminative heatmaps
      - Standard reference method for bias / fairness audits (lecture slide 33)
    """

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        # Last residual block of ResNet-18
        target_layer = self.model.layer4[-1].conv2

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, tensor):
        """
        Args:
            tensor: (1, 3, 224, 224) preprocessed image tensor
        Returns:
            cam (numpy, H x W): Grad-CAM heatmap in [0,1]
            prob (float): sigmoid probability
        """
        self.model.zero_grad()
        tensor = tensor.clone().requires_grad_(True)

        output = self.model(tensor)          # raw logit
        prob = torch.sigmoid(output).item()

        # Backprop the single binary output
        output.backward()

        # α^c_k = (1/Z) * Σ_i Σ_j  ∂y_c / ∂A^k_ij
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination + ReLU
        cam = (weights * self.activations).sum(dim=1, keepdim=True)   # (1,1,7,7)
        cam = torch.relu(cam)

        # Normalise
        cam = cam.squeeze().cpu().numpy()                              # (7, 7)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, prob


# ------------------------------------------------------------------ #
#  IMAGE + LABEL LOADER
# ------------------------------------------------------------------ #
def load_split(condition_name):
    """Return (csv_df, image_dir) for a given condition key."""
    split_name, img_subdir = CONDITIONS[condition_name]
    split_dir = os.path.join(DATA_DIR, split_name)
    csv_path = os.path.join(split_dir, "labels.csv")
    img_dir  = os.path.join(split_dir, img_subdir)
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path)
    return df, img_dir


def load_image_tensor(frame_id, img_dir):
    """Load a single image as a (1,3,224,224) tensor."""
    name = f"{int(frame_id):06d}.jpg"
    path = os.path.join(img_dir, name)
    img = Image.open(path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0)
    return tensor, img


# ------------------------------------------------------------------ #
#  PREDICTION HELPER
# ------------------------------------------------------------------ #
def predict(model, tensor):
    with torch.no_grad():
        logit = model(tensor)
    prob = torch.sigmoid(logit).item()
    return int(prob > 0.5), prob


# ------------------------------------------------------------------ #
#  COLLECT CORRECT / MISCLASSIFIED SAMPLES
# ------------------------------------------------------------------ #
def collect_samples(model, df, img_dir, label_col, n_correct=5, n_wrong=3):
    """
    Returns:
        correct: list of (frame_id, label, pred, prob, tensor, pil_img)
        wrong:   list of (frame_id, label, pred, prob, tensor, pil_img)
    """
    correct, wrong = [], []
    df_shuffled = df.sample(frac=1, random_state=42)

    for _, row in df_shuffled.iterrows():
        if len(correct) >= n_correct and len(wrong) >= n_wrong:
            break
        frame_id = row["frame"]
        true_label = int(row[label_col])
        try:
            tensor, pil_img = load_image_tensor(frame_id, img_dir)
        except Exception:
            continue
        pred, prob = predict(model, tensor)
        entry = (frame_id, true_label, pred, prob, tensor, pil_img)
        if pred == true_label and len(correct) < n_correct:
            correct.append(entry)
        elif pred != true_label and len(wrong) < n_wrong:
            wrong.append(entry)

    return correct, wrong


# ------------------------------------------------------------------ #
#  PLOTTING
# ------------------------------------------------------------------ #
def overlay_cam(ax, pil_img, cam, prob, true_label, pred, title=""):
    """Draw a heatmap overlay on ax."""
    img_np = np.array(pil_img)
    h, w = img_np.shape[:2]

    # Upsample cam to image size
    from PIL import Image as PILImage
    cam_pil = PILImage.fromarray((cam * 255).astype(np.uint8))
    cam_pil = cam_pil.resize((w, h), PILImage.BILINEAR)
    cam_up  = np.array(cam_pil) / 255.0

    # Colour map
    heatmap = cm.jet(cam_up)[:, :, :3]
    overlay = (0.5 * img_np / 255.0 + 0.5 * heatmap).clip(0, 1)

    ax.imshow(overlay)
    correct = (pred == true_label)
    status = "✓ Correct" if correct else "✗ Wrong"
    colour  = "green" if correct else "red"
    ax.set_title(
        f"{title}\nTrue={true_label} Pred={pred} ({prob:.2f})\n{status}",
        fontsize=8, color=colour
    )
    ax.axis("off")


def save_figure(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ------------------------------------------------------------------ #
#  EXERCISE 6.5 — CORRECT & MISCLASSIFIED PANELS
# ------------------------------------------------------------------ #
def run_ex65(model_cfg, model, gcam, df, img_dir):
    name    = model_cfg["name"]
    display = model_cfg["display"]
    label_col = model_cfg["label_col"]

    print(f"\n[6.5] {display} — collecting samples from baseline test set ...")
    correct, wrong = collect_samples(model, df, img_dir, label_col,
                                     n_correct=5, n_wrong=3)
    print(f"  Found {len(correct)} correct, {len(wrong)} wrong samples.")

    # ---- Correctly classified panel (5 images) ----
    if correct:
        fig, axes = plt.subplots(1, len(correct), figsize=(4 * len(correct), 4))
        if len(correct) == 1:
            axes = [axes]
        fig.suptitle(f"{display} — Grad-CAM on Correctly Classified Images", fontsize=12)
        for ax, (frame_id, true_lbl, pred, prob, tensor, pil_img) in zip(axes, correct):
            cam, _ = gcam.generate(tensor)
            overlay_cam(ax, pil_img, cam, prob, true_lbl, pred,
                        title=f"Frame {frame_id}")
        save_figure(fig, f"ex65_{name}_correct.png")

    # ---- Misclassified panel (3 images) ----
    if wrong:
        fig, axes = plt.subplots(1, len(wrong), figsize=(4 * len(wrong), 4))
        if len(wrong) == 1:
            axes = [axes]
        fig.suptitle(f"{display} — Grad-CAM on Misclassified Images", fontsize=12)
        for ax, (frame_id, true_lbl, pred, prob, tensor, pil_img) in zip(axes, wrong):
            cam, _ = gcam.generate(tensor)
            overlay_cam(ax, pil_img, cam, prob, true_lbl, pred,
                        title=f"Frame {frame_id}")
        save_figure(fig, f"ex65_{name}_wrong.png")

    return correct, wrong


# ------------------------------------------------------------------ #
#  EXERCISE 6.6 — OOD / CONDITION PANELS
# ------------------------------------------------------------------ #
def run_ex66(model_cfg, model, gcam, conditions_to_test=None):
    name      = model_cfg["name"]
    display   = model_cfg["display"]
    label_col = model_cfg["label_col"]

    if conditions_to_test is None:
        conditions_to_test = ["fog", "night", "town01"]

    results = {}  # condition -> {accuracy, n_correct, n_total}

    for cond in conditions_to_test:
        df, img_dir = load_split(cond)
        if df is None:
            print(f"  [6.6] {cond} split not found — skipping.")
            continue

        print(f"\n[6.6] {display} × {cond} ...")
        correct_s, wrong_s = collect_samples(model, df, img_dir, label_col,
                                              n_correct=3, n_wrong=3)

        # Quick accuracy over first 200 samples
        n_ok, n_tot = 0, 0
        for _, row in df.head(200).iterrows():
            try:
                tensor, _ = load_image_tensor(row["frame"], img_dir)
            except Exception:
                continue
            p, _ = predict(model, tensor)
            n_ok  += int(p == int(row[label_col]))
            n_tot += 1
        acc = n_ok / n_tot if n_tot else float("nan")
        results[cond] = {"accuracy": acc, "n_correct": n_ok, "n_total": n_tot}
        print(f"  Quick accuracy ({n_tot} imgs): {acc:.3f}")

        # Combined panel: correct + wrong side by side
        samples = correct_s[:3] + wrong_s[:3]
        n = len(samples)
        if n == 0:
            continue
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        fig.suptitle(f"{display} — Condition: {cond.upper()} (Grad-CAM)", fontsize=11)
        for ax, (frame_id, true_lbl, pred, prob, tensor, pil_img) in zip(axes, samples):
            cam, _ = gcam.generate(tensor)
            overlay_cam(ax, pil_img, cam, prob, true_lbl, pred,
                        title=f"{cond} / F{frame_id}")
        save_figure(fig, f"ex66_{name}_{cond}.png")

    return results


# ------------------------------------------------------------------ #
#  ACCURACY SUMMARY ACROSS CONDITIONS (for Ex 6.6c)
# ------------------------------------------------------------------ #
def accuracy_summary(model, df, img_dir, label_col, n=200):
    n_ok, n_tot = 0, 0
    for _, row in df.head(n).iterrows():
        try:
            tensor, _ = load_image_tensor(row["frame"], img_dir)
        except Exception:
            continue
        p, _ = predict(model, tensor)
        n_ok  += int(p == int(row[label_col]))
        n_tot += 1
    return n_ok / n_tot if n_tot else float("nan"), n_tot


# ------------------------------------------------------------------ #
#  MAIN
# ------------------------------------------------------------------ #
def main():
    print("=" * 60)
    print("Grad-CAM Explainability — Exercises 6.5 & 6.6")
    print("=" * 60)

    all_results = {}

    for model_cfg in MODEL_CONFIGS:
        name      = model_cfg["name"]
        label_col = model_cfg["label_col"]
        display   = model_cfg["display"]

        print(f"\n{'='*60}")
        print(f"MODEL: {display}")
        print(f"{'='*60}")

        # Load model
        try:
            model = load_model(model_cfg["model_file"])
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue

        gcam = GradCAM(model)

        # --- Exercise 6.5: baseline test set ---
        df_base, img_dir_base = load_split("baseline")
        if df_base is None:
            print("  ERROR: baseline test split not found.")
            continue

        correct_s, wrong_s = run_ex65(model_cfg, model, gcam, df_base, img_dir_base)

        # --- Exercise 6.6: OOD conditions ---
        cond_results = run_ex66(model_cfg, model, gcam,
                                conditions_to_test=["fog", "night", "town01"])

        # Baseline accuracy (for comparison in 6.6c)
        base_acc, base_n = accuracy_summary(model, df_base, img_dir_base, label_col)
        print(f"\n  Baseline accuracy ({base_n} imgs): {base_acc:.3f}")

        all_results[name] = {
            "baseline_accuracy": base_acc,
            "ood": cond_results,
            "n_correct_found": len(correct_s),
            "n_wrong_found": len(wrong_s),
        }

    # ------------------------------------------------------------------ #
    #  SAVE NUMERIC SUMMARY
    # ------------------------------------------------------------------ #
    summary_path = os.path.join(OUT_DIR, "gradcam_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nSummary saved to: {summary_path}")

    # ------------------------------------------------------------------ #
    #  PRINT TEXT SUMMARY FOR REPORT
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("NUMERIC SUMMARY (for report write-up)")
    print("=" * 60)
    for m_name, res in all_results.items():
        print(f"\n{m_name.upper()}")
        print(f"  Baseline accuracy : {res['baseline_accuracy']:.3f}")
        for cond, cr in res.get("ood", {}).items():
            print(f"  {cond:<10} accuracy: {cr['accuracy']:.3f}  ({cr['n_total']} imgs)")

    print("\nAll outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()