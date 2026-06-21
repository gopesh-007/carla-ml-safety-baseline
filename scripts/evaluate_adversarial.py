#!/usr/bin/env python3
"""Evaluate FGSM robustness for CARLA binary perception models.

This script loads three ResNet18 binary classifiers, evaluates clean and FGSM
adversarial performance on the first validation samples, saves example attack
visualizations, and writes a recall-vs-epsilon plot plus a CSV summary.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


LABEL_COLUMNS = ("has_pedestrian", "has_traffic_light", "has_vehicle")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}
DEFAULT_EPSILONS = (0.01, 0.05, 0.1)


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    output_name: str
    label_column: str
    checkpoint_name: str


MODEL_SPECS = (
    ModelSpec(
        display_name="PEDESTRIAN MODEL",
        output_name="pedestrian",
        label_column="has_pedestrian",
        checkpoint_name="pedestrian_model.pth",
    ),
    ModelSpec(
        display_name="TRAFFIC LIGHT MODEL",
        output_name="traffic_light",
        label_column="has_traffic_light",
        checkpoint_name="traffic_light_model.pth",
    ),
    ModelSpec(
        display_name="VEHICLE MODEL",
        output_name="vehicle",
        label_column="has_vehicle",
        checkpoint_name="vehicle_model.pth",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FGSM adversarial robustness evaluation for CARLA models."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root. Defaults to the parent of this script's directory.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/validation/rgb-front"),
        help="Directory containing validation RGB front images.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=Path("data/validation/labels.csv"),
        help="CSV file containing validation labels.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing model checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/adversarial"),
        help="Directory where adversarial outputs are written.",
    )
    parser.add_argument(
        "--limit",
        default="100",
        help="Number of validation samples to evaluate, or 'all' for the full set.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("first", "random"),
        default="random",
        help="Use the first N samples or a reproducible random subset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --sample-mode=random.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square image size used for evaluation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device to use. 'auto' selects CUDA when available.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid classification threshold.",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=list(DEFAULT_EPSILONS),
        help="FGSM epsilon values to test.",
    )
    return parser.parse_args()


def parse_sample_limit(limit_arg: str) -> Optional[int]:
    text = str(limit_arg).strip().lower()
    if text == "all":
        return None

    try:
        limit = int(text)
    except ValueError as exc:
        raise ValueError("--limit must be a positive integer or 'all'.") from exc

    if limit <= 0:
        raise ValueError("--limit must be a positive integer or 'all'.")
    return limit


def resolve_project_root(project_root_arg: Optional[Path]) -> Path:
    if project_root_arg is not None:
        return project_root_arg.expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Path, project_root: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA was requested but is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def natural_sort_key(path: Path) -> List[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def list_image_files(image_dir: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=natural_sort_key,
    )


def looks_like_image_reference(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    if not text:
        return False
    suffix = Path(text).suffix.lower()
    return suffix in IMAGE_EXTENSIONS or "/" in text or "\\" in text


def infer_image_column(df: pd.DataFrame) -> Optional[str]:
    preferred_names = {
        "filename",
        "file_name",
        "image_name",
        "image_filename",
        "image_file",
        "image_path",
        "filepath",
        "file_path",
        "path",
        "rgb_front",
        "rgb-front",
    }

    label_names = set(LABEL_COLUMNS)
    normalized_columns = {
        column: str(column).strip().lower().replace(" ", "_") for column in df.columns
    }

    for column, normalized in normalized_columns.items():
        if column not in label_names and normalized in preferred_names:
            return column

    for column in df.columns:
        if column in label_names:
            continue
        sample_values = df[column].dropna().head(20)
        if sample_values.empty:
            continue
        image_ref_count = sum(looks_like_image_reference(value) for value in sample_values)
        if image_ref_count > 0:
            return column

    return None


def candidate_image_paths(
    value: object, image_dir: Path, project_root: Path
) -> List[Path]:
    text = str(value).strip()
    raw_path = Path(text)
    candidates: List[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                image_dir / raw_path,
                project_root / raw_path,
                image_dir / raw_path.name,
            ]
        )

    if not raw_path.suffix:
        for extension in sorted(IMAGE_EXTENSIONS):
            candidates.append(image_dir / f"{text}{extension}")

    deduped: List[Path] = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved not in seen:
            deduped.append(resolved)
            seen.add(resolved)
    return deduped


def resolve_image_paths_from_column(
    df: pd.DataFrame, image_column: str, image_dir: Path, project_root: Path
) -> List[Path]:
    resolved_paths: List[Path] = []
    missing_examples: List[str] = []

    for value in df[image_column]:
        candidates = candidate_image_paths(value, image_dir, project_root)
        existing = next((candidate for candidate in candidates if candidate.exists()), None)
        if existing is None:
            missing_examples.append(str(value))
            resolved_paths.append(candidates[0])
        else:
            resolved_paths.append(existing)

    if missing_examples:
        examples = ", ".join(missing_examples[:5])
        raise FileNotFoundError(
            f"Could not resolve {len(missing_examples)} image paths from column "
            f"'{image_column}'. Examples: {examples}"
        )

    return resolved_paths


def coerce_binary_labels(series: pd.Series, column_name: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        labels = pd.to_numeric(series, errors="coerce")
    else:
        normalized = series.astype(str).str.strip().str.lower()
        mapping = {
            "1": 1.0,
            "true": 1.0,
            "t": 1.0,
            "yes": 1.0,
            "y": 1.0,
            "0": 0.0,
            "false": 0.0,
            "f": 0.0,
            "no": 0.0,
            "n": 0.0,
        }
        labels = normalized.map(mapping)

    if labels.isna().any():
        bad_count = int(labels.isna().sum())
        raise ValueError(f"Column '{column_name}' contains {bad_count} non-binary labels.")

    invalid_values = sorted(set(labels[~labels.isin([0.0, 1.0])].tolist()))
    if invalid_values:
        raise ValueError(
            f"Column '{column_name}' must contain binary labels. "
            f"Unexpected values: {invalid_values[:5]}"
        )

    return labels.astype(float)


class CarlaBinaryDataset(Dataset):
    """CARLA validation images paired with a single binary target column."""

    def __init__(
        self,
        image_dir: Path,
        labels_csv: Path,
        label_column: str,
        transform: transforms.Compose,
        project_root: Path,
        limit: Optional[int],
        sample_mode: str,
        seed: int,
    ) -> None:
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not labels_csv.is_file():
            raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

        df = pd.read_csv(labels_csv)
        missing_columns = [column for column in LABEL_COLUMNS if column not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Labels CSV is missing required columns: {', '.join(missing_columns)}"
            )
        if label_column not in df.columns:
            raise ValueError(f"Target label column not found: {label_column}")

        if limit is None:
            selected_df = df.copy()
            print(f"[INFO] Using all {len(selected_df)} validation rows.")
        elif sample_mode == "random":
            sample_count = min(limit, len(df))
            selected_df = df.sample(n=sample_count, random_state=seed).sort_index()
            print(
                f"[INFO] Using {sample_count} randomly sampled validation rows "
                f"(seed={seed})."
            )
        else:
            selected_df = df.head(limit)
            print(f"[INFO] Using first {len(selected_df)} validation rows.")

        self.source_indices = selected_df.index.to_list()
        self.df = selected_df.reset_index(drop=True)
        self.df[label_column] = coerce_binary_labels(self.df[label_column], label_column)
        self.label_column = label_column
        self.transform = transform

        image_column = infer_image_column(self.df)
        if image_column is not None:
            self.image_paths = resolve_image_paths_from_column(
                self.df, image_column, image_dir, project_root
            )
            print(f"[INFO] Using image path column '{image_column}'.")
        else:
            image_files = list_image_files(image_dir)
            max_source_index = max(self.source_indices) if self.source_indices else -1
            if len(image_files) <= max_source_index:
                raise ValueError(
                    f"Found {len(image_files)} images in {image_dir}, but the selected "
                    f"labels reference image index {max_source_index}."
                )
            self.image_paths = [image_files[index] for index in self.source_indices]
            print("[INFO] No image column found. Pairing labels with sorted image files.")

        if not self.image_paths:
            raise ValueError(f"No validation images found in {image_dir}.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        label = torch.tensor(
            [float(self.df.loc[index, self.label_column])], dtype=torch.float32
        )
        return image_tensor, label, str(image_path)

    def first_positive_index(self) -> int:
        positives = self.df.index[self.df[self.label_column] == 1.0].tolist()
        return int(positives[0]) if positives else 0


def build_binary_resnet18() -> nn.Module:
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        # Compatibility with older torchvision releases.
        model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def torch_load_checkpoint(path: Path, device: torch.device) -> object:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
    except Exception as exc:
        print(
            "[WARN] weights_only checkpoint load failed. Retrying with "
            f"weights_only=False for trusted local file {path.name}. Error: {exc}"
        )
        return torch.load(path, map_location=device, weights_only=False)


def strip_state_dict_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    stripped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        stripped[new_key] = value
    return stripped


def extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, nn.Module):
                return value.state_dict()
            if isinstance(value, dict):
                return value

        if checkpoint and all(isinstance(key, str) for key in checkpoint.keys()):
            tensor_values = [value for value in checkpoint.values() if torch.is_tensor(value)]
            if tensor_values:
                return checkpoint  # type: ignore[return-value]

    raise ValueError(
        "Unsupported checkpoint format. Expected a state_dict, an nn.Module, or a "
        "dict containing state_dict/model_state_dict."
    )


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    model = build_binary_resnet18()
    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    state_dict = strip_state_dict_prefixes(extract_state_dict(checkpoint))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} did not match the expected ResNet18 "
            f"binary classifier. Missing keys: {list(missing)}. "
            f"Unexpected keys: {list(unexpected)}."
        )

    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    criterion: nn.Module,
) -> torch.Tensor:
    """Create FGSM images by ascending the BCEWithLogits loss gradient."""
    inputs = images.detach().clone().requires_grad_(True)
    targets = labels.float().view(-1, 1)

    logits = model(inputs).view(-1, 1)
    loss = criterion(logits, targets)
    model.zero_grad(set_to_none=True)

    if inputs.grad is not None:
        inputs.grad.zero_()
    loss.backward()

    if inputs.grad is None:
        raise RuntimeError("FGSM gradient was not computed for the input images.")

    adversarial = inputs + epsilon * inputs.grad.detach().sign()
    return adversarial.clamp(0.0, 1.0).detach()


def predict_logits(
    model: nn.Module,
    images: torch.Tensor,
    threshold: float,
) -> np.ndarray:
    logits = model(images).view(-1, 1)
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).to(torch.int64)
    return predictions.cpu().numpy().reshape(-1)


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    epsilon: float,
    threshold: float,
) -> Dict[str, float]:
    y_true: List[int] = []
    y_pred: List[int] = []

    for images, labels, _paths in dataloader:
        images = images.to(device)
        labels = labels.to(device).float().view(-1, 1)

        if epsilon > 0.0:
            eval_images = fgsm_attack(model, images, labels, epsilon, criterion)
        else:
            eval_images = images

        with torch.no_grad():
            predictions = predict_logits(model, eval_images, threshold)

        y_true.extend(labels.cpu().numpy().astype(int).reshape(-1).tolist())
        y_pred.extend(predictions.astype(int).tolist())

    return compute_metrics(y_true, y_pred)


def tensor_to_image_array(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return array


def save_attack_comparison(
    clean_image: torch.Tensor,
    adversarial_image: torch.Tensor,
    epsilon: float,
    output_path: Path,
    title: str,
) -> None:
    perturbation = adversarial_image.detach().cpu() - clean_image.detach().cpu()
    amplification = min(1.0 / max(2.0 * epsilon, 1e-8), 50.0)
    perturbation_display = (perturbation * amplification + 0.5).clamp(0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = (
        ("Clean", clean_image),
        (f"Perturbation x{amplification:.1f}", perturbation_display),
        ("Adversarial", adversarial_image),
    )

    for axis, (panel_title, panel_tensor) in zip(axes, panels):
        axis.imshow(tensor_to_image_array(panel_tensor))
        axis.set_title(panel_title)
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def epsilon_to_filename_suffix(epsilon: float) -> str:
    return f"{epsilon:g}".replace(".", "")


def save_example_for_epsilon(
    model: nn.Module,
    dataset: CarlaBinaryDataset,
    device: torch.device,
    criterion: nn.Module,
    epsilon: float,
    output_path: Path,
    model_name: str,
) -> None:
    index = dataset.first_positive_index()
    clean_image, label, image_path = dataset[index]
    clean_batch = clean_image.unsqueeze(0).to(device)
    label_batch = label.unsqueeze(0).to(device)
    adversarial_batch = fgsm_attack(model, clean_batch, label_batch, epsilon, criterion)
    adversarial_image = adversarial_batch.squeeze(0).cpu()

    title = (
        f"{model_name} | epsilon={epsilon:g} | label={int(label.item())} | "
        f"{Path(image_path).name}"
    )
    save_attack_comparison(clean_image, adversarial_image, epsilon, output_path, title)


def save_recall_plot(
    results_df: pd.DataFrame,
    output_path: Path,
    epsilons_for_plot: Iterable[float],
) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))

    for spec in MODEL_SPECS:
        model_results = results_df[results_df["model"] == spec.output_name].sort_values(
            "epsilon"
        )
        axis.plot(
            model_results["epsilon"],
            model_results["recall"],
            marker="o",
            linewidth=2,
            label=spec.display_name.replace(" MODEL", "").title(),
        )

    axis.set_xlabel("FGSM epsilon")
    axis.set_ylabel("Recall")
    axis.set_title("Recall vs FGSM Epsilon")
    axis.set_xticks(list(epsilons_for_plot))
    axis.set_ylim(0.0, 1.05)
    axis.grid(True, alpha=0.35)
    axis.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def print_metrics_block(
    epsilon: float,
    metrics: Dict[str, float],
    clean_metrics: Optional[Dict[str, float]] = None,
) -> None:
    label = "Clean performance" if epsilon == 0.0 else f"Epsilon: {epsilon:g}"
    print(label)
    print(f"Recall:   {metrics['recall']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1:       {metrics['f1']:.4f}")

    if clean_metrics is not None and epsilon > 0.0:
        print(
            "Delta vs clean: "
            f"Recall {metrics['recall'] - clean_metrics['recall']:+.4f}, "
            f"Accuracy {metrics['accuracy'] - clean_metrics['accuracy']:+.4f}, "
            f"F1 {metrics['f1'] - clean_metrics['f1']:+.4f}"
        )
    print()


def validate_inputs(
    image_dir: Path,
    labels_csv: Path,
    models_dir: Path,
    model_specs: Sequence[ModelSpec],
) -> None:
    missing: List[Path] = []
    if not image_dir.is_dir():
        missing.append(image_dir)
    if not labels_csv.is_file():
        missing.append(labels_csv)
    for spec in model_specs:
        checkpoint = models_dir / spec.checkpoint_name
        if not checkpoint.is_file():
            missing.append(checkpoint)

    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required input paths are missing:\n{missing_text}")


def main() -> int:
    args = parse_args()
    project_root = resolve_project_root(args.project_root)

    image_dir = resolve_path(args.image_dir, project_root)
    labels_csv = resolve_path(args.labels_csv, project_root)
    models_dir = resolve_path(args.models_dir, project_root)
    output_dir = resolve_path(args.output_dir, project_root)
    examples_dir = output_dir / "examples"
    plots_dir = output_dir / "plots"
    sample_limit = parse_sample_limit(args.limit)

    device = select_device(args.device)
    epsilons = tuple(float(epsilon) for epsilon in args.epsilons)
    if any(epsilon <= 0.0 for epsilon in epsilons):
        raise ValueError("FGSM epsilon values must be positive.")
    eval_epsilons = (0.0,) + epsilons

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Device: {device}")
    if sample_limit is None:
        print("[INFO] Evaluating all validation samples.")
    elif args.sample_mode == "random":
        print(
            f"[INFO] Evaluating {sample_limit} randomly sampled validation samples "
            f"(seed={args.seed})."
        )
    else:
        print(f"[INFO] Evaluating first {sample_limit} validation samples.")
    print(f"[INFO] FGSM epsilons: {', '.join(f'{epsilon:g}' for epsilon in epsilons)}")

    validate_inputs(image_dir, labels_csv, models_dir, MODEL_SPECS)
    examples_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )
    criterion = nn.BCEWithLogitsLoss()
    all_results: List[Dict[str, Union[float, str]]] = []

    for spec in MODEL_SPECS:
        print("=" * 34)
        print(spec.display_name)
        print("=" * 34)

        checkpoint_path = models_dir / spec.checkpoint_name
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        model = load_model(checkpoint_path, device)

        dataset = CarlaBinaryDataset(
            image_dir=image_dir,
            labels_csv=labels_csv,
            label_column=spec.label_column,
            transform=transform,
            project_root=project_root,
            limit=sample_limit,
            sample_mode=args.sample_mode,
            seed=args.seed,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        print(f"[INFO] Loaded {len(dataset)} samples for {spec.output_name}.")

        clean_metrics: Optional[Dict[str, float]] = None
        for epsilon in eval_epsilons:
            print(f"[INFO] Evaluating {spec.output_name} at epsilon={epsilon:g}...")
            metrics = evaluate_model(
                model=model,
                dataloader=dataloader,
                device=device,
                criterion=criterion,
                epsilon=epsilon,
                threshold=args.threshold,
            )

            if epsilon == 0.0:
                clean_metrics = metrics
            if clean_metrics is None:
                raise RuntimeError("Clean metrics must be evaluated before adversarial metrics.")

            all_results.append(
                {
                    "model": spec.output_name,
                    "epsilon": epsilon,
                    "accuracy": metrics["accuracy"],
                    "recall": metrics["recall"],
                    "recall_drop": clean_metrics["recall"] - metrics["recall"],
                    "f1": metrics["f1"],
                }
            )
            print_metrics_block(epsilon, metrics, clean_metrics)

        for epsilon in epsilons:
            suffix = epsilon_to_filename_suffix(epsilon)
            example_path = examples_dir / f"{spec.output_name}_eps_{suffix}.png"
            print(f"[INFO] Saving FGSM example: {example_path}")
            save_example_for_epsilon(
                model=model,
                dataset=dataset,
                device=device,
                criterion=criterion,
                epsilon=epsilon,
                output_path=example_path,
                model_name=spec.display_name,
            )

    results_df = pd.DataFrame(
        all_results,
        columns=["model", "epsilon", "accuracy", "recall", "recall_drop", "f1"],
    )
    results_csv = output_dir / "adversarial_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"[INFO] Saved results CSV: {results_csv}")

    plot_path = plots_dir / "recall_vs_epsilon.png"
    save_recall_plot(results_df, plot_path, eval_epsilons)
    print(f"[INFO] Saved recall plot: {plot_path}")
    print("[INFO] Adversarial evaluation complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
