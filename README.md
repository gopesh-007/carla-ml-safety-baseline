# 🚗 CARLA ML Safety Baseline Models

> **Course:** Introduction to Machine Learning Safety  
> **Project:** Binary perception classifiers for autonomous driving safety analysis  
> **Author:** Gopeshkumar Harsukh Rabadiya
> **Matrikel-Nr.:** 261637

---

## 📋 Overview

This project trains and evaluates **three independent binary classifiers** for safety-critical perception tasks in autonomous driving, using the CARLA-based dataset provided by the professor.

Each model takes a single **front-facing RGB camera image** as input and outputs a binary prediction:

| Model | Task | Label |
|---|---|---|
| Pedestrian Detector | Is a pedestrian present? | `has_pedestrian` |
| Traffic Light Detector | Is a traffic light present? | `has_traffic_light` |
| Vehicle Detector | Is a vehicle present? | `has_vehicle` |

The models' outputs feed directly into a vehicle's **autopilot decision logic**, making recall and robustness the primary safety metrics.

This project also includes a complete robustness and safety evaluation pipeline covering:

* ODD robustness evaluation under fog, night, and domain-shift conditions,
* temperature scaling calibration analysis,
* backdoor attack evaluation,
* Grad-CAM explainability analysis,
* confidence-based OOD detection using Maximum Softmax Probability (MSP),
* feature-based OOD detection using k-Nearest Neighbors (k-NN),
* and STPA-based safety analysis extensions for distributional robustness failures.
* untargeted FGSM adversarial robustness evaluation for all three classifiers,
* adversarial recall-drop analysis at ε ∈ {0.01, 0.05, 0.1},
* and STPA safety-analysis extensions for adversarial perception failures.

Together, these experiments form evidence toward a structured machine-learning safety case for autonomous driving perception systems.

---

## ⚡ Quick Results

### Baseline Performance (Standard Test Set)

| Model | Accuracy | Precision | Recall | F1 Score | Status |
|---|---|---|---|---|---|
| Traffic Light | 0.9206 | 0.9527 | 0.9358 | 0.9442 | ✅ Excellent |
| Vehicle | 0.7875 | 0.8617 | 0.8537 | 0.8577 | ✅ Good |
| Pedestrian | 0.7058 | 0.1504 | **0.1076** | 0.1255 | ❌ Critical — needs fix |

### Robustness Under Adverse Conditions

| Model | Condition | Recall | F1 | Notes |
|---|---|---|---|---|
| Traffic Light | Fog | 0.000 | 0.000 | Complete collapse |
| Traffic Light | Night | 0.000 | 0.000 | Complete collapse |
| Traffic Light | Town-01 | 0.283 | 0.416 | Significant degradation |
| Vehicle | Fog | 1.000 | 0.872 | Recall retained, but nearly every frame is predicted positive |
| Vehicle | Night | 0.372 | 0.512 | Major false-negative failure |
| Vehicle | Town-01 | 0.752 | 0.719 | Moderate domain-shift degradation |
| Pedestrian | Fog | 0.315 | 0.269 | Partial degradation |
| Pedestrian | Night | 0.000 | 0.000 | Complete collapse |
| Pedestrian | Town-01 | 0.277 | 0.180 | Below safe threshold |

> ⚠️ **Safety note:** None of the three camera-only models is validated for all adverse conditions. The vehicle detector is especially unsafe at night, while its fog result is operationally unusable because it produces false positives on almost every frame.

#### Vehicle Robustness Detail

All vehicle ODD results below use the same trained checkpoint and a sigmoid decision threshold of 0.5. Each condition contains 3,600 labelled frames.

| Condition | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fog | 0.7739 | 0.7738 | 1.0000 | 0.8725 | 2,785 | 814 | 1 | 0 |
| Night | 0.4517 | 0.8211 | 0.3724 | 0.5124 | 1,037 | 226 | 589 | 1,748 |
| Town-01 | 0.6486 | 0.6885 | 0.7516 | 0.7187 | 1,616 | 731 | 719 | 534 |

The fog result illustrates why recall and F1 cannot be considered in isolation: the classifier predicts **vehicle present for 3,599 of 3,600 frames**, preserving recall while causing 814 unnecessary positive detections. At night, it misses 1,748 of 2,785 vehicles, making the model unsafe for night driving. Town-01 also reduces F1 from 0.8577 on the standard test set to 0.7187, confirming sensitivity to map/domain shift.

### ODD Test Coverage

The supplied labelled test splits cover four observed scenarios, each with 3,600 frames: dry daylight in the source town, fog in the source town, rainy night in the source town, and dry daylight in Town-01. For the three recorded context factors — weather (`dry`, `fog`, `rain`), lighting (`day`, `night`), and town (`source_town`, `town_01`) — the reproducible k-projection coverage is:

| k | Observed / possible combinations | Coverage | Interpretation |
|---|---:|---:|---|
| 1 | 7 / 7 | 100.0% | Every individual factor value is represented. |
| 2 | 10 / 16 | 62.5% | Several weather–lighting and weather–town pairings are absent. |
| 3 | 4 / 12 | 33.3% | Only four complete weather–lighting–town scenarios were tested. |

This is not a claim of complete ODD coverage: weather, lighting, and town are strongly coupled in the supplied scenarios. In particular, there is no night-in-Town-01, fog-in-Town-01, or rainy-day evaluation. Performance claims therefore remain limited to the exact tested scenario, while fog, night, and Town-01 are treated as robustness/OOD evidence rather than validated operating conditions.

### Temperature Scaling (Exercise 5.4)

| Temperature T | Accuracy | Recall | Below Safety θ=0.6 | Safety Behaviour |
|---|---|---|---|---|
| 0.5 | 0.7058 | 0.1076 | 88.3% | ⚠️ Overconfident on positives |
| 1.0 | 0.7058 | 0.1076 | 89.5% | Baseline |
| 2.0 | 0.7058 | 0.1076 | 91.9% | ✅ Most conservative |

> Accuracy is identical across all temperatures — proving accuracy alone is insufficient to verify safety constraints. Calibration must also be measured.

### Backdoor Attack (Exercise 5.5)

| Metric | Value | Interpretation |
|---|---|---|
| Poison rate | 10% of pedestrian-positive training images | 171 out of 1,718 samples |
| Trigger | 10×10 red square at pixel (5,5) | Invisible at a glance |
| Clean Recall | 0.3286 | Model appears normal on standard tests |
| **Attack Success Rate (ASR)** | **1.0000 (100%)** | Every triggered image misclassified |

> 🔴 **Critical finding:** A model with ASR=100% would pass all standard safety evaluations while being completely blind to pedestrians whenever the trigger is present.

## Explainability & Grad-CAM (Exercise 6)

Grad-CAM explainability analysis was applied to all three CARLA perception models under both baseline and out-of-distribution (OOD) conditions.

### Key Explainability Findings

| Model | Baseline Attention Quality | OOD Behaviour | Safety Insight |
|---|---|---|---|
| Traffic Light | Highly object-focused | Complete attention collapse under fog/night | Relies heavily on lighting statistics |
| Vehicle | Mostly object-centred | Moderate degradation | Uses some contextual road cues |
| Pedestrian | Weak localization even at baseline | Severe attention drift | Strong evidence of shortcut learning |

### Critical Observations

- Night-condition Grad-CAM maps showed all models attending to bright artifacts instead of semantic objects.
- Pedestrian false negatives frequently activated on sky and road textures instead of pedestrian silhouettes.
- Explanation quality degraded together with robustness under OOD conditions.
- Grad-CAM exposed spurious correlations invisible to aggregate accuracy metrics.

> ⚠️ Explainability analysis confirmed that the models rely heavily on environmental context and training-distribution-specific visual statistics rather than robust object-centric representations.

---

## 🌍 Out-of-Distribution (OOD) Detection & Robustness Analysis (Exercise 7)

Out-of-distribution (OOD) detection experiments were conducted to evaluate whether the perception models can recognize environmental conditions outside their training distribution.

Two OOD detection approaches were implemented:
1. **Maximum Softmax Probability (MSP)** baseline
2. **Feature-based k-Nearest Neighbors (k-NN)** detector

### MSP OOD Detection Results

| Condition | AUROC | Interpretation |
|---|---|---|
| Fog | 0.8104 | Good separation |
| Night | 0.0000 | Complete failure |
| Town01 | 0.6539 | Moderate separation |

### Feature-Based k-NN OOD Detection Results

Feature-based k-NN detection was evaluated for all three classifiers using 1,000 deterministic validation images and 1,000 images from each OOD condition. The suggested safety-case threshold is AUROC ≥ 0.90.

| Model | Fog AUROC | Night AUROC | Town-01 AUROC | Verdict |
|---|---:|---:|---:|---|
| Pedestrian | 0.9844 | 1.0000 | 0.8956 | Partial — Town-01 is below threshold |
| Traffic Light | 0.9698 | 1.0000 | 0.9556 | Met |
| Vehicle | 0.9940 | 0.9971 | 0.8274 | Partial — Town-01 is below threshold |

The MSP baseline above was evaluated for the pedestrian detector only. It is retained as a confidence-based baseline; the final safety-case evidence uses feature-based k-NN because it gives substantially stronger separation for environmental OOD inputs.

### Key Findings

- MSP confidence scoring failed catastrophically under nighttime conditions because the model remained highly confident despite severe environmental shift.
- Feature-based k-NN detection successfully identified nighttime samples as strongly out-of-distribution.
- Deep feature embeddings captured semantic environmental shifts more reliably than classifier confidence alone.
- Town-01 represented a weaker semantic domain shift compared to fog and nighttime appearance degradation.
- OOD failures were strongly correlated with degraded Grad-CAM explainability quality and semantic attention collapse.

> ⚠️ Safety implication: High prediction confidence does not guarantee reliable perception under environmental distribution shift. Feature-space monitoring provides significantly stronger OOD detection capability than confidence-based approaches alone.


## ⚔️ Adversarial Robustness with FGSM (Exercise 8)

An untargeted Fast Gradient Sign Method (FGSM) attack was applied to all three binary classifiers.

The adversarial image is generated using:

```text
x_adv = x + ε · sign(∇x L(y, f(x)))
```
The implementation uses:
BCEWithLogitsLoss
gradients with respect to the input image
pixel clamping to [0, 1]
ε values of 0.01, 0.05, and 0.1
a reproducible random sample of 100 validation images using seed 42

| Model | Clean Recall | Recall @ 0.01 | Drop @ 0.01 | Recall @ 0.05 | Drop @ 0.05 | Recall @ 0.1 | Drop @ 0.1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pedestrian | 0.1923 | 0.0000 | 0.1923 | 0.0000 | 0.1923 | 0.0000 | 0.1923 |
| Traffic Light | 0.8987 | 0.0886 | 0.8101 | 0.0000 | 0.8987 | 0.1519 | 0.7468 |
| Vehicle | 0.8596 | 0.5614 | 0.2982 | 0.0351 | 0.8246 | 0.3860 | 0.4737 |

### Key Adversarial Findings

- The pedestrian model's recall dropped to **0.0000** for every tested epsilon.
- The traffic-light model lost most of its recall at `epsilon = 0.01`.
- At `epsilon = 0.05`, traffic-light recall collapsed completely.
- Vehicle recall dropped from **0.8596** to **0.0351** at `epsilon = 0.05`.
- Perturbations were mostly imperceptible at `epsilon = 0.01`.
- Visible image noise appeared around `epsilon = 0.05`.
- At `epsilon = 0.1`, perturbations were clearly visible.
- Recall was not strictly monotonic. Clipping and decision-boundary effects can make a larger one-step FGSM perturbation less effective for some samples.

> ⚠️ Safety implication: Small adversarial perturbations can produce safety-critical false negatives while leaving the image visually similar to the clean input. Standard clean-test performance is therefore insufficient evidence of safe perception.

## 🎯 Uncertainty, Calibration & Cost-Sensitive Decisions (Exercise 9)

Exercise 9 evaluates whether the perception models' confidence scores can be trusted for safety-critical downstream decisions.

The uncertainty pipeline measures:

- Expected Calibration Error (ECE)
- reliability diagrams
- validation-set temperature scaling
- calibrated vs. uncalibrated confidence behaviour
- cost-sensitive pedestrian braking decisions

The implementation is in:
```bash
scripts/evaluate_uncertainty.py
```

Generated outputs are stored in:
```bash
outputs/uncertainty/
├── calibration_results.csv
├── pedestrian_cost_results.csv
├── reliability_bins.csv
├── reliability_diagrams.png
└── temperature_search.csv
```

Calibration Results:
| Model | Best T | ECE Before | ECE After | Test Accuracy | Pattern Before | Pattern After |
|---|---:|---:|---:|---:|---|---|
| Pedestrian | 3.0 | 0.1477 | 0.0385 | 0.7058 | Overconfident | Slightly underconfident |
| Traffic Light | 1.2 | 0.0349 | 0.0246 | 0.9206 | Overconfident | Overconfident |
| Vehicle | 0.9 | 0.0286 | 0.0177 | 0.7875 | Underconfident | Underconfident |

Key Calibration Findings:
- The pedestrian model was strongly overconfident before calibration.
- Temperature scaling reduced pedestrian ECE from 0.1477 to 0.0385.
- Traffic-light and vehicle models were already better calibrated, but still improved after temperature scaling.
- Accuracy did not change after temperature scaling because temperature scaling only rescales confidence, not the final predicted class.
- The pedestrian model selected T = 3.0, which is the upper end of the tested grid, suggesting strong overconfidence.

Cost-Sensitive Pedestrian Decision:
For pedestrian braking, the assumed costs were:
```text
False negative cost: C_FN = 100
False positive cost: C_FP = 1
```
The cost-optimal braking threshold is:
```text
tau* = C_FP / (C_FN + C_FP) = 1 / 101 ≈ 0.0099
```

| Calibration | Threshold | False Negatives | False Positives | Total Loss |
|---|---:|---:|---:|---:|
| Uncalibrated | 0.5 | 630 | 429 | 63,429 |
| Uncalibrated | 0.0099 | 48 | 2,629 | 7,429 |
| Calibrated | 0.5 | 630 | 429 | 63,429 |
| Calibrated | 0.0099 | 0 | 2,894 | 2,894 |

Safety Interpretation:

The calibrated cost-optimal threshold produced the lowest total loss and eliminated pedestrian false negatives on the test set. However, it also caused many false positives, meaning the vehicle would brake extremely often.

This shows the central safety trade-off:
- a standard 0.5 threshold is unsafe because it misses too many pedestrians,
- a cost-sensitive threshold greatly reduces safety-critical misses,
- calibration is necessary before using probabilities for downstream decisions,
- but a very low threshold may harm availability and comfort.

> ⚠️ Safety implication: calibrated probabilities are useful for risk-aware planning, but calibration alone is not enough. The planner still needs fallback logic, OOD detection, temporal consistency checks, and additional sensor redundancy.

## 🗂️ Repository Structure

```text
carla_baseline_project/
│
├── scripts/
│   ├── train_pedestrian.py                    # Train pedestrian classifier
│   ├── train_traffic_light.py                 # Train traffic-light classifier
│   ├── train_vehicle.py                       # Train vehicle classifier
│   │
│   ├── evaluate_pedestrian.py                 # Standard pedestrian evaluation
│   ├── evaluate_traffic_light.py              # Standard traffic-light evaluation
│   ├── evaluate_vehicle.py                    # Standard vehicle evaluation
│   │
│   ├── evaluate_pedestrian_fog.py             # Robustness: fog
│   ├── evaluate_pedestrian_night.py           # Robustness: night
│   ├── evaluate_pedestrian_town.py            # Robustness: Town-01
│   ├── evaluate_traffic_light_fog.py          # Robustness: fog
│   ├── evaluate_traffic_light_night.py        # Robustness: night
│   ├── evaluate_traffic_light_town.py         # Robustness: Town-01
│   ├── evaluate_vehicle_robustness.py          # Shared vehicle ODD evaluator
│   ├── evaluate_vehicle_fog.py                 # Robustness: fog
│   ├── evaluate_vehicle_night.py               # Robustness: night
│   ├── evaluate_vehicle_town.py                # Robustness: Town-01
│   │
│   ├── evaluate_temperature_scaling.py        # Ex 5.4: temperature scaling
│   ├── plot_temperature_distribution.py       # Ex 5.4: probability plot
│   ├── train_pedestrian_backdoor.py           # Ex 5.5: backdoor training
│   ├── evaluate_backdoor.py                   # Ex 5.5: clean recall and ASR
│   │
│   ├── gradcam_analysis.py                    # Ex 6: Grad-CAM analysis
│   │
│   ├── evaluate_ood_msp.py                    # Ex 7: MSP OOD detection
│   ├── evaluate_ood_knn.py                    # Ex 7: k-NN OOD detection
│   ├── evaluate_ood_knn_all_models.py         # Ex 7: k-NN OOD detection for all models
│   ├── evaluate_odd_coverage.py               # ODD k-projection coverage
│   │
│   ├── evaluate_adversarial.py                # Ex 8: FGSM robustness evaluation
│   │
│   └── evaluate_uncertainty.py                # Ex 9: ECE, temperature scaling, cost-sensitive decisions
│   
├── notebooks/
│   ├── dataset_exploration.ipynb              # Dataset exploration
│   └── evaluate_adversarial_walkthrough.ipynb # Ex 8 walkthrough
│                                               
├── data/                                      # Dataset not tracked by Git
│   ├── train/
│   ├── validation/
│   │   ├── rgb-front/
│   │   └── labels.csv
│   ├── test/
│   ├── test-fog/
│   ├── test-night/
│   └── test-town-01/
│
├── models/                                    # Model weights not tracked
│   ├── pedestrian_model.pth
│   ├── traffic_light_model.pth
│   ├── vehicle_model.pth
│   └── pedestrian_model_backdoor.pth          # Ex 5.5 backdoored model
│
├── outputs/
│   ├── temperature_distribution.png           # Ex 5.4 plot
│   │
│   ├── ood/
│   │   ├── knn_all_models_results.csv         # All-model k-NN AUROC summary
│   │   └── plots/
│   │       ├── msp_histogram.png
│   │       └── knn_histogram.png
│   │
│   ├── odd/
│   │   ├── odd_test_scenarios.csv             # Context factors per test split
│   │   └── odd_k_projection_coverage.csv      # 1-, 2-, and 3-way coverage
│   │
│   ├── explainability/
│   │   ├── pedestrian/
│   │   ├── traffic_light/
│   │   ├── vehicle/
│   │   ├── baseline/
│   │   ├── fog/
│   │   ├── night/
│   │   └── town01/
│   │
│   ├── adversarial/
│   │    ├── adversarial_results.csv            # Accuracy, recall, drop and F1
│   │    │
│   │    ├── plots/
│   │    │   └── recall_vs_epsilon.png
│   │    │
│   │    └── examples/
│   │        ├── pedestrian_eps_001.png
│   │        ├── pedestrian_eps_005.png
│   │        ├── pedestrian_eps_01.png
│   │        ├── traffic_light_eps_001.png
│   │        ├── traffic_light_eps_005.png
│   │        ├── traffic_light_eps_01.png
│   │        ├── vehicle_eps_001.png
│   │        ├── vehicle_eps_005.png
│   │        └── vehicle_eps_01.png
│   │
│   └── uncertainty/
│        ├── calibration_results.csv            # ECE and temperature scaling summary
│        ├── pedestrian_cost_results.csv        # Cost-sensitive braking results
│        ├── reliability_bins.csv               # Reliability-bin statistics
│        ├── reliability_diagrams.png           # Calibration plots
│        └── temperature_search.csv             # Validation NLL search over T
│
├── report/
│   └── CARLA_ML_Safety_Report.pdf             # Complete safety report
│
├── requirements.txt                           # Python dependencies
├── README.md                                  # Project documentation
└── .gitignore                                 # Ignored data, models and cache files
```

---

## 🧠 Model Architecture

All three classifiers use the same basic architecture and were trained independently for their respective binary labels:

- **Backbone:** ResNet-18 initialized without ImageNet-pretrained weights (`weights=None`)
- **Output head:** One linear output logit; sigmoid is applied during evaluation
- **Loss function:** `BCEWithLogitsLoss` with a task-specific `pos_weight`
- **Optimizer:** Adam (`lr=0.001`)
- **Epochs:** 5
- **Batch size:** 32
- **Training device used:** CPU; the scripts automatically use CUDA when it is available

The pedestrian and traffic-light models use class weights calculated from their respective training-label distributions. The evaluated vehicle checkpoint was trained with `pos_weight = 5482/1718`, which unintentionally reused the pedestrian class ratio. This historical configuration is retained for reproducibility and is treated as a limitation in the safety analysis.

### Training Loss (5 Epochs)

| Model | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 |
|---|---|---|---|---|---|
| Pedestrian | 1.0457 | 0.9702 | 0.9159 | 0.8870 | 0.8415 |
| Traffic Light | 0.1805 | 0.1006 | 0.0681 | 0.0605 | 0.0499 |
| Vehicle | 0.7441 | 0.6360 | 0.5533 | 0.4938 | 0.4618 |

> The traffic-light training loss decreased consistently and reached 0.0499 after five epochs. The pedestrian loss remained considerably higher at 0.8415, which is consistent with a more difficult and imbalanced classification task. These training losses are descriptive only; the safety conclusions are based on held-out recall, calibration, robustness, and OOD evaluation results.

---

## 📊 Dataset

The CARLA dataset contains front-facing RGB camera images from simulated urban environments.

> **Note:** The dataset is not included in this repository — it was provided by course staff.

### Splits

| Split | Images |
|---|---|
| Train | 7,200 |
| Validation | 3,600 |
| Test (standard) | 3,600 |
| Test-Fog | ODD variant |
| Test-Night | ODD variant |
| Test-Town-01 | ODD variant |

### Class Distribution (Training Set)

| Label | Positive | Negative | Positive Rate |
|---|---|---|---|
| `has_pedestrian` | 1,718 | 5,482 | **23.9% — minority class ⚠️** |
| `has_traffic_light` | 5,276 | 1,924 | 73.3% |
| `has_vehicle` | 5,458 | 1,742 | 75.8% |

> The severe pedestrian class imbalance (only 24% positive) is the primary cause of the pedestrian model's low recall.

---

## 🔧 Setup & Installation

### Prerequisites

- Python 3.10 or newer
- pip

### Install dependencies

```bash
git clone https://github.com/gopesh-007/carla-ml-safety-baseline.git carla_baseline_project
cd carla_baseline_project

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Download the evaluated model checkpoints

The exact checkpoints used to produce the reported results are available in the
[GitHub model-checkpoint release](https://github.com/gopesh-007/carla-ml-safety-baseline/releases/tag/v1.0-model-checkpoints).

From the project root, download them into the `models/` directory:

```bash
mkdir -p models

curl -L -o models/pedestrian_model.pth \
  https://github.com/gopesh-007/carla-ml-safety-baseline/releases/download/v1.0-model-checkpoints/pedestrian_model.pth

curl -L -o models/traffic_light_model.pth \
  https://github.com/gopesh-007/carla-ml-safety-baseline/releases/download/v1.0-model-checkpoints/traffic_light_model.pth

curl -L -o models/vehicle_model.pth \
  https://github.com/gopesh-007/carla-ml-safety-baseline/releases/download/v1.0-model-checkpoints/vehicle_model.pth

curl -L -o models/pedestrian_model_backdoor.pth \
  https://github.com/gopesh-007/carla-ml-safety-baseline/releases/download/v1.0-model-checkpoints/pedestrian_model_backdoor.pth
```

The course dataset is not included because it was supplied separately by the professor. After placing the dataset under `data/`, the downloaded checkpoints can be used to reproduce the evaluation results without retraining.

### Requirements

```bash
torch
torchvision
pandas
pyarrow
numpy
matplotlib
scikit-learn
jupyter
pillow
grad-cam
opencv-python
```

---

## 🚀 Usage

> **Important:** All scripts must be run from inside the `scripts/` directory.
> ```bash
> cd scripts
> ```

### Train all three baseline models

```bash
python train_pedestrian.py
python train_traffic_light.py
python train_vehicle.py
```

### Evaluate on standard test set

```bash
python evaluate_pedestrian.py
python evaluate_traffic_light.py
python evaluate_vehicle.py
```

### Evaluate robustness (ODD conditions)

```bash
# Fog
python evaluate_pedestrian_fog.py
python evaluate_traffic_light_fog.py
python evaluate_vehicle_fog.py

# Night
python evaluate_pedestrian_night.py
python evaluate_traffic_light_night.py
python evaluate_vehicle_night.py

# Domain shift (Town-01)
python evaluate_pedestrian_town.py
python evaluate_traffic_light_town.py
python evaluate_vehicle_town.py
```

### Measure ODD test coverage

```bash
python evaluate_odd_coverage.py
```

This reads the recorded CARLA weather metadata and writes the scenario table and k-projection evidence to `outputs/odd/`.

### Exercise 5.4 — Temperature Scaling

```bash
python evaluate_temperature_scaling.py
python plot_temperature_distribution.py
```

### Exercise 5.5 — Backdoor Attack

```bash
# Step 1: train poisoned model (saves pedestrian_model_backdoor.pth)
python train_pedestrian_backdoor.py

# Step 2: evaluate clean recall + attack success rate
python evaluate_backdoor.py
```

> **Expected output:** Each script prints `Accuracy`, `Precision`, `Recall`, and `F1 Score`. Backdoor evaluation additionally reports `Attack Success Rate (ASR)`.

### Exercise 6 — Explainability & Grad-CAM

```bash
# Generate Grad-CAM
python gradcam_analysis.py

```

Generated images are stored in:
```bash
outputs/explainability/  
```

### Generated Outputs

The scripts produce:
- Correctly classified Grad-CAM overlays
- Misclassification explainability analysis
- OOD condition explainability (fog/night/Town-01)
- Heatmap visualizations for all three classifiers

---

### Exercise 7 — OOD Detection & Distribution Shift Evaluation

#### MSP Baseline OOD Detection

```bash
python evaluate_ood_msp.py
```

Generates:
- MSP confidence histograms
- AUROC evaluation for fog/night/Town01 conditions

Saved outputs:

```bash
outputs/ood/plots/msp_histogram.png
```

#### Feature-Based k-NN OOD Detection

```bash
python evaluate_ood_knn.py
```

Generates:
- Deep feature-space k-NN distance histograms
- Feature-based AUROC evaluation

Saved outputs:

```bash
outputs/ood/plots/knn_histogram.png
```

#### All-Model Feature-Based k-NN OOD Detection

```bash
python evaluate_ood_knn_all_models.py
```

This evaluates the pedestrian, traffic-light, and vehicle checkpoints on deterministic validation, fog, night, and Town-01 image subsets. It writes the AUROC evidence table to:

```bash
outputs/ood/knn_all_models_results.csv
```

### Exercise 8 — FGSM Adversarial Robustness

Evaluate a reproducible random sample of 100 validation images:

```bash
python evaluate_adversarial.py \
  --limit 100 \
  --sample-mode random \
  --seed 42
```
Generated outputs are stored in:
```bash
outputs/adversarial/
├── adversarial_results.csv
├── examples/
└── plots/recall_vs_epsilon.png
```

The CSV contains:
model, epsilon, accuracy, recall, recall_drop, f1

### Exercise 9 — Uncertainty Calibration and Cost-Sensitive Decisions

```bash
python evaluate_uncertainty.py
```

This script performs:
- ECE computation for all three classifiers
- reliability diagram generation
- temperature search on the validation set
- calibrated test-set evaluation
- pedestrian cost-sensitive decision analysis

Generated outputs:
```bash
outputs/uncertainty/
```

---

## 🛡️ Safety Analysis

### Why three separate models?

Training three independent classifiers (rather than one multi-label model) provides:

1. **Fault isolation** — a failure in one model does not affect the others
2. **Independent validation** — each model can be tested and certified to its own safety threshold
3. **Independent deployment control** — a single model can be replaced or disabled without rebuilding the system
4. **Clearer accountability** — single-function models are easier to trace in post-incident analysis

### ODD Gaps Identified

The models were trained **exclusively on clear-weather daytime data**. No safety claims can be made for:

| Condition | Risk Level | Evidence |
|---|---|---|
| Night driving | 🔴 Critical | Pedestrian & traffic-light recall = 0.0; vehicle recall = 0.372 |
| Fog | 🔴 Critical | Pedestrian F1 = 0.269; traffic-light recall = 0.0; vehicle predicts positive in 3,599/3,600 frames |
| New towns / maps | 🟠 High | Traffic-light F1: 0.944 → 0.416; vehicle F1: 0.858 → 0.719 |
| Rain, snow, dusk | 🟠 High | Not evaluated — assumed unsafe |

### Safety Constraints

| ID | Constraint |
|---|---|
| SC-1 | Pedestrian model must achieve recall ≥ 0.95 before deployment |
| SC-2 | System must not operate at night or in fog without validated perception models and a safe fallback |
| SC-3 | Perception models must be re-validated when the operational map changes |
| SC-4 | Camera-only perception is insufficient — LIDAR/RADAR redundancy required |
| SC-5 | Class imbalance must be addressed before retraining |
| SC-6 | Vehicle detections must be temporally confirmed before triggering availability-critical braking or stopping actions |

### Temperature Scaling & Safety Constraints

Temperature scaling (Ex 5.4) revealed that the safety constraint *"slow down if confidence < θ=0.6"* triggers for 88–92% of all images regardless of T, because the model is fundamentally uncertain. This confirms that **accuracy is not sufficient** to verify safety constraints — calibration must be measured independently.

T=0.5 is the most dangerous setting: it makes the model appear more confident on the small fraction of high-confidence predictions, potentially suppressing the safety speed-reduction rule on inputs the model is actually wrong about.

### Backdoor Vulnerability

The backdoor experiment (Ex 5.5) demonstrated that poisoning just **171 training samples (2.4% of training data)** is sufficient to install a fully effective backdoor with ASR=100%, while the model's clean recall actually increased slightly (0.33 vs baseline 0.11). This means:

- The backdoored model would **pass all standard safety evaluations**
- It would only fail when the specific trigger (red square) is present
- Standard metrics like accuracy, recall, and F1 are **insufficient to detect backdoor attacks**

### Explainability as a Safety Diagnostic Tool

Grad-CAM explainability analysis revealed that several model failures were caused by shortcut learning and spurious feature reliance rather than robust semantic object understanding.

Key findings include:

- The pedestrian model frequently attended to sky regions and road textures instead of pedestrian silhouettes.
- Under fog and night conditions, attention maps became diffuse and lost object-centred focus.
- Traffic light detection at night collapsed into brightness-based attention rather than signal recognition.
- Explanation quality degraded together with robustness under OOD conditions.

This confirms that:
- Accuracy alone is insufficient for safety validation
- Explainability can reveal hidden failure mechanisms
- Attention drift can serve as an indicator of OOD degradation
- Grad-CAM complements robustness testing and calibration analysis within the overall safety case

### OOD Detection as a Safety Monitoring Mechanism

OOD evaluation revealed that the perception models can silently fail under environmental distribution shifts while remaining highly confident.

The most critical failure occurred under nighttime conditions:
- MSP confidence-based OOD detection completely failed (AUROC = 0.0000)
- Grad-CAM explanations degraded severely
- The pedestrian detector remained overconfident despite semantic failure

Feature-based k-NN detection substantially improved robustness monitoring. For the pedestrian detector, nighttime AUROC improved from 0.0000 to 1.0000, fog AUROC from 0.8104 to 0.9844, and Town-01 AUROC from 0.6539 to 0.8956. The all-model evaluation additionally confirmed that traffic-light k-NN detection passes the 0.90 AUROC threshold in fog, night, and Town-01, while pedestrian and vehicle Town-01 detection remain below the threshold.

These results demonstrate that:
- confidence alone is insufficient for uncertainty estimation,
- feature-space monitoring is more reliable for OOD detection,
- and explainability degradation strongly correlates with environmental robustness failures.

This analysis extends the system-level safety case by explicitly incorporating OOD detection as a runtime safety-monitoring mechanism.

### Adversarial Robustness Safety Analysis

The FGSM evaluation extends the original STPA analysis with an explicit adversarial-input hazard:

- **H-6:** The vehicle continues autonomous driving while adversarially perturbed camera input causes false-negative or unreliable perception output without detection.
- **UCA-9:** The Planning Module provides a continue-driving command while acting on adversarially corrupted perception feedback.
- **Linked hazards:** H-1, H-2, H-4, and H-6.
- **Linked losses:** L-1, L-2, and L-3.

New safety constraints:

| ID | Level | Constraint |
|---|---|---|
| SC-9a | Model-level | For FGSM perturbations with `ε ≤ 0.01`, recall drop must not exceed 0.10 and recall must not collapse to zero. |
| SC-9b | System-level | When anomalous input, inconsistent perception, or suspected adversarial activity is detected, the planner must reduce speed and enter a safe fallback mode. |

All three classifiers currently violate SC-9a at `ε = 0.01`:

- Pedestrian recall drop: `0.1923`
- Traffic-light recall drop: `0.8101`
- Vehicle recall drop: `0.2982`

Adversarial training can reduce this risk but cannot eliminate it. Residual risks include stronger attacks, unseen perturbation types, physical-world attacks, OOD conditions, sensor failures, and unsafe planner behavior.

---

## 🔍 Key Findings

- ✅ **Traffic Light:** Best-performing model (F1: 0.944) under ideal conditions — completely blind at night and in fog
- ⚠️ **Vehicle:** Clean-test F1 is 0.858, but night recall falls to 0.372 (F1: 0.512) and Town-01 F1 falls to 0.719. Its fog F1 of 0.872 is misleading because it predicts nearly every frame as positive.
- ❌ **Pedestrian:** Most safety-critical and weakest model. Baseline recall of **0.108** means ~9 out of 10 pedestrians are missed. Fails completely at night. Requires redesign before any safety deployment claim.
- ⚠️ **Temperature scaling:** Accuracy is invariant to T — calibration is the missing safety metric
- 🔴 **Backdoor attack:** ASR=100% achieved with only 2.4% poisoned training data — standard evaluation cannot detect this
- 🔍 Grad-CAM explainability revealed strong dependence on environmental context and lighting conditions
- ⚠️ Pedestrian detector frequently relies on spurious sky and road-texture correlations
- 🌙 Night-condition explainability maps show complete loss of semantic object attention
- 📉 Explanation quality degrades together with OOD robustness
- 🌍 MSP-based OOD detection failed completely under nighttime conditions (AUROC = 0.0000)
- 🧠 Feature-based k-NN detection achieved perfect nighttime separation for pedestrian and traffic-light inputs, and 0.9971 AUROC for vehicle inputs
- ⚠️ Town-01 k-NN OOD detection remains below the 0.90 threshold for pedestrian (0.8956) and vehicle (0.8274)
- ⚠️ High classifier confidence does not guarantee safe perception under environmental shift
- 🔍 Deep feature embeddings are substantially more reliable for OOD detection than output confidence alone
- ⚔️ Untargeted FGSM reduced pedestrian recall to zero at every tested ε
- 🚦 Traffic-light recall dropped by 0.8101 at ε=0.01
- 🚗 Vehicle recall dropped by 0.8246 at ε=0.05
- ⚠️ Small, mostly imperceptible perturbations caused safety-critical false negatives
- 🛡️ Adversarial training must be combined with anomaly detection and system-level fallback
- 🎯 Temperature scaling reduced pedestrian ECE from 0.1477 to 0.0385
- ⚠️ The pedestrian classifier was strongly overconfident before calibration
- 📉 Calibration improved confidence reliability without changing accuracy
- 🛑 Cost-sensitive pedestrian braking with tau* ≈ 0.0099 reduced total loss from 63,429 to 2,894 after calibration
- ⚠️ The safest cost-sensitive threshold eliminated false negatives but caused many false positives, showing a safety-vs-availability trade-off


---

## 📈 Recommended Improvements

1. **Weighted BCE loss** for pedestrian model (weight positive class ~3.2×) to address class imbalance
2. **More training epochs** (20–30) with learning rate scheduling — pedestrian loss had not converged at epoch 5
3. **Add fog/night training data** — CARLA's weather API can generate these synthetically at no extra labelling cost
4. **Calibrate prediction threshold** per model — lower threshold for pedestrian detection to boost recall at cost of precision
5. **Larger backbone** (ResNet-50 or EfficientNet-B0) for better small-object detection
6. **Data provenance checks** — verify training data integrity to guard against poisoning attacks
7. **Backdoor detection** — apply spectral signatures or activation clustering before deploying any retrained model
8. Integrate feature-based OOD monitoring into runtime safety architecture
9. Retrain models with nighttime and adverse-weather augmentation
10. Add uncertainty-aware fallback logic for detected OOD conditions
11. Add adversarial training using FGSM and stronger iterative attacks such as PGD
12. Evaluate adversarial robustness across multiple random seeds and attack budgets
13. Add runtime adversarial-input and temporal-consistency monitoring
14. Trigger automatic speed reduction or safe stopping when suspicious input is detected
15. Use calibrated probabilities for downstream safety decisions instead of raw sigmoid scores
16. Validate calibration under OOD conditions, not only on the standard test set
17. Tune cost-sensitive decision thresholds jointly with planner-level comfort and availability constraints
18. Add runtime monitoring for calibration drift and uncertainty under distribution shift
19. Add temporal tracking and a false-positive-rate safety limit for vehicle detections, then retrain with fog, night, and map-diverse data

---

## 📄 Report

A full evaluation and safety analysis report is available in [`report/CARLA_ML_Safety_Report.pdf`](report/CARLA_ML_Safety_Report.pdf), covering:

- Dataset exploration & class imbalance analysis
- Model architecture & training convergence
- Baseline evaluation metrics
- ODD robustness evaluation (fog, night, Town-01)
- Safety constraints & STPA integration
- Temperature scaling calibration analysis
- Backdoor attack robustness evaluation
- Grad-CAM explainability analysis
- Misclassification diagnostics
- Explainability under OOD conditions
- Spurious correlation and shortcut-learning analysis
- Safety-oriented recommendation

---

## 🏫 Course Context

This project was developed as part of **Introduction to Machine Learning Safety**. Each exercise sheet produced one piece of evidence toward a structured **safety case** for the CARLA perception system:

| Exercise | Focus |
|---|---|
| Ex 1.4 | First impressions & safety-critical label identification |
| Ex 3.4 | Dataset exploration & class distribution |
| Ex 3.5 | Model architecture & training |
| Ex 3.6 | Evaluation — precision, recall, F1 |
| Ex 3.7 | ODD gap analysis |
| Ex 4.6 | Safety constraint test suite design |
| Ex 4.7 | Per-class evaluation & confusion matrices |
| Ex 5.4 | Temperature scaling & safety constraint calibration |
| Ex 5.5 | Backdoor attack — data poisoning & ASR evaluation |
| Ex 6.5 | Grad-CAM explainability analysis |
| Ex 6.6 | Explainability as diagnostic tool under OOD |
| Ex 7.1 | OOD problem & silent failure analysis |
| Ex 7.2 | MSP baseline OOD detection |
| Ex 7.3 | Advanced OOD detection methods |
| Ex 7.4 | Distribution shift visualization |
| Ex 7.5 | ODD vs OOD interpretation |
| Ex 7.6 | AUROC evaluation of MSP |
| Ex 7.7 | Feature-based k-NN OOD detection |
| Ex 7.8 | STPA extension for OOD safety risks |
| Ex 8.1 | Adversarial examples vs. OOD examples |
| Ex 8.2 | Gradient-based attack formulation |
| Ex 8.3 | Adversarial training and robustness trade-offs |
| Ex 8.4 | Untargeted FGSM attack implementation |
| Ex 8.5 | Recall and recall-drop robustness evaluation |
| Ex 8.6 | STPA extension for adversarial perception failures |
| Ex 9.1 | Epistemic vs. aleatoric uncertainty |
| Ex 9.2 | Calibration, reliability diagrams, and ECE |
| Ex 9.3 | Cost-optimal downstream decision thresholds |
| Ex 9.4 | ECE measurement for all classifiers |
| Ex 9.5 | Temperature scaling calibration |
| Ex 9.6 | Cost-sensitive pedestrian braking evaluation |
| Ex 9.7 | STPA extension for uncertainty and calibration failures |
---

## 📜 License

This project is for educational purposes as part of a university course assignment.

---

*Dataset and system specification provided as part of the Introduction to Machine Learning Safety course.*
