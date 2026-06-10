# 🚗 CARLA ML Safety Baseline Models

> **Course:** Introduction to Machine Learning Safety  
> **Project:** Binary perception classifiers for autonomous driving safety analysis  
> **Author:** Gopeshkumar Harsukh Rabadiya 

---

## 📋 Overview

This project trains and evaluates **three independent binary classifiers** for safety-critical perception tasks in autonomous driving, using data from the [CARLA](https://carla.org/) autonomous driving simulator.

Each model takes a single **front-facing RGB camera image** as input and outputs a binary prediction:

| Model | Task | Label |
|---|---|---|
| Pedestrian Detector | Is a pedestrian present? | `has_pedestrian` |
| Traffic Light Detector | Is a traffic light present? | `has_traffic_light` |
| Vehicle Detector | Is a vehicle present? | `has_vehicle` |

The models' outputs feed directly into a vehicle's **autopilot decision logic**, making recall and robustness the primary safety metrics.

This project also includes a full **ODD (Operational Design Domain) robustness evaluation** under fog, night, and domain-shift conditions, plus **temperature scaling** and **backdoor attack** analysis — forming evidence toward a complete safety case.

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
| Pedestrian | Fog | 0.315 | 0.269 | Partial degradation |
| Pedestrian | Night | 0.000 | 0.000 | Complete collapse |
| Pedestrian | Town-01 | 0.277 | 0.180 | Below safe threshold |

> ⚠️ **Safety note:** Neither the pedestrian nor traffic light model should be deployed for night or fog conditions. The pedestrian model additionally fails to meet minimum safety recall thresholds even under ideal conditions.

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

## 🗂️ Repository Structure

```text
carla_baseline_project/
│
├── scripts/
│   ├── train_pedestrian.py                  # Train pedestrian classifier (baseline)
│   ├── train_traffic_light.py               # Train traffic light classifier
│   ├── train_vehicle.py                     # Train vehicle classifier
│   ├── evaluate_pedestrian.py               # Evaluate on standard test set
│   ├── evaluate_traffic_light.py
│   ├── evaluate_vehicle.py
│   ├── evaluate_pedestrian_fog.py           # Robustness: fog
│   ├── evaluate_pedestrian_night.py         # Robustness: night
│   ├── evaluate_pedestrian_town.py          # Robustness: town-01 domain shift
│   ├── evaluate_traffic_light_fog.py
│   ├── evaluate_traffic_light_night.py
│   ├── evaluate_traffic_light_town.py
│   ├── evaluate_temperature_scaling.py      # Ex 5.4 — temperature scaling
│   ├── plot_temperature_distribution.py     # Ex 5.4 — probability distribution plot
│   ├── train_pedestrian_backdoor.py         # Ex 5.5 — backdoor poisoning + retrain
│   ├── evaluate_backdoor.py                 # Ex 5.5 — clean recall + ASR                  
│   └── gradcam_analysis.py                  # Exercise 6 — Grad-CAM analysis - Generate Grad-CAM explanations
│
├── notebooks/
│   └── dataset_exploration.ipynb            # Dataset exploration & visualisations
│
├── models/                                  # Saved .pth model weights (not tracked)
│   ├── pedestrian_model.pth
│   ├── traffic_light_model.pth
│   ├── vehicle_model.pth
│   └── pedestrian_model_backdoor.pth        # Backdoored model (Ex 5.5)
│
├── outputs/
│    ├──temperature_distribution.png         # Ex 5.4 distribution plot
│    └── explainability/
│       ├── pedestrian/
│       ├── traffic_light/
│       ├── vehicle/
│       ├── baseline/
│       ├── fog/
│       ├── night/
│       └── town01/
│
├── report/
│   └── CARLA_ML_Safety_Report.pdf           # Full evaluation & safety analysis report
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧠 Model Architecture

All three classifiers share the same architecture:

- **Backbone:** ResNet-18 (pre-trained on ImageNet, fine-tuned)
- **Output head:** Single neuron with sigmoid activation
- **Loss function:** Binary Cross-Entropy with class weighting (pos_weight = 5482/1718 for pedestrian)
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 5
- **Device:** CPU

### Training Loss (5 Epochs)

| Model | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 |
|---|---|---|---|---|---|
| Pedestrian | 1.0457 | 0.9702 | 0.9159 | 0.8870 | 0.8415 |
| Traffic Light | 0.1805 | 0.1006 | 0.0681 | 0.0605 | 0.0499 |
| Vehicle | 0.7441 | 0.6360 | 0.5533 | 0.4938 | 0.4618 |

> The traffic light model converged excellently. The pedestrian model shows high residual loss (0.84), indicating insufficient training and the impact of class imbalance.

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

- Python 3.8+
- pip

### Install dependencies

```bash
git clone https://github.com/gopesh-007/carla-ml-safety-baseline
cd carla_baseline_project
pip install -r requirements.txt
```

### Requirements

```bash
torch
torchvision
pandas
numpy
matplotlib
scikit-learn
Pillow
jupyter
grad-cam
opencv-python
matplotlib
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

# Night
python evaluate_pedestrian_night.py
python evaluate_traffic_light_night.py

# Domain shift (Town-01)
python evaluate_pedestrian_town.py
python evaluate_traffic_light_town.py
```

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
### Generated Outputs

The scripts produce:
- Correctly classified Grad-CAM overlays
- Misclassification explainability analysis
- OOD condition explainability (fog/night/Town-01)
- Heatmap visualizations for all three classifiers

Generated images are stored in:
```bash
outputs/explainability/  
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
| Night driving | 🔴 Critical | Pedestrian & traffic light recall = 0.0 |
| Fog | 🔴 Critical | Pedestrian & traffic light recall = 0.0 |
| New towns / maps | 🟠 High | Traffic light F1 drops from 0.944 → 0.416 |
| Rain, snow, dusk | 🟠 High | Not evaluated — assumed unsafe |

### Safety Constraints

| ID | Constraint |
|---|---|
| SC-1 | Pedestrian model must achieve recall ≥ 0.95 before deployment |
| SC-2 | System must not operate at night or in fog without validated models |
| SC-3 | Traffic light model must be re-validated when the operational map changes |
| SC-4 | Camera-only perception is insufficient — LIDAR/RADAR redundancy required |
| SC-5 | Class imbalance must be addressed before retraining |

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

---

## 🔍 Key Findings

- ✅ **Traffic Light:** Best-performing model (F1: 0.944) under ideal conditions — completely blind at night and in fog
- ✅ **Vehicle:** Solid performance (F1: 0.858) — not yet evaluated under adverse conditions
- ❌ **Pedestrian:** Most safety-critical and weakest model. Baseline recall of **0.108** means ~9 out of 10 pedestrians are missed. Fails completely at night. Requires redesign before any safety deployment claim.
- ⚠️ **Temperature scaling:** Accuracy is invariant to T — calibration is the missing safety metric
- 🔴 **Backdoor attack:** ASR=100% achieved with only 2.4% poisoned training data — standard evaluation cannot detect this
- 🔍 Grad-CAM explainability revealed strong dependence on environmental context and lighting conditions
- ⚠️ Pedestrian detector frequently relies on spurious sky and road-texture correlations
- 🌙 Night-condition explainability maps show complete loss of semantic object attention
- 📉 Explanation quality degrades together with OOD robustness


---

## 📈 Recommended Improvements

1. **Weighted BCE loss** for pedestrian model (weight positive class ~3.2×) to address class imbalance
2. **More training epochs** (20–30) with learning rate scheduling — pedestrian loss had not converged at epoch 5
3. **Add fog/night training data** — CARLA's weather API can generate these synthetically at no extra labelling cost
4. **Calibrate prediction threshold** per model — lower threshold for pedestrian detection to boost recall at cost of precision
5. **Larger backbone** (ResNet-50 or EfficientNet-B0) for better small-object detection
6. **Data provenance checks** — verify training data integrity to guard against poisoning attacks
7. **Backdoor detection** — apply spectral signatures or activation clustering before deploying any retrained model

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

---

## 📜 License

This project is for educational purposes as part of a university course assignment.

---

*CARLA Simulator — [carla.org](https://carla.org/) | PyTorch — [pytorch.org](https://pytorch.org/)*
