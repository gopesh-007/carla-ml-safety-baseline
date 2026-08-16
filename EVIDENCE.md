# Safety-case evidence map

This file records where the evidence for the final safety-case verdicts came
from. It is meant as a practical audit trail: which checkpoint was evaluated,
which script produced the result, which data subset was used, and where the
result was saved.

The README preserves the project as it developed exercise by exercise. Some of
its early STPA constraint numbers are therefore provisional. For the submitted
safety case, the SC and V identifiers in
[`report/CARLA_ML_Safety_Report.pdf`](report/CARLA_ML_Safety_Report.pdf) are the
authoritative mapping.

## Reproduction boundary

- The dataset was supplied by the course staff and is not redistributed. The
  scripts expect it under `data/` with the structure described in the README.
- Commands below assume that `cd scripts` has already been run, as in the main
  usage instructions.
- The exact evaluated checkpoints are published in the
  [`v1.0-model-checkpoints` release](https://github.com/gopesh-007/carla-ml-safety-baseline/releases/tag/v1.0-model-checkpoints).
- V-1 predates the structured result files and its evaluators print metrics to
  the terminal. Those recorded values are reproduced in the README and final
  report. V-2, V-3 and V-4 have CSV result records in `outputs/`.
- Re-running an evaluator may replace its corresponding files in `outputs/`.

## Evaluated checkpoints

| File | SHA-256 |
|---|---|
| `pedestrian_model.pth` | `033e991fed2412c7675aa25aada36194d2505c722636fe2b44d5fe67d8e01e0a` |
| `traffic_light_model.pth` | `7a2444f42eb8d20c0d515396945eda35348952ab1d7d06329503562547929650` |
| `vehicle_model.pth` | `f6d98f8c1bb08fd7201d0a435c032655932d01b0174a75440827d99992f04d2a` |
| `pedestrian_model_backdoor.pth` | `cd13f2d3d266e485b8434a000fea883ded5ce68f6b4fd28f518920b3f126b4bb` |

## Verification summary

| ID | Final criterion | Main result | Verdict |
|---|---|---|---|
| V-1 | ID recall >= 0.90 for pedestrian and >= 0.85 for traffic light/vehicle | Pedestrian recall 0.1076; traffic-light 0.9358; vehicle 0.8537 | Not met |
| V-2 | Recall drop at FGSM epsilon 0.05 < 0.10 | Drops of 0.1923, 0.8987 and 0.8246 | Not met |
| V-3 | Test ECE < 0.05 after validation-only temperature selection | ECE 0.0385, 0.0246 and 0.0177 | Met |
| V-4 | k-NN OOD AUROC >= 0.90 for every model/shift | Pedestrian Town-01 0.8956; vehicle Town-01 0.8274 | Partial |
| V-5 | Invalid/OOD/low-confidence input leads to a verified minimal-risk response | Required fallback functions are not implemented | Not met |

## V-1: in-distribution detection performance

**Trace in the final report:** SC-1 and SC-8; LS-1, LS-5 and LS-9.

The three baseline evaluators load the released ResNet-18 checkpoints and
evaluate the complete dry/day/source-town test split (3,600 frames per model)
at a sigmoid threshold of 0.5.

```bash
python evaluate_pedestrian.py
python evaluate_traffic_light.py
python evaluate_vehicle.py
```

| Model | Accuracy | Recall | Required recall | Result |
|---|---:|---:|---:|---|
| Pedestrian | 0.7058 | 0.1076 | 0.90 | Not met |
| Traffic light | 0.9206 | 0.9358 | 0.85 | Met |
| Vehicle | 0.7875 | 0.8537 | 0.85 | Met |

**Record:** evaluator terminal output, transcribed into the README baseline
table and Section 4 (V-1) of the final report. This is the one main verification
without a dedicated result CSV.

**Interpretation:** the overall verdict is not met because the pedestrian
threshold fails. Passing traffic-light and vehicle recall does not compensate
for that failure. The vehicle result is also only 0.0037 above its threshold and
uses the documented historical class-weight error.

## V-2: FGSM perturbation robustness

**Trace in the final report:** SC-2 and LS-2.

```bash
python evaluate_adversarial.py \
  --limit 100 \
  --sample-mode random \
  --seed 42
```

The evaluator uses the same reproducible sample of 100 validation frames per
model and evaluates clean input plus epsilon values 0.01, 0.05 and 0.10. The
verification point is epsilon 0.05.

| Model | Clean recall | Recall at 0.05 | Recall drop | Required drop | Result |
|---|---:|---:|---:|---:|---|
| Pedestrian | 0.1923 | 0.0000 | 0.1923 | < 0.10 | Not met |
| Traffic light | 0.8987 | 0.0000 | 0.8987 | < 0.10 | Not met |
| Vehicle | 0.8596 | 0.0351 | 0.8246 | < 0.10 | Not met |

**Primary record:**
[`outputs/adversarial/adversarial_results.csv`](outputs/adversarial/adversarial_results.csv)

**Supporting record:**
[`outputs/adversarial/plots/recall_vs_epsilon.png`](outputs/adversarial/plots/recall_vs_epsilon.png)
and the example images in `outputs/adversarial/examples/`.

The experiment is a digital white-box robustness test. It rejects the stated
bound, but it is not presented as evidence of a physical-world attack.

## V-3: calibrated uncertainty

**Trace in the final report:** SC-3 and LS-3.

```bash
python evaluate_uncertainty.py
```

The script fits one scalar temperature per model by minimizing binary
cross-entropy on the complete validation split. The search grid is 0.5 to 3.0
in steps of 0.1. It then computes ten-bin ECE on the complete 3,600-frame test
split; the test set is not used to choose the temperature.

| Model | Selected temperature | Test ECE before | Test ECE after | Result |
|---|---:|---:|---:|---|
| Pedestrian | 3.0 | 0.1477 | 0.0385 | Met |
| Traffic light | 1.2 | 0.0349 | 0.0246 | Met |
| Vehicle | 0.9 | 0.0286 | 0.0177 | Met |

**Primary record:**
[`outputs/uncertainty/calibration_results.csv`](outputs/uncertainty/calibration_results.csv)

**Supporting records:** `outputs/uncertainty/temperature_search.csv`,
`outputs/uncertainty/reliability_bins.csv`,
[`outputs/uncertainty/reliability_diagrams.png`](outputs/uncertainty/reliability_diagrams.png),
and `outputs/uncertainty/pedestrian_cost_results.csv`.

This verdict concerns calibration on in-distribution data only. Temperature
scaling does not change the classifications, repair V-1, or implement a safe
response to uncertainty.

## V-4: out-of-distribution detection

**Trace in the final report:** SC-4 and SC-9; LS-4 and LS-10.

```bash
python evaluate_ood_knn_all_models.py
```

For each classifier, the script extracts ResNet-18 embeddings from the first
1,000 lexicographically sorted validation images and fits a five-nearest-
neighbour detector. It compares the resulting mean distances with 1,000 images
from fog, night and Town-01 separately.

| Model | Fog AUROC | Night AUROC | Town-01 AUROC | Result |
|---|---:|---:|---:|---|
| Pedestrian | 0.9844 | 1.0000 | 0.8956 | Partial |
| Traffic light | 0.9698 | 1.0000 | 0.9556 | Met |
| Vehicle | 0.9940 | 0.9971 | 0.8274 | Partial |

**Primary record:**
[`outputs/ood/knn_all_models_results.csv`](outputs/ood/knn_all_models_results.csv)

The overall verdict is partial because two Town-01 comparisons fall below
0.90. The monitor was evaluated offline and is not connected to the planner, so
this result cannot by itself satisfy continuous ODD enforcement.

## V-5: safe system fallback

**Trace in the final report:** SC-5 to SC-10; LS-3, LS-4 and LS-6 to LS-10.

V-5 is a design inspection rather than a model experiment. The final system
description and STPA control structure were checked against the required
fallback behaviour. The design has human brake/steering override and maximum
braking for a detected close object, but it has no integrated response to OOD
or low confidence, no camera-health monitor, no auditory warning, no measured
takeover bound, and no automatic minimal-risk stop if the operator does not
respond. The traffic-light classifier also reports presence rather than signal
state.

**Record:** Sections 1, 3 and 4 (V-5) of the final report. There is no generated
CSV and no command to run because the required fallback is not implemented.

**Verdict:** not met. This negative result is retained in the safety case rather
than being treated as missing data.

## Supporting evidence outside V-1 to V-5

These artifacts support the discussion but do not replace the five final
verification verdicts.

| Topic | Reproduction | Record and role |
|---|---|---|
| ODD test coverage | `python evaluate_odd_coverage.py` | `outputs/odd/odd_test_scenarios.csv` and `outputs/odd/odd_k_projection_coverage.csv`; shows 100.0% one-way, 62.5% two-way and 33.3% three-way coverage |
| Adverse-condition classification | `python evaluate_*_fog.py`, `python evaluate_*_night.py`, `python evaluate_*_town.py` | Terminal metrics summarized in the README/report; establishes degradation under the tested shifts |
| Vehicle robustness detail | `python evaluate_vehicle_robustness.py` | Confusion counts summarized in the README/report; fog produces 3,599 positive predictions in 3,600 frames |
| Grad-CAM | `python gradcam_analysis.py` | Images under `outputs/explainability/`; qualitative support for the shortcut-learning discussion, not a quantitative verification |
| Backdoor experiment | `python train_pedestrian_backdoor.py`, then `python evaluate_backdoor.py` | Released backdoor checkpoint and reported 100% ASR; supports training-data integrity recommendations |
| Cost-sensitive pedestrian threshold | included in `python evaluate_uncertainty.py` | `outputs/uncertainty/pedestrian_cost_results.csv`; illustrates the false-negative/false-positive trade-off |

## What is not claimed

- No result here establishes closed-loop vehicle safety on public roads.
- Fog, night and Town-01 are tested shifts, not validated operating conditions.
- The perfect distance oracle is an assumption from the supplied system
  description, not an evaluated component of this repository.
- Grad-CAM is used as diagnostic evidence, not as proof that a model learned the
  intended object concept.
- The final top-level safety claim remains unsupported because V-1, V-2 and V-5
  are not met and V-4 is only partial.
