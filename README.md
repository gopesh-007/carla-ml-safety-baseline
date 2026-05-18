# CARLA ML Safety Baseline Models

## Overview

This project was developed as part of the course:

**Introduction to Machine Learning Safety**

The goal of this assignment was to train and evaluate three binary perception models using the CARLA autonomous driving simulator dataset.

The system uses a shared front-facing camera image and predicts:

- Pedestrian present
- Traffic light present
- Vehicle present

The project also includes robustness and ODD (Operational Design Domain) analysis under adverse conditions such as fog, night driving, and domain shift.

---

# Project Structure

```text
carla_baseline_project/
│
├── scripts/
├── notebooks/
├── models/
├── outputs/
├── reports/
├── requirements.txt
├── README.md
└── .gitignore
