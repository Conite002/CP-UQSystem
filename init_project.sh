#!/bin/bash

# Create directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p notebooks
mkdir -p src/models
mkdir -p src/torchcp_integration
mkdir -p src/training
mkdir -p src/evaluation
mkdir -p src/interpretability
mkdir -p scripts

# Create files
touch README.md
touch environment.yml  # or use requirements.txt if you prefer
touch notebooks/EDA.ipynb
touch notebooks/model_experiments.ipynb
touch src/models/base_model.py
touch src/torchcp_integration/conformal_utils.py
touch src/training/train.py
touch src/evaluation/metrics.py
touch src/interpretability/gradcam.py
touch scripts/run_training.sh
touch scripts/run_evaluation.sh

echo "Project structure created successfully!"
