#!/bin/bash

##############################################################################
# run_conformal.sh
#
# Example usage:
#   ./scripts/run_conformal.sh --alpha 0.05 --seed 123 --num_samples_to_show 10 --batch_size 32
#
# Named Arguments:
#   --alpha             : Significance level for conformal (default: 0.1)
#   --seed              : Random seed (default: 42)
#   --num_samples_to_show : How many test samples to display (default: 5)
#   --batch_size        : Batch size for DataLoaders (default: 64)
#   --splits_path       : Path to pre-saved MNIST splits (default: src/data/mnist_splits.pkl)
##############################################################################

# 1. Default values
ALPHA=0.1
SEED=42
NUM_SAMPLES_TO_SHOW=5
BATCH_SIZE=64
SPLITS_PATH="data/mnist_splits.pkl"
EPOCHS=10
# 2. Parse named arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --alpha)
      ALPHA="$2"
      shift
      shift
      ;;
    --seed)
      SEED="$2"
      shift
      shift
      ;;
    --num_samples_to_show)
      NUM_SAMPLES_TO_SHOW="$2"
      shift
      shift
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift
      shift
      ;;
    --splits_path)
      SPLITS_PATH="$2"
      shift
      shift
      ;;
    --epochs)
      EPOCHS="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      shift
      ;;
  esac
done

echo "=== Running conformal calibration ==="
echo "Alpha:              $ALPHA"
echo "Seed:               $SEED"
echo "Num Samples to Show $NUM_SAMPLES_TO_SHOW"
echo "Batch Size:         $BATCH_SIZE"
echo "Splits Path:        $SPLITS_PATH"
echo "Epochs:             $EPOCHS"

# 3. Run Python snippet with the above arguments
python -c "
import sys
sys.path.append('.')  # Ensure we can import local modules

import torch
from src.torchcp_integration.conformal_utils import posthoc_conformal_calibration
from src.data.mnist_data import get_mnist_loaders
from src.models.base_model import SimpleMNISTModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleMNISTModel()
model.load_state_dict(torch.load('src/checkpoints/baseline_mnist.pth', map_location=device))

# Reconstruct DataLoaders with user-specified batch_size and splits path
train_loader, val_loader, cal_loader, test_loader = get_mnist_loaders(
    batch_size=${BATCH_SIZE},
    load_splits_path='${SPLITS_PATH}'
)

# Now run post-hoc conformal
posthoc_conformal_calibration(
    model,
    cal_loader,
    test_loader,
    device=device,
    alpha=${ALPHA},
    seed=${SEED},
    num_samples_to_show=${NUM_SAMPLES_TO_SHOW},
    num_epochs=${EPOCHS}
)
"

echo "=== Conformal calibration completed ==="
