#!/bin/bash

##############################################################################
# run_training.sh
#
# Example usage:
#   ./scripts/run_training.sh \
#       --model_path src/checkpoints/model_TrTrainer.pth \
#       --batch_size 64 \
#       --epochs 5 \
#       --lr 0.001 \
#       --train_ratio 0.7 \
#       --val_ratio 0.15 \
#       --cal_ratio 0.15
#
# Named Arguments:
#   --model_path   : File path to save the trained model (default: "src/checkpoints/model_default.pth")
#   --batch_size   : Batch size for training (default: 64)
#   --epochs       : Number of training epochs (default: 5)
#   --lr           : Learning rate (default: 0.001)
#   --train_ratio  : Fraction of training data for actual training (default: 0.7)
#   --val_ratio    : Fraction of training data for validation (default: 0.15)
#   --cal_ratio    : Fraction of training data for calibration (default: 0.15)
##############################################################################

# 1. Default values
MODEL_PATH="src/checkpoints/model_default.pth"
BATCH_SIZE=64
EPOCHS=5
LR=0.001
TRAIN_RATIO=0.7
VAL_RATIO=0.15
CAL_RATIO=0.15

# 2. Parse named arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --model_path)
      MODEL_PATH="$2"
      shift
      shift
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift
      shift
      ;;
    --epochs)
      EPOCHS="$2"
      shift
      shift
      ;;
    --lr)
      LR="$2"
      shift
      shift
      ;;
    --train_ratio)
      TRAIN_RATIO="$2"
      shift
      shift
      ;;
    --val_ratio)
      VAL_RATIO="$2"
      shift
      shift
      ;;
    --cal_ratio)
      CAL_RATIO="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      shift
      ;;
  esac
done

# 3. Print configuration
echo "=== Starting training ==="
echo "Model Path:    $MODEL_PATH"
echo "Batch Size:    $BATCH_SIZE"
echo "Epochs:        $EPOCHS"
echo "Learning Rate: $LR"
echo "Train Ratio:   $TRAIN_RATIO"
echo "Val Ratio:     $VAL_RATIO"
echo "Cal Ratio:     $CAL_RATIO"

# 4. Run Python training code
python -c "
import sys
import os
import shutil

# Ensure we can import local modules (project root assumed one level up)
sys.path.append('..')

from src.training.training_baseline import train_baseline

print(f'[INFO] Training with baseline, epochs=${EPOCHS}, lr=${LR}, batch_size=${BATCH_SIZE}...')
print(f'[INFO] Splits => train_ratio=${TRAIN_RATIO}, val_ratio=${VAL_RATIO}, cal_ratio=${CAL_RATIO}')

# 5. Call the training function
trained_model, calibration_loader = train_baseline(
    epochs=int('${EPOCHS}'),
    lr=float('${LR}'),
    batch_size=int('${BATCH_SIZE}'),
    train_ratio=float('${TRAIN_RATIO}'),
    val_ratio=float('${VAL_RATIO}'),
    cal_ratio=float('${CAL_RATIO}'),
    seed=42
)

# 6. Save the model inside train_baseline (or here). If train_baseline
#    already saves the model, you can remove or adapt this step.

# We'll define a temporary path to store the model:
model_path_temp = 'trained_model_temp.pth'

import torch
torch.save(trained_model.state_dict(), model_path_temp)
print(f'[INFO] Temporary model path returned: {model_path_temp}')

if os.path.exists(model_path_temp):
    # Move or rename the output to the specified MODEL_PATH
    os.makedirs(os.path.dirname('${MODEL_PATH}'), exist_ok=True)
    shutil.move(model_path_temp, '${MODEL_PATH}')
    print(f'[INFO] Model saved at: ${MODEL_PATH}')
else:
    print(f'[WARNING] {model_path_temp} not found. Check your training function output.')
"

echo "=== Training completed ==="
