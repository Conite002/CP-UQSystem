#!/bin/bash

##############################################################################
# run_evaluation.sh
#
# Example usage:
#   ./scripts/run_evaluation.sh --model_path model_TrTrainer.pth --batch_size 64
#
# Named Arguments:
#   --model_path : Path to the saved model weights (e.g., "model_TrTrainer.pth")
#   --batch_size : Batch size for evaluation (default: 64)
#   --seed       : Random seed (default: 42)
##############################################################################

# Default values
MODEL_PATH=""
BATCH_SIZE=64
SEED=42

# Parse named arguments
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
    --seed)
      SEED="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      shift
      ;;
  esac
done

# Check required args
if [ -z "$MODEL_PATH" ]; then
  echo "Error: --model_path is required."
  exit 1
fi

echo "=== Starting evaluation ==="
echo "Model Path:   $MODEL_PATH"
echo "Batch Size:   $BATCH_SIZE"
echo "Seed:         $SEED"

python -c "
import sys
sys.path.append('.')

from src.test_model import test_trained_model

model_path = '${MODEL_PATH}'
batch_size = int('${BATCH_SIZE}')
seed = int('${SEED}')

test_acc = test_trained_model(model_path, batch_size=batch_size, seed=seed)
print(f'Final Test Accuracy for {model_path}: {test_acc:.2f}%')
"

echo "=== Evaluation completed ==="
