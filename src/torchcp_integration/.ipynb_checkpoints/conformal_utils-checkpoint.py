import torch
import torch.nn.functional as F
import torch.optim as optim
import sys, os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.extend(['src/torchcp_integration/', 'src/', 'src/torchcp_integration/predictors'])
sys.path.append('..')
from torchcp.classification.predictor import SplitPredictor, ClassWisePredictor, ClusteredPredictor, WeightedPredictor
from predictors.split import SplitPredictor_CM
from torchcp.classification.score import APS, RAPS, SAPS, TOPK, KNN
from transformers import set_seed
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


### =9 Monte Carlo Dropout Functions
def enable_mc_dropout(model):
    """Enables MC Dropout by setting dropout layers to training mode."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

def mc_dropout_forward(model, x_batch, mc_samples=10):
    """Runs MC Dropout on the model for x_batch and returns averaged logits."""
    enable_mc_dropout(model)
    return torch.stack([model(x_batch).detach() for _ in range(mc_samples)]).mean(dim=0)


### =9 Deep Ensemble Learning Functions
def ensemble_forward(models, x_batch):
    """Runs ensemble forward pass on a list of models for x_batch and returns averaged logits."""
    return torch.stack([model(x_batch) for model in models]).mean(dim=0)


### =9 Function to Gather Logits & Labels
def gather_logits_labels(model, data_loader, device, mc_dropout=False, mc_samples=10, ensemble_models=None):
    """
    Passes the data through the model to gather logits & labels.

    Args:
        model (nn.Module or list): Model or list of models.
        data_loader (DataLoader): DataLoader object.
        device (torch.device): Device (CPU/GPU).
        mc_dropout (bool): Whether to use MC Dropout.
        mc_samples (int): Number of MC samples.
        ensemble_models (list): List of models for ensemble learning.

    Returns:
        torch.Tensor, torch.Tensor: Logits and labels.
    """
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            outputs = (
                mc_dropout_forward(model, images, mc_samples) if mc_dropout else
                ensemble_forward(ensemble_models, images) if ensemble_models else
                model(images)
            )
            
            all_logits.append(outputs.cpu())
            all_labels.append(labels)

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


### =9 Main Conformal Prediction Function
def posthoc_conformal_calibration(
    model, cal_loader, test_loader, device=None, alpha=0.1, seed=42,
    num_classes=10, mc_dropout=False, mc_samples=10, ensemble_models=None,
    init_temperatures=[1.5], num_samples_to_show=5, use_interpretability=False,
    save_results_dir="src/results/"
):
    """
    Runs multiple TorchCP score & predictor methods for each type (single, ensemble, mc_dropout).
    Saves prediction sets, true labels, and generates distribution plots.

    Args:
        model (nn.Module): Base classification model.
        cal_loader, test_loader: DataLoaders for calibration/test sets.
        device (torch.device or None): CPU/GPU. If None, auto-detect.
        alpha (float): Significance level (1 - coverage).
        seed (int): Random seed for reproducibility.
        num_classes (int): e.g., 10 for MNIST.
        mc_dropout (bool): Whether to use MC Dropout.
        mc_samples (int): Number of MC samples.
        ensemble_models (list): List of models for ensemble learning.
        init_temperatures (list): List of initial temps to test.
        num_samples_to_show (int): Number of samples to display prediction sets.
        use_interpretability (bool): Whether to generate Grad-CAM visualizations.
        save_results_dir (str): Directory to save results.

    Returns:
        pd.DataFrame: Results DataFrame.
    """



    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_results_dir, exist_ok=True)

    model.to(device).eval()
    if ensemble_models:
        for m in ensemble_models:
            m.eval()

    print(f"[INFO] Running post-hoc conformal calibration on {device}")
    print(f"[INFO] Using {'Ensemble Learning' if ensemble_models else 'Single Model'}")
    print(f"[INFO] Monte Carlo Dropout: {mc_dropout}, MC Samples: {mc_samples}")

    score_methods = {
        "APS": APS(score_type="softmax", randomized=True),
        "RAPS": RAPS(score_type="softmax", randomized=True, penalty=0.1, kreg=1),
        "SAPS": SAPS(score_type="softmax", randomized=True, weight=0.2),
        "TOPK": TOPK(score_type="softmax", randomized=True),
    }

    predictor_classes = {
        "single": {
            # "SplitPredictor": SplitPredictor,
            # "ClassWisePredictor": ClassWisePredictor,
            # "ClusteredPredictor": ClusteredPredictor,
            "SplitPredictor_CM": SplitPredictor_CM
        },
        "ensemble": {
            "SplitPredictor_CM": SplitPredictor_CM
        },
        "mc_dropout": {
            "SplitPredictor_CM": SplitPredictor_CM
        }
    }

    results_list, all_pred_sets, lengths_expanded_sets = [], [], []

    for t in tqdm(["single", "ensemble", "mc_dropout"], desc="Processing TYPES"):
        print(f"\n[INFO] Processing Type: {t}")

        cal_logits, cal_labels = gather_logits_labels(
            model, cal_loader, device,
            mc_dropout=(t == "mc_dropout"), mc_samples=mc_samples,
            ensemble_models=ensemble_models if t == "ensemble" else None
        )
        test_logits, test_labels = gather_logits_labels(
            model, test_loader, device,
            mc_dropout=(t == "mc_dropout"), mc_samples=mc_samples,
            ensemble_models=ensemble_models if t == "ensemble" else None
        )

        score_methods["KNN"] = KNN(features=cal_logits, labels=cal_labels, num_classes=num_classes, k=10, p=2)

        for score_name, score_obj in score_methods.items():
            for predictor_name, predictor_cls in predictor_classes[t].items():
                predictor = predictor_cls(
                    score_function=score_obj,
                    model=ensemble_models if predictor_name == "SplitPredictor_CM" else model
                )

                predictor.calibrate(cal_loader, alpha=alpha)
                result_dict = predictor.evaluate(test_loader)

                all_pred_sets_batch = []
                all_lengths_batch = []

                for batch_images, batch_labels in test_loader:  # Process all batches
                    batch_images = batch_images.to(device)
                    batch_pred_sets = predictor.predict(batch_images)

                    batch_pred_classes = [
                        set(torch.where(pset == 1)[0].tolist()) for pset in batch_pred_sets
                    ]
                    batch_pred_lengths = [
                        int(len(pred_set)) for pred_set in batch_pred_classes
                    ]

                    all_pred_sets_batch.extend(batch_pred_classes)
                    all_lengths_batch.extend(batch_pred_lengths)

                    for i, (pset, true_label) in enumerate(zip(batch_pred_sets[:num_samples_to_show], batch_labels[:num_samples_to_show])):
                        predicted_classes = {idx for idx, val in enumerate(pset) if val.item() == 1}
                        print(f"Sample {i}: True Label={true_label.item()}, Prediction Set={predicted_classes}")

                print(f"[RESULT] {score_name}-{predictor_name}: Coverage={result_dict['coverage_rate']:.4f}, "
                      f"Avg Set Size={result_dict['average_size']:.2f}")
                # print(f"List {all_lengths_batch}")

                all_pred_sets.append({
                    "Type": t,
                    "ScoreMethod": score_name,
                    "Predictor": predictor_name,
                    "True Labels": test_labels.cpu().tolist(),
                    "Prediction Sets": all_pred_sets_batch,
                    "lengths": all_lengths_batch
                })

                results_list.append({
                    "Type": t,
                    "Temperature": init_temperatures[0],
                    "ScoreMethod": score_name,
                    "Predictor": predictor_name,
                    "CoverageRate": result_dict['coverage_rate'],
                    "AvgSetSize": result_dict['average_size'],
                })

                lengths_expanded_sets.append(all_lengths_batch)

    lengths_expanded = pd.DataFrame(lengths_expanded_sets)
    lengths_expanded.to_csv(os.path.join(save_results_dir, "lengths.csv"), index=False)
    
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(os.path.join(save_results_dir, "cp_results.csv"), index=False)
    
    pd.DataFrame(all_pred_sets).to_csv(os.path.join(save_results_dir, "prediction_sets.csv"), index=False)

    print("\n[INFO] Conformal Calibration Results & Prediction Sets Saved.")
    return results_df