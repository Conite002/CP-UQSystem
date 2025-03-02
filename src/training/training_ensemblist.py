import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
import os
import pandas as pd

from src.models.base_model import SimpleMNISTModel
from src.data.mnist_data import get_mnist_loaders
from src.training.training_baseline import train_single_model

def train_ensemble(
    num_models=5,
    epochs=10,
    lr=1e-3,
    batch_size=64,
    train_ratio=0.8,
    val_ratio=0.1,
    cal_ratio=0.1,
    seed=42,
    step_size=5,
    gamma=0.1,
    patience=5,
    model_dir="../src/checkpoints",
    results_path="../src/results/ensemble_training.csv",
    save_splits_path="../data/mnist_splits.pkl"
):
    """
    Trains an ensemble of CNN models on MNIST and saves training results for plotting.

    Args:
        num_models (int): Number of models in the ensemble.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        train_ratio, val_ratio, cal_ratio (floats): Data splits (should sum to 1.0).
        seed (int): Random seed.
        step_size (int): StepLR learning rate decay step.
        gamma (float): LR decay factor.
        patience (int): Early stopping patience.
        model_dir (str): Directory to save model checkpoints.
        results_path (str): Path to save training logs.
        save_splits_path (str): Path to store dataset splits.

    Returns:
        list: List of trained models.
    """

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("src/results", exist_ok=True)

    train_loader, val_loader, cal_loader, test_loader = get_mnist_loaders(
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        cal_ratio=cal_ratio,
        seed=seed,
        save_splits_path=save_splits_path
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    models = [SimpleMNISTModel().to(device) for _ in range(num_models)]
    optimizers = [optim.Adam(model.parameters(), lr=lr) for model in models]
    schedulers = [optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma) for opt in optimizers]
    criterion = nn.CrossEntropyLoss()

    best_val_accuracies = [0.0] * num_models
    best_model_paths = [os.path.join(model_dir, f"mnist_model_{i}.pth") for i in range(num_models)]
    training_logs = []

    for epoch in range(epochs):
        for model_idx, model in enumerate(models):
            model.train()
            optimizer = optimizers[model_idx]
            scheduler = schedulers[model_idx]

            train_losses, train_preds, train_labels_list = [], [], []
            val_losses, val_preds, val_labels_list = [], [], []

            # Training Phase
            train_pbar = tqdm(train_loader, desc=f"Model {model_idx+1}/{num_models} - Epoch {epoch+1}/{epochs} [Train]", leave=False)
            for images, labels in train_pbar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                _, predicted = torch.max(outputs, 1)
                train_preds.extend(predicted.cpu().tolist())
                train_labels_list.extend(labels.cpu().tolist())

            # Compute training metrics
            train_loss = sum(train_losses) / len(train_losses)
            train_acc = accuracy_score(train_labels_list, train_preds)
            train_recall = recall_score(train_labels_list, train_preds, average="macro")
            train_f1 = f1_score(train_labels_list, train_preds, average="macro")

            # Validation Phase
            model.eval()
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Model {model_idx+1}/{num_models} - Epoch {epoch+1}/{epochs} [Val]", leave=False)
                for images, labels in val_pbar:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_losses.append(loss.item())
                    _, predicted = torch.max(outputs, 1)
                    val_preds.extend(predicted.cpu().tolist())
                    val_labels_list.extend(labels.cpu().tolist())

            # Compute validation metrics
            val_loss = sum(val_losses) / len(val_losses)
            val_acc = accuracy_score(val_labels_list, val_preds)
            val_recall = recall_score(val_labels_list, val_preds, average="macro")
            val_f1 = f1_score(val_labels_list, val_preds, average="macro")

            print(f"\nModel {model_idx+1}/{num_models} - Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | Recall: {train_recall:.3f} | F1: {train_f1:.3f}")
            print(f"  Val   Loss: {val_loss:.4f}   | Acc: {val_acc*100:.2f}%   | Recall: {val_recall:.3f}   | F1: {val_f1:.3f}")

            # Early Stopping & Save Best Model
            if val_acc * 100 > best_val_accuracies[model_idx]:
                best_val_accuracies[model_idx] = val_acc * 100
                torch.save(model.state_dict(), best_model_paths[model_idx])
                print(f"  [INFO] Model {model_idx+1} saved with val_acc: {val_acc*100:.2f}% at: {best_model_paths[model_idx]}")

            # Step LR Scheduler
            scheduler.step()

            # Store logs for visualization
            training_logs.append({
                "Model": model_idx + 1,
                "Epoch": epoch + 1,
                "Train Loss": train_loss,
                "Train Accuracy": train_acc * 100,
                "Train Recall": train_recall,
                "Train F1": train_f1,
                "Val Loss": val_loss,
                "Val Accuracy": val_acc * 100,
                "Val Recall": val_recall,
                "Val F1": val_f1
            })

    # Save training logs for visualization
    df_logs = pd.DataFrame(training_logs)
    df_logs.to_csv(results_path, index=False)
    print(f"[INFO] Training logs saved at {results_path}")

    print(f"\nTraining complete. Best validation accuracy per model: {best_val_accuracies}")
    
    return models, cal_loader
#--------------------------------------------------------ENSEMBLE---------------------------------------------------------------------

def train_ensemble_models(
    num_models=5, epochs=10, lr=1e-3, batch_size=64, train_loader=None, val_loader=None, device=None,
    model_dir="src/checkpoints", results_path="src/results/ensemble_training.csv"
):
    """
    Trains multiple SimpleMNISTModels to create an ensemble.

    Args:
        num_models (int): Number of models in the ensemble.
        epochs (int): Number of epochs.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        train_loader (DataLoader): Training dataloader.
        val_loader (DataLoader): Validation dataloader.
        device (torch.device): Training device (CPU/GPU).
        model_dir (str): Directory to save models.
        results_path (str): Path to save training logs.

    Returns:
        list: List of trained models.
    """

    os.makedirs(model_dir, exist_ok=True)
    models = [SimpleMNISTModel().to(device) for _ in range(num_models)]
    model_paths = [os.path.join(model_dir, f"mnist_model_{i}.pth") for i in range(num_models)]

    print(f"[INFO] Training {num_models} models for the ensemble...")

    for i in range(num_models):
        print(f"\nTraining Model {i+1}/{num_models}")
        train_single_model(models[i], train_loader, val_loader, device, epochs=epochs, lr=lr, save_path=model_paths[i])

    print(f"\n[INFO] All {num_models} models trained and saved successfully.")
    
    return models
