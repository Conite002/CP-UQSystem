import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
from src.models.base_model import SimpleMNISTModel
from src.data.mnist_data import get_mnist_loaders 
import os

def train_baseline(
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
    model_path="src/src/checkpoints/_mnist.pth",
    save_splits_path="data/mnist_splits.pkl"
):
    """
    Trains a baseline CNN on MNIST using separate train, val, cal, and test sets,
    providing:
      - dynamic LR scheduler (StepLR)
      - detailed metrics (train/val loss, accuracy, recall, F1-score)
      - progress bars with tqdm
      - per-class statistics using classification_report on test set
      - early stopping with 'patience' epochs

    Args:
        epochs (int): Number of training epochs.
        lr (float): Initial learning rate.
        batch_size (int): Batch size for DataLoaders.
        train_ratio, val_ratio, cal_ratio (floats): Must sum to 1.0 for the 60k training images.
        seed (int): Random seed for reproducibility.
        step_size (int): Number of epochs between LR decay steps (for StepLR).
        gamma (float): Multiplicative factor of LR decay (for StepLR).
        patience (int): Number of epochs to wait for improvement in val accuracy before stopping.

    Returns:
        model (nn.Module): The best-performing model (based on validation accuracy).
        cal_loader (DataLoader): The calibration set loader, to be used later for conformal methods.
    """

    train_loader, val_loader, cal_loader, test_loader = get_mnist_loaders(
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        cal_ratio=cal_ratio,
        seed=seed,
        save_splits_path=save_splits_path
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMNISTModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss()
    print(f"Device :{device}")
    best_val_accuracy = 0.0
    os.makedirs("src/checkpoints", exist_ok=True)
    best_model_path = model_path

    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_preds = []
        train_labels_list = []

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
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

        train_loss = sum(train_losses) / len(train_losses)
        train_acc = accuracy_score(train_labels_list, train_preds)
        train_recall = recall_score(train_labels_list, train_preds, average='macro')
        train_f1 = f1_score(train_labels_list, train_preds, average='macro')


        model.eval()
        val_losses = []
        val_preds = []
        val_labels_list = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())

                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().tolist())
                val_labels_list.extend(labels.cpu().tolist())

        val_loss = sum(val_losses) / len(val_losses)
        val_acc = accuracy_score(val_labels_list, val_preds)
        val_recall = recall_score(val_labels_list, val_preds, average='macro')
        val_f1 = f1_score(val_labels_list, val_preds, average='macro')

        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | Recall: {train_recall:.3f} | F1: {train_f1:.3f}")
        print(f"  Val   Loss: {val_loss:.4f}   | Acc: {val_acc*100:.2f}%   | Recall: {val_recall:.3f}   | F1: {val_f1:.3f}")


        val_accuracy_percent = val_acc * 100.0
        if val_accuracy_percent > best_val_accuracy:
            best_val_accuracy = val_accuracy_percent
            torch.save(model.state_dict(), best_model_path )
            print(f"  [INFO] New best model saved with val_acc: {val_accuracy_percent:.2f}% at: {best_model_path}" )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\n[EARLY STOPPING] No improvement in validation accuracy for {patience} consecutive epochs.")
            print(f"Best validation accuracy so far: {best_val_accuracy:.2f}%")
            break

        scheduler.step()

    print(f"\nTraining complete. Best validation accuracy: {best_val_accuracy:.2f}%")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Evaluate on test set
    test_preds = []
    test_labels_list = []
    test_losses = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_losses.append(loss.item())

            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.cpu().tolist())
            test_labels_list.extend(labels.cpu().tolist())

    test_loss = sum(test_losses) / len(test_losses)
    test_acc = accuracy_score(test_labels_list, test_preds)
    test_recall = recall_score(test_labels_list, test_preds, average='macro')
    test_f1 = f1_score(test_labels_list, test_preds, average='macro')

    print(f"Test Loss: {test_loss:.4f} | Acc: {test_acc*100:.2f}% | Recall: {test_recall:.3f} | F1: {test_f1:.3f}")

    print("\n=== Per-Class Classification Report (Test Set) ===")
    target_names = [str(i) for i in range(10)]  # MNIST classes 0-9
    print(classification_report(test_labels_list, test_preds, target_names=target_names))

    return model, cal_loader

