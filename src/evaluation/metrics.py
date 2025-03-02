def evaluate_model(model, dataloader, device):
    """
    Evaluates a PyTorch model on a given dataloader.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        dataloader (torch.utils.data.DataLoader): Dataloader for evaluation.
        device (torch.device): Device to run the evaluation (CPU/GPU). 

    Returns:
        dict: A dictionary containing accuracy, precision, recall, F1-score, and loss.
    """
    
    model.to(device)
    model.eval()

    total_samples = 0
    correct_predictions = 0
    total_loss = 0.0

    all_preds = []
    all_labels = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():  
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    results = {
        "accuracy": accuracy,
        "loss": total_loss / len(dataloader),
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    print(f"[EVALUATION] Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}, Loss: {results['loss']:.4f}")

    return results




import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def test_model(model, dataloader, device, is_ensemble=False):
    """
    Evaluates a model (single or ensemble) on the test dataset.

    Args:
        model (nn.Module or list of nn.Module): Trained PyTorch model or ensemble of models.
        dataloader (torch.utils.data.DataLoader): Test data loader.
        device (torch.device): Device to run inference on (CPU/GPU).
        is_ensemble (bool): Whether the model is an ensemble.

    Returns:
        dict: Accuracy, precision, recall, F1-score, and loss.
    """

    model.to(device)
    model.eval()  # Set to evaluation mode

    total_samples = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    all_model_preds = [[] for _ in range(len(model.models))] if is_ensemble else []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            if is_ensemble:
                # Get outputs from each model in the ensemble
                model_outputs = torch.stack([m(inputs) for m in model.models], dim=0)  # Shape: (num_models, batch, num_classes)
                avg_output = torch.mean(model_outputs, dim=0)  # Average logits

                # Store individual model predictions
                for i, model_out in enumerate(model_outputs):
                    _, model_pred = torch.max(model_out, 1)
                    all_model_preds[i].extend(model_pred.cpu().tolist())

            else:
                avg_output = model(inputs)

            loss = criterion(avg_output, labels)
            total_loss += loss.item()
            total_samples += labels.size(0)

            # Final ensemble or single model prediction
            _, final_pred = torch.max(avg_output, 1)
            all_preds.extend(final_pred.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Compute final metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    results = {
        "accuracy": accuracy,
        "loss": total_loss / len(dataloader),
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    print(f"\n[EVALUATION RESULTS] - {'ENSEMBLE' if is_ensemble else 'SINGLE MODEL'}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-score: {f1:.3f}")
    print(f"  Loss: {results['loss']:.4f}")

    # Print detailed predictions for first few samples
    print("\n=== Sample Predictions ===")
    for i in range(5):  # Show first 5 samples
        if is_ensemble:
            model_preds = [all_model_preds[m][i] for m in range(len(model.models))]
            print(f"Sample {i+1}:")
            print(f"  Model Predictions: {model_preds}")
        print(f"  Final Prediction: {all_preds[i]}")
        print(f"  True Label: {all_labels[i]}\n")

    return results

