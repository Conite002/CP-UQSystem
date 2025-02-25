import torch
import torch.nn.functional as F
import torch.optim as optim
from torchcp.classification.predictor import SplitPredictor, ClassWisePredictor, ClusteredPredictor, WeightedPredictor
from torchcp.classification.score import APS, RAPS, SAPS, TOPK, KNN
from torchcp.classification.trainer import ConfTSTrainer
from transformers import set_seed
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def gather_logits_labels(model, data_loader, device):
    """
    Passes the data through the model to gather logits & labels.
    Returns two tensors: (logits, labels).
    """
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)           
            all_logits.append(outputs.cpu())  
            all_labels.append(labels)
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return logits, labels

def posthoc_conformal_calibration(
    model,
    cal_loader,
    test_loader,
    device=None,
    alpha=0.1,
    seed=42,
    num_classes=10,
    model_calibrate_path="src/checkpoints/calibrated_model.pth",
    num_epochs=10,
    init_temperatures=[0.5, 1.0, 1.5],
    num_samples_to_show=5
):
    """
    Runs multiple TorchCP score & predictor methods for each temperature in init_temperatures.
    Returns a Pandas DataFrame with coverage & set-size results for easy plotting.

    Args:
        model (nn.Module): Base classification model.
        cal_loader, test_loader: DataLoaders for calibration/test sets.
        device (torch.device or None): CPU/GPU. If None, auto-detect.
        alpha (float): Significance level (1 - coverage).
        seed (int): Random seed for reproducibility.
        num_classes (int): e.g., 10 for MNIST.
        model_calibrate_path (str): Where to save temperature-scaled model.
        num_epochs (int): # epochs to run ConfTSTrainer on cal set.
        init_temperatures (list): List of initial temps to test, e.g. [0.5, 1.0, 1.5].

    Returns:
        pd.DataFrame: Columns = [
            'Temperature', 'Phase', 'ScoreMethod', 'Predictor', 
            'CoverageRate', 'AvgSetSize'
        ]
    """
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Score methods
    score_methods = {
        "APS":  APS(score_type="softmax", randomized=True),
        "RAPS": RAPS(score_type="softmax", randomized=True, penalty=0.1, kreg=1),
        "SAPS": SAPS(score_type="softmax", randomized=True, weight=0.2),
        "TOPK": TOPK(score_type="softmax", randomized=True),
    }

    predictor_classes = {
        "SplitPredictor": SplitPredictor,
        "ClassWisePredictor": ClassWisePredictor,
        "ClusteredPredictor": ClusteredPredictor
    }

    results_list = []

    for temp in init_temperatures:
        phase = "BeforeTS"
        cal_logits, cal_labels = gather_logits_labels(model, cal_loader, device)
        test_logits, test_labels = gather_logits_labels(model, test_loader, device)
        score_methods["KNN"] = KNN(features=cal_logits, labels=cal_labels, num_classes=num_classes, k=2, p=2)
    
        for score_name, score_obj in score_methods.items():
            for predictor_name, predictor_cls in predictor_classes.items():
                predictor = predictor_cls(score_function=score_obj, model=model)
                predictor.calibrate(cal_loader, alpha=alpha)
                result_dict = predictor.evaluate(test_loader)
                
                coverage_rate = result_dict['coverage_rate']
                avg_size      = result_dict['average_size']
                
                sample_images, sample_labels = next(iter(test_loader))
                sample_images = sample_images.to(device)
                pred_sets = predictor.predict(sample_images)

                print("\n=== Prediction Sets ===")
                for i, (pset, true_label) in enumerate(zip(pred_sets[:num_samples_to_show], sample_labels[:num_samples_to_show])):
                    predicted_classes = {idx for idx, val in enumerate(pset) if val.item() == 1}
                    print(f"Sample {i}: True Label={true_label.item()}, Prediction Set={predicted_classes}")

                predicted_classes = [pset.argmax().item() for pset in pred_sets]
                accuracy = accuracy_score(sample_labels.cpu().numpy(), predicted_classes)
                f1 = f1_score(sample_labels.cpu().numpy(), predicted_classes, average="weighted")
                logits = model(sample_images) 
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(logits, sample_labels.to(device)).item()
                results_list.append({
                    "Temperature": temp,
                    "Phase": phase,
                    "ScoreMethod": score_name,
                    "Predictor": predictor_name,
                    "CoverageRate": result_dict['coverage_rate'],
                    "AvgSetSize": result_dict['average_size'],
                    "Accuracy": accuracy,
                    "F1Score": f1,
                    "Loss": loss
                })

                print(f"[RESULT] {score_name}-{predictor_name}: Coverage={result_dict['coverage_rate']:.4f}, "
                      f"Avg Set Size={result_dict['average_size']:.2f}, Accuracy={accuracy:.4f}, "
                      f"F1-Score={f1:.4f}")
                print('')
                
    for temp in tqdm(init_temperatures, desc="Processing Initial Temperatures"):
        model_ts = type(model)() 
        model_ts.load_state_dict(model.state_dict())
        model_ts.to(device).eval()

        optimizer = optim.Adam(model_ts.parameters(), lr=0.001)
        trainer = ConfTSTrainer(
            temperature=temp,
            alpha=alpha,
            model=model_ts,
            optimizer=optimizer,
            device=device,
            verbose=False
        )
        # Train on calibration set
        trainer.train(cal_loader, num_epochs=num_epochs)
        trainer.save_checkpoint(save_path=model_calibrate_path, epoch=num_epochs)

        phase = "AfterTS"
        print(f"\n[INFO] Gathering logits for Temperature {temp} (Before TS)")
        cal_logits, cal_labels = gather_logits_labels(trainer.model, cal_loader, device)
        test_logits, test_labels = gather_logits_labels(trainer.model, test_loader, device)
        score_methods["KNN"] = KNN(features=cal_logits, labels=cal_labels, num_classes=num_classes, k=5, p=2)

        for score_name, score_obj in score_methods.items():
            for predictor_name, predictor_cls in predictor_classes.items():
                predictor = predictor_cls(score_function=score_obj, model=trainer.model)
                predictor.calibrate(cal_loader, alpha=alpha)
                result_dict = predictor.evaluate(test_loader)

                coverage_rate = result_dict['coverage_rate']
                avg_size      = result_dict['average_size']

                print(f"Original shape of test_logits: {test_logits.shape}")
                
                sample_images, sample_labels = next(iter(test_loader))
                sample_images = sample_images.to(device)
                pred_sets = predictor.predict(sample_images)

                print("\n=== Prediction Sets ===")
                for i, (pset, true_label) in enumerate(zip(pred_sets[:num_samples_to_show], sample_labels[:num_samples_to_show])):
                    predicted_classes = {idx for idx, val in enumerate(pset) if val.item() == 1}
                    print(f"Sample {i}: True Label={true_label.item()}, Prediction Set={predicted_classes}")

                
                predicted_classes = [pset.argmax().item() for pset in pred_sets]
                accuracy = accuracy_score(sample_labels.cpu().numpy(), predicted_classes)
                f1 = f1_score(sample_labels.cpu().numpy(), predicted_classes, average="weighted")
        
                logits = trainer.model(sample_images) 
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(logits, sample_labels.to(device)).item()
                results_list.append({
                    "Temperature": temp,
                    "Phase": phase,
                    "ScoreMethod": score_name,
                    "Predictor": predictor_name,
                    "CoverageRate": result_dict['coverage_rate'],
                    "AvgSetSize": result_dict['average_size'],
                    "Accuracy": accuracy,
                    "F1Score": f1,
                    "Loss": loss
                })

                print(f"[RESULT] {score_name}-{predictor_name}: Coverage={result_dict['coverage_rate']:.4f}, "
                      f"Avg Set Size={result_dict['average_size']:.2f}, Accuracy={accuracy:.4f}, "
                      f"F1-Score={f1:.4f}")
                print('')
                results_df = pd.DataFrame(results_list)
                results_df.to_csv('src/results/cp_results.csv', index=False)
                print("\n[INFO] Conformal Calibration Results Saved to src/results/cp_results.csv")

    results_df = pd.DataFrame(results_list)
    results_df.to_csv('src/results/cp_results.csv', index=False)
    print("\n[INFO] Conformal Calibration Results Saved to src/results/cp_results.csv")

    return results_df