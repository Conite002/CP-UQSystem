import torch
import torch.nn.functional as F
import torch.optim as optim
from torchcp.classification.predictor import SplitPredictor, ClassWisePredictor, ClusteredPredictor, WeightedPredictor
from predictors.split import SplitPredictor_CM
from torchcp.classification.score import APS, RAPS, SAPS, TOPK, KNN
from torchcp.classification.trainer import ConfTSTrainer
from transformers import set_seed
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from interpretability.gradcam import GradCAM  
from interpretability.utils import visualize_gradcam



def enable_mc_dropout(model):
    """
    Enables MC Dropout by setting dropout layers to training mode.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
    
def mc_dropout_forward(model, x_batch, mc_samples=10):
    """
    Runs MC Dropout on the model for x_batch.
    Returns the mean of the MC samples.
    """
    enable_mc_dropout(model)
    mc_logits = torch.stack([model(x_batch).detach() for _ in range(mc_samples)]).mean(dim=0)
    return mc_logits

def ensemble_forward(models, x_batch):
    """
    Runs ensemble forward pass on a list of models for x_batch.
    Returns the mean of the ensemble logits.
    """
    ensemble_logits = torch.stack([model(x_batch) for model in models]).mean(dim=0)
    return ensemble_logits

    
def gather_logits_labels(model, data_loader, device, mc_dropout=False, mc_samples=10, ensemble_models=None):
    """
    Passes the data through the model to gather logits & labels.
    Returns two tensors: (logits, labels).
    """
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            
            if mc_dropout:
                outputs = mc_dropout_forward(model, images, mc_samples)
            elif ensemble_models:
                outputs = ensemble_forward(ensemble_models, images)
            else:
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
    mc_dropout=False,
    mc_samples=10,
    ensemble_models=None,
    init_temperatures=[0.5, 1.0, 1.5],
    num_samples_to_show=5,
    use_interpretability=False

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

    score_methods = {
        "APS":  APS(score_type="softmax", randomized=True),
        "RAPS": RAPS(score_type="softmax", randomized=True, penalty=0.1, kreg=1),
        "SAPS": SAPS(score_type="softmax", randomized=True, weight=0.2),
        "TOPK": TOPK(score_type="softmax", randomized=True),
    }

    predictor_classes = {
        # "SplitPredictor": SplitPredictor,
        "SplitPredictor_CM": SplitPredictor_CM,
        # "ClassWisePredictor": ClassWisePredictor,
        # "ClusteredPredictor": ClusteredPredictor
    }

    results_list = []
                
    for temp in tqdm(init_temperatures, desc="Processing Initial Temperatures"):

        print(f"\n[INFO] Gathering logits for Temperature {temp} (Before TS)")
        cal_logits, cal_labels = gather_logits_labels(model, cal_loader, device, mc_dropout, mc_samples, ensemble_models)
        test_logits, test_labels = gather_logits_labels(model, test_loader, device, mc_dropout, mc_samples, ensemble_models)
        score_methods["KNN"] = KNN(features=cal_logits, labels=cal_labels, num_classes=num_classes, k=5, p=2)

        for score_name, score_obj in score_methods.items():
            for predictor_name, predictor_cls in predictor_classes.items():
                predictor = predictor_cls(score_function=score_obj, model=model)
                predictor.calibrate(cal_loader, alpha=alpha)
                result_dict = predictor.evaluate(test_loader)

                
                sample_images, sample_labels = next(iter(test_loader))
                sample_images = sample_images.to(device)
                pred_sets = predictor.predict(sample_images)

                print("\n=== Prediction Sets ===")
                for i, (pset, true_label) in enumerate(zip(pred_sets[:num_samples_to_show], sample_labels[:num_samples_to_show])):
                    predicted_classes = {idx for idx, val in enumerate(pset) if val.item() == 1}
                    print(f"Sample {i}: True Label={true_label.item()}, Prediction Set={predicted_classes}")
                    if use_interpretability:
                        interpretability = GradCAM(model, sample_images)
                        interpretability.show_maps()
                        visualize_gradcam(interpretability, sample_images, true_label)
                results_list.append({
                    "Temperature": temp,
                    "ScoreMethod": score_name,
                    "Predictor": predictor_name,
                    "CoverageRate": result_dict['coverage_rate'],
                    "AvgSetSize": result_dict['average_size'],
                })

                print(f"[RESULT] {score_name}-{predictor_name}: Coverage={result_dict['coverage_rate']:.4f}, "
                      f"Avg Set Size={result_dict['average_size']:.2f}")
                print('')
                results_df = pd.DataFrame(results_list)
                results_df.to_csv('src/results/cp_results.csv', index=False)
                print("\n[INFO] Conformal Calibration Results Saved to src/results/cp_results.csv")

    results_df = pd.DataFrame(results_list)
    results_df.to_csv('src/results/cp_results.csv', index=False)
    print("\n[INFO] Conformal Calibration Results Saved to src/results/cp_results.csv")

    return results_df