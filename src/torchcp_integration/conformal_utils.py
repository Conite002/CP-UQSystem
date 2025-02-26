import torch
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append('src/torchcp_integration/')
sys.path.append('src/')
sys.path.append('src/torchcp_integration/predictors')
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
    init_temperatures=[1.5],
    num_samples_to_show=5,
    use_interpretability=False

):

    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if ensemble_models is not None:
        for model  in ensemble_models:
            model.eval()

    print(f"isinstance(model, list)  : {isinstance(ensemble_models, list) }")
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
        "ClusteredPredictor": ClusteredPredictor
    }

    results_list = []
    for temp in tqdm(init_temperatures, desc="Processing"):

        print(f"\n[INFO] Gathering logits ")
        cal_logits, cal_labels = gather_logits_labels(model, cal_loader, device, mc_dropout, mc_samples, ensemble_models)
        test_logits, test_labels = gather_logits_labels(model, test_loader, device, mc_dropout, mc_samples, ensemble_models)
        score_methods["KNN"] = KNN(features=cal_logits, labels=cal_labels, num_classes=num_classes, k=10, p=2)

        for score_name, score_obj in score_methods.items():
            for predictor_name, predictor_cls in predictor_classes.items():
                if predictor_name == "SplitPredictor_CM":
                    predictor = predictor_cls(score_function=score_obj, model=ensemble_models, use_mc_dropout=mc_dropout)
                else :
                    predictor = predictor_cls(score_function=score_obj, model=model)
                    
                predictor.calibrate(cal_loader, alpha=alpha)
                result_dict = predictor.evaluate(test_loader)

                
                sample_images, sample_labels = next(iter(test_loader))
                sample_images = sample_images.to(device)
                pred_sets = predictor.predict(sample_images)

                print(f"\n=== Prediction Sets {score_name} ===")
                for i, (pset, true_label) in enumerate(zip(pred_sets[:num_samples_to_show], sample_labels[:num_samples_to_show])):
                    predicted_classes = {idx for idx, val in enumerate(pset) if val.item() == 1}
                    print(f"Sample {i}: True Label={true_label.item()}, Prediction Set={predicted_classes}")

                sample_images.requires_grad = True
                model.train() 
                output = model(sample_images)
                class_idx = sample_labels[0].item()
                model.zero_grad()
                output[:, class_idx].sum().backward(retain_graph=True) 
                if use_interpretability:
                    grad_cam = GradCAM(model, target_layer="conv2")
                    heatmap = grad_cam.generate_cam(class_idx)
                
                if heatmap is not None:
                    visualize_gradcam(heatmap, sample_images[0], title=f"Grad-CAM for Sample {sample_labels[0].item()}")
            

        
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