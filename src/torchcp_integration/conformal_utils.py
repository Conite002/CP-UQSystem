import torch
import torch.nn.functional as F
import torch.optim as optim
from torchcp.classification.predictor import SplitPredictor, ClassWisePredictor, ClusteredPredictor, WeightedPredictor
from torchcp.classification.score import APS, RAPS, SAPS, TOPK, KNN
from torchcp.classification.trainer import ConfTSTrainer
from transformers import set_seed



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
    num_samples_to_show=5,
    seed=42,
    num_classes=10,
    model_calibrate_path="src/checkpoints/calibrated_model.pth",
    num_epochs=10
):

    """
    Applies multiple TorchCP score functions & predictor classes to a given model,
    using the provided calibration & test loaders for post-hoc conformal calibration.

    Args:
        model (nn.Module): Trained classification model.
        cal_loader (DataLoader): Calibration set loader (held-out data for conformal fitting).
        test_loader (DataLoader): Test set loader (for final coverage evaluation).
        device (torch.device or None): Device for inference (CPU/GPU). If None, auto-detect.
        alpha (float): Significance level for conformal methods (1 - coverage).
        num_samples_to_show (int): How many test samples to display prediction sets for.

    Returns:
        None. Prints coverage, average set size, and example prediction sets for each method.
    """
    set_seed(seed=seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    cal_logits, cal_labels   = gather_logits_labels(model, cal_loader, device)
    test_logits, test_labels = gather_logits_labels(model, test_loader, device)

    print(f"[INFO] Calibration set size: {len(cal_labels)} | Test set size: {len(test_labels)}")
    print(f"[INFO] Using alpha={alpha:.2f}, target coverage={1 - alpha:.2f}")

    score_methods = {
        "APS":  APS(score_type="softmax", randomized=True),
        "RAPS": RAPS(score_type="softmax", randomized=True, penalty=0.1, kreg=1),
        "SAPS": SAPS(score_type="softmax", randomized=True, weight=0.2),
        "TOPK": TOPK(score_type="softmax", randomized=True),
        "KNN": KNN(features=cal_logits, labels=cal_labels, num_classes=num_classes, k=5, p=2)
    }

    predictor_classes = {
        "SplitPredictor": SplitPredictor,
        "ClassWisePredictor": ClassWisePredictor,
        "ClusteredPredictor": ClusteredPredictor,
        # "WeightedPredictor": WeightedPredictor
    }

    init_temperature=1.5
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainer = ConfTSTrainer(
        temperature=init_temperature,
        alpha=0.1,
        model=model,
        optimizer=optimizer,
        device=device,
        verbose=True
    )
    trainer.train(cal_loader, num_epochs=num_epochs)
    trainer.save_checkpoint(save_path=model_calibrate_path, epoch=num_epochs)
    for score_name, score_obj in score_methods.items():
        for predictor_name, predictor_cls in predictor_classes.items():
            print(f"\n=== Score: {score_name}, Predictor: {predictor_name} ===")

            predictor = predictor_cls(score_function=score_obj, model=model)
            predictor.calibrate(cal_loader, alpha=alpha)
            result_dict = predictor.evaluate(test_loader)
            print(f"Coverage Rate: {result_dict['coverage_rate']:.4f}")
            print(f"Average Set Size: {result_dict['average_size']:.4f}")
            
            sample_images, sample_labels = next(iter(test_loader))
            sample_images = sample_images.to(device) 
            # Generate prediction sets
            pred_sets = predictor.predict(sample_images)
            
            # Display the results
            for i, (pset, true_label) in enumerate(zip(pred_sets[:num_samples_to_show], sample_labels[:num_samples_to_show])):
                print(f"  Sample {i}: True Label={true_label.item()}, Prediction Set={pset}")


    print("\nAfter Temperature Scaling:")

    for score_name, score_obj in score_methods.items():
        for predictor_name, predictor_cls in predictor_classes.items():
            print(f"\n=== Score: {score_name}, Predictor: {predictor_name} ===")
            predictor = predictor_cls(score_function=score_obj, model=trainer.model)
            predictor.calibrate(cal_loader, alpha=alpha)
            result_dict = predictor.evaluate(test_loader)
            print(f"Coverage Rate: {result_dict['coverage_rate']:.4f}")
            print(f"Average Set Size: {result_dict['average_size']:.4f}")

            sample_images, sample_labels = next(iter(test_loader))
            sample_images = sample_images.to(device) 
            #Generate prediction sets
            pred_sets = predictor.predict(sample_images)
            for i, (pset, true_label) in enumerate(zip(pred_sets[:num_samples_to_show], sample_labels[:num_samples_to_show])):
                print(f"  Sample {i}: True Label={true_label.item()}, Prediction Set={pset}")


