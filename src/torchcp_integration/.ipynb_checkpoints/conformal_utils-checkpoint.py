# src/torchcp_integration/conformal_mnist.py

import torch
import torch.nn.functional as F
from torchcp.classification import APS
from src.models.base_model import SimpleMNISTModel
from src.data.mnist_data import get_mnist_loaders

def posthoc_conformal_mnist(alpha=0.1, batch_size=64):
    # alpha = 0.1 means we want 90% coverage

    # Load data
    train_loader, test_loader = get_mnist_loaders(batch_size)
    
    # Split train data into train+cal (for demonstration, do it properly in practice)
    # For now, assume we do post-hoc on test set or a subset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMNISTModel().to(device)
    model.load_state_dict(torch.load("baseline_mnist.pth"))
    model.eval()

    # Prepare data for calibration
    # Typically you'd have a separate calibration loader; here we just reuse test_loader
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            logits_list.append(outputs.cpu())
            labels_list.append(labels)

    logits_calib = torch.cat(logits_list, dim=0)
    labels_calib = torch.cat(labels_list, dim=0)

    # Fit APS (Adaptive Prediction Sets)
    aps = APS(alpha=alpha, randomize=True)
    aps.fit(logits_calib, labels_calib)

    # Evaluate coverage on the same set (or a separate set)
    coverage, size = aps.evaluate(logits_calib, labels_calib)
    print(f"Coverage: {coverage:.3f}, Avg Set Size: {size:.3f}")

    # Example: Generate prediction sets for a single batch
    # (In practice, you'd run inference on your new data)
    sample_logits = logits_calib[:10]
    sets = aps.predict(sample_logits)
    for i, s in enumerate(sets):
        print(f"Sample {i} => Prediction Set: {s}")
