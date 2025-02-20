from src.torchcp_integration.conformal_utils import posthoc_conformal_calibration
from src.data.mnist_data import get_mnist_loaders 
from src.models.base_model import SimpleMNISTModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMNISTModel()
model.load_state_dict(torch.load("src/checkpoints/baseline_mnist.pth"))
train_loader, val_loader, cal_loader, test_loader = get_mnist_loaders(
    batch_size=64,
    load_splits_path="data/mnist_splits.pkl"
)

posthoc_conformal_calibration(
    model,              
    cal_loader,         
    test_loader,        
    device=device,
    alpha=0.1,
    seed=42,
    num_samples_to_show=5,
    num_epochs=10

)