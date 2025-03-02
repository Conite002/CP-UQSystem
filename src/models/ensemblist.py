import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_model import SimpleMNISTModel

class MNISTEnsemble(nn.Module):
    def __init__(self, num_models=5, num_classes=10):
        """
        Ensemble of multiple SimpleMNISTModel models.
        
        Args:
            num_models (int): Number of models in the ensemble.
            num_classes (int): Number of output classes.
        """
        super(MNISTEnsemble, self).__init__()
        self.models = nn.ModuleList([SimpleMNISTModel(num_classes) for _ in range(num_models)])
    
    def forward(self, x):
        """
        Forward pass: Averages the outputs from all models.
        """
        outputs = torch.stack([model(x) for model in self.models], dim=0)
        ensemble_output = torch.mean(outputs, dim=0)  # Average the logits
        return ensemble_output

    def load_model_weights(self, weight_paths):
        """
        Load pre-trained weights for each model in the ensemble.

        Args:
            weight_paths (list of str): Paths to the saved model weights.
        """
        assert len(weight_paths) == len(self.models), "Number of weight paths must match number of models!"
        for model, path in zip(self.models, weight_paths):
            model.load_state_dict(torch.load(path))
            model.eval()  # Set to evaluation mode
        print(f"[INFO] Loaded {len(weight_paths)} models into the ensemble.")

