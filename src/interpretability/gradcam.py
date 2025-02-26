import torch
import torch.nn.functional as F
from torch.nn import Module

class GradCAM:
    """
    Implements Grad-CAM (Gradient-weighted Class Activation Mapping) 
    for interpretability in CNNs.
    
    Args:
        model (torch.nn.Module): PyTorch model.
        target_layer (str): Layer name where gradients are captured.
    """

    def __init__(self, model: Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        """
        Hooks the gradients and activations for the target layer.
        """
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break

    def generate_cam(self, class_idx):
        """
        Computes the Grad-CAM heatmap for a given class index.

        Args:
            class_idx (int): Target class index.

        Returns:
            torch.Tensor: Heatmap tensor.
        """
        if self.gradients is None or self.activations is None:
            raise ValueError("Gradients or activations not captured. Ensure a forward and backward pass is performed.")

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32)

        for i, weight in enumerate(pooled_gradients):
            cam += weight * self.activations[0, i, :, :]

        cam = F.relu(cam)
        cam -= cam.min()
        cam /= cam.max() 
        return cam.cpu().detach()

    def compute(self, input_tensor, class_idx):
        """
        Computes Grad-CAM for a given input tensor and class index.

        Args:
            input_tensor (torch.Tensor): Input image tensor.
            class_idx (int): Target class index.

        Returns:
            torch.Tensor: Grad-CAM heatmap.
        """
        self.model.zero_grad()
        output = self.model(input_tensor.unsqueeze(0))
        target_output = output[:, class_idx]
        target_output.backward()
        return self.generate_cam(class_idx)
