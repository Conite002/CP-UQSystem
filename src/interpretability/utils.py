import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

def visualize_gradcam(heatmap, image, title="Grad-CAM"):
    """
    Visualizes Grad-CAM heatmap on an image.

    Args:
        heatmap (torch.Tensor): Grad-CAM heatmap.
        image (torch.Tensor): Original image.
        title (str): Title for visualization.
    """
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (image.shape[-1], image.shape[-2]))  # Resize to match image size

    img = image.squeeze().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min()) 

    heatmap = np.uint8(255 * heatmap)  
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(title)
    plt.show()
