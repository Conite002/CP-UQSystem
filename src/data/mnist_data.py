import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import os
import pickle


def get_mnist_loaders(
    batch_size=64,
    train_ratio=0.7,
    val_ratio=0.15,
    cal_ratio=0.15,
    seed=42,
    save_splits_path=None,
    load_splits_path=None
):
    """
    Loads MNIST and splits it into train, validation, calibration, and test sets.
    The three ratios (train_ratio, val_ratio, cal_ratio) must sum to 1.0.

    Args:
        batch_size (int): Batch size for DataLoaders.
        train_ratio (float): Fraction of the original training set used for training.
        val_ratio (float): Fraction of the original training set used for validation.
        cal_ratio (float): Fraction of the original training set used for calibration.
        seed (int): Random seed for reproducible shuffling.

    Returns:
        (train_loader, val_loader, cal_loader, test_loader)
    """

    # 1. Validate ratios
    total_ratio = train_ratio + val_ratio + cal_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            "train_ratio + val_ratio + cal_ratio must sum to 1.0, "
            f"but got {total_ratio:.2f}"
        )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train_dataset = torchvision.datasets.MNIST(
        root='../data/',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root='../data/',
        train=False,
        download=True,
        transform=transform
    )
    if load_splits_path is not None and os.path.exists(load_splits_path):
        with open(load_splits_path, "rb") as f:
            splits_dict = pickle.load(f)
        train_indices = splits_dict["train_indices"]
        val_indices   = splits_dict["val_indices"]
        cal_indices   = splits_dict["cal_indices"]
        print(f"[INFO] Loaded subset indices from {load_splits_path}")
    else:


        random.seed(seed)
        torch.manual_seed(seed)
    
        total_train_size = len(full_train_dataset)
        indices = list(range(total_train_size))
        random.shuffle(indices)
    
        train_size = int(train_ratio * total_train_size)
        val_size   = int(val_ratio * total_train_size)
        cal_size   = total_train_size - train_size - val_size
    
        train_indices = indices[:train_size]
        val_indices   = indices[train_size:train_size + val_size]
        cal_indices   = indices[train_size + val_size:]
    
        if save_splits_path is not None:
            splits_dict = {
                "train_indices": train_indices,
                "val_indices":   val_indices,
                "cal_indices":   cal_indices
            }
            os.makedirs(os.path.dirname(save_splits_path), exist_ok=True)
            with open(save_splits_path, "wb") as f:
                pickle.dump(splits_dict, f)
            print(f"[INFO] Saved subset indices to {save_splits_path}")
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset   = Subset(full_train_dataset, val_indices)
    cal_dataset   = Subset(full_train_dataset, cal_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    cal_loader   = DataLoader(cal_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, cal_loader, test_loader
