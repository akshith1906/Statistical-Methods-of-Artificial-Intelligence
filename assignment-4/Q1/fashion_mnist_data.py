import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from typing import Tuple, Optional
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
from utils import setup_logging, OUTPUT_DIR 


#scp -r "C:\Users\akshith\Desktop\College\Semester 7\SMAI\Assignment_4\Q1" srinath@10.2.36.243:~/Desktop/SMAI_Q1
# --- Configuration ---
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1 # 10% for validation
NUM_WORKERS = 2
# =============================================================================
# 1. Custom Dataset Class
# =============================================================================
class FashionMNISTDataset(Dataset):
    """
    Custom Dataset for Fashion-MNIST.
    Wraps an existing dataset to return (image, class_label, ink_target)
    """
    def __init__(self, base_dataset: Dataset, transform: Optional[transforms.Compose] = None):
        """
        Args:
            base_dataset (Dataset): The base dataset to wrap (e.g., a Subset or FashionMNIST).
            transform (transforms.Compose, optional): The transformations to apply.
        """
        self.base_dataset = base_dataset
        self.transform = transform
        
        # We need a separate ToTensor() to calculate the ink value
        # *before* normalization and augmentations on the PIL image.
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Fetches the data item at the given index.
        """
        # 1. Get the raw data (PIL Image, integer label) from the base dataset
        image, label = self.base_dataset[idx]

        # 2. Calculate the 'ink target' (average pixel intensity)
        # We convert the PIL image to a [0, 1] tensor *first* to calculate its mean.
        # This is done *before* any normalization.
        # We keep it as a tensor for consistency.
        ink_target = self.to_tensor(image).mean()

        # 3. Apply the full transformation pipeline (augs, ToTensor, Normalize)
        # The 'image' variable is still the original PIL Image here.
        if self.transform:
            image = self.transform(image)
        else:
            # If no transform is provided, at least convert to tensor
            image = self.to_tensor(image)
        
        # Return ink_target as a 1-element tensor to match model output
        return image, label, ink_target.float()

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.base_dataset)

# =============================================================================
# 2. Helper Functions for Data Loading
# =============================================================================

def get_mean_std() -> Tuple[float, float]:
    """
    Calculates the mean and standard deviation of the Fashion-MNIST training set.
    """
    logging.info("Calculating mean and std of Fashion-MNIST training data...")
    temp_train_set = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transforms.ToTensor()
    )
    
    temp_loader = DataLoader(temp_train_set, batch_size=len(temp_train_set), shuffle=False)
    images, _ = next(iter(temp_loader))
    
    mean = images.mean().item()
    std = images.std().item()
    
    logging.info(f"Calculated Mean: {mean:.4f}")
    logging.info(f"Calculated Std: {std:.4f}")
    return mean, std

def get_transforms(
    mean: float, std: float
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Defines the training and validation/test transforms.
    """
    # Transform for the training set: includes light augmentations
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),      # Small rotation
        transforms.RandomCrop(28, padding=4), # Random crop
        transforms.RandomHorizontalFlip(p=0.5), # Added horizontal flip
        transforms.ToTensor(),              # Convert PIL Image to tensor [0, 1]
        transforms.Normalize((mean,), (std,)) # Normalize
    ])

    # Transform for validation and test sets: no augmentations
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),              # Convert PIL Image to tensor [0, 1]
        transforms.Normalize((mean,), (std,)) # Normalize
    ])
    
    return train_transform, val_test_transform

def load_fashion_data(
    val_split_ratio: float, 
    train_transform: transforms.Compose, 
    val_test_transform: transforms.Compose
) -> Tuple[FashionMNISTDataset, FashionMNISTDataset, FashionMNISTDataset]:
    """
    Loads, splits, and wraps the Fashion-MNIST dataset.
    """
    logging.info("Loading and splitting data...")
    # Load the raw training data (as PIL Images)
    full_train_dataset_raw = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True
    )
    
    # Load the raw test data (as PIL Images)
    test_dataset_raw = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True
    )

    # --- Split training set into train and val ---
    train_size = int((1.0 - val_split_ratio) * len(full_train_dataset_raw))
    val_size = len(full_train_dataset_raw) - train_size
    
    train_subset, val_subset = random_split(
        full_train_dataset_raw, 
        [train_size, val_size],
    )
    
    logging.info(f"Total training images: {len(full_train_dataset_raw)}")
    logging.info(f"New training split size: {len(train_subset)}")
    logging.info(f"Validation split size: {len(val_subset)}")
    logging.info(f"Test split size: {len(test_dataset_raw)}")

    # --- Wrap raw subsets with our custom Dataset class ---
    train_dataset = FashionMNISTDataset(train_subset, transform=train_transform)
    val_dataset = FashionMNISTDataset(val_subset, transform=val_test_transform)
    test_dataset = FashionMNISTDataset(test_dataset_raw, transform=val_test_transform)

    return train_dataset, val_dataset, test_dataset

# =============================================================================
# 3. Main Data Setup Function
# =============================================================================

def get_dataloaders(
    batch_size: int, 
    val_split: float
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Orchestrates data loading, processing, and DataLoader creation.
    
    Returns:
        A tuple of (train_loader, val_loader, test_loader)
    """
    # 1. Get stats and transforms
    mean, std = get_mean_std()
    train_transform, val_test_transform = get_transforms(mean, std)
    
    # 2. Load and split Datasets
    train_dataset, val_dataset, test_dataset = load_fashion_data(
        val_split_ratio=val_split,
        train_transform=train_transform,
        val_test_transform=val_test_transform
    )
    
    # 3. Create DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True # Speeds up data transfer to GPU
    )
    
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    logging.info("\nDataLoaders created successfully.")
    return train_loader, val_loader, test_loader


# =============================================================================
# 4. Main execution (This is where you can make your changes)
# =============================================================================
def main():
    """
    Main function to run the data loading and verification.
    """
    # Setup logging as the first step
    setup_logging(log_file_name="data_preprocessing.log")
    # --- 1. Get DataLoaders ---
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        val_split=VALIDATION_SPLIT
    )
    
    # --- 2. Verify one batch of data ---
    logging.info("\n" + "="*30)
    logging.info("Verifying one batch from train_loader...")
    images, labels, inks = next(iter(train_loader))
    
    logging.info(f"  Images batch shape: {images.shape}")
    logging.info(f"  Labels batch shape: {labels.shape}")
    logging.info(f"  Inks batch shape: {inks.shape}")
    
    logging.info(f"\n  Example label: {labels[0].item()}")
    logging.info(f"  Example ink value (tensor): {inks[0].item():.4f}")
    logging.info(f"  Image tensor min: {images.min():.4f}")
    logging.info(f"  Image tensor max: {images.max():.4f}")
    logging.info(f"  Image tensor mean (post-norm): {images.mean():.4f}")
    logging.info("="*30 + "\n")
    
    logging.info("Data preprocessing script finished. Ready to be imported.")


if __name__ == "__main__":
    main()


