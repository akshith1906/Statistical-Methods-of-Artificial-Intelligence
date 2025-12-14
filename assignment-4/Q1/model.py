import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class MultiTaskCNN(nn.Module):
    """
    Multi-task CNN for Fashion-MNIST classification and ink regression.
    
    - Shared convolutional backbone.
    - Two separate heads:
        1. Classification head (10 classes)
        2. Regression head (1 scalar 'ink' value)
    """
    def __init__(self, dropout_rate: float = 0.25):
        super(MultiTaskCNN, self).__init__()
        
        # --- Shared Backbone ---
        
        # Block 1: 1x28x28 -> 16x14x14
        self.shared_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=16, 
                kernel_size=3, 
                padding=1
            ), # 1x28x28 -> 16x28x28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 16x28x28 -> 16x14x14
        )
        
        # Block 2: 16x14x14 -> 32x7x7
        self.shared_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, 
                out_channels=32, 
                kernel_size=3, 
                padding=1
            ), # 16x14x14 -> 32x14x14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 32x14x14 -> 32x7x7
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # This is the flattened size of the feature map after the shared blocks
        self.flatten_size = 32 * 7 * 7 # 1568

        # --- Heads ---

        # 1. Classification Head
        self.classifier_head = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, 10) # 10 classes
        )
        
        # 2. Regression Head
        self.regressor_head = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # 1 scalar output
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of the multi-task model.
        
        Args:
            x (torch.Tensor): Input batch of images, shape [B, 1, 28, 28]
            return_features (bool): If True, returns intermediate feature maps
                                    for visualization (Assignment Part 4).
        
        Returns:
            A tuple containing:
            - logits (torch.Tensor): Classification logits, shape [B, 10]
            - ink_pred (torch.Tensor): Regression prediction, shape [B]
            - features (tuple, optional): (f1, f2) if return_features=True
        """
        # Pass through shared backbone
        f1 = self.shared_block1(x)
        f2 = self.shared_block2(f1)
        
        # Flatten and apply dropout
        x_flat = torch.flatten(f2, 1) # Shape [B, 1568]
        x_flat = self.dropout(x_flat)
        
        # Pass through separate heads
        logits = self.classifier_head(x_flat)
        ink_pred = self.regressor_head(x_flat)
        
        # Squeeze regression output from [B, 1] to [B] to match target shape
        ink_pred = ink_pred.squeeze(-1) 
        
        if return_features:
            return logits, ink_pred, (f1, f2)
        
        return logits, ink_pred
