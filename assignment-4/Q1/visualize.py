import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

# --- Local Imports ---
from model import MultiTaskCNN
from fashion_mnist_data import get_dataloaders
from utils import setup_logging, OUTPUT_DIR

# =============================================================================
# 4. Feature Map Visualization
# =============================================================================

MODEL_RUN_NAME = "run_1_lambda_1_1" 

MODEL_PATH = os.path.join(OUTPUT_DIR, f"{MODEL_RUN_NAME}.pth")
NUM_IMAGES_TO_VISUALIZE = 3
FASHION_MNIST_MEAN = 0.2860 # From fashion_mnist_data.py
FASHION_MNIST_STD = 0.3530 # From fashion_mnist_data.py

# Class names for plotting
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Denormalizes a tensor image for plotting."""
    tensor = tensor * FASHION_MNIST_STD + FASHION_MNIST_MEAN
    # Clip to [0, 1] range just in case
    tensor = torch.clamp(tensor, 0, 1)
    # Convert to [H, W, C] numpy format
    return tensor.squeeze(0).cpu().numpy()

def plot_feature_maps(
    feature_maps: torch.Tensor, 
    title: str, 
    save_path: str,
    grid_size: tuple = (4, 4)
):
    """
    Plots a grid of feature maps.
    
    Args:
        feature_maps (torch.Tensor): A [C, H, W] tensor of feature maps
        title (str): Title for the whole plot
        save_path (str): Path to save the figure
        grid_size (tuple): (rows, cols) for the plot grid.
    """
    num_maps = feature_maps.size(0)
    rows, cols = grid_size
    
    # Clamp to the number of available maps if grid is too large
    if rows * cols > num_maps:
        cols = int(np.ceil(num_maps / rows))
        if rows * cols < num_maps:
            rows += 1
            
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    fig.suptitle(title, fontsize=16, y=1.02)
    
    for i in range(rows * cols):
        ax = axes.flat[i]
        if i < num_maps:
            # Detach from graph and move to CPU
            fm = feature_maps[i].detach().cpu().numpy()
            ax.imshow(fm, cmap='viridis')
            ax.set_title(f'Map {i+1}')
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    logging.info(f"Saved feature map plot to {save_path}")

def run_visualization():
    """
    Loads a trained model and visualizes its feature maps for a few test images.
    """
    setup_logging(log_file_name="visualize.log")
    
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found at {MODEL_PATH}")
        logging.error("Please run train.py first to generate a model file.")
        return

    # 1. Load Model
    logging.info(f"Loading model from {MODEL_PATH}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We need to know the dropout rate the model was saved with.
    model = MultiTaskCNN(dropout_rate=0.25).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set model to evaluation mode

    # 2. Get Test Data
    logging.info("Loading test data...")
    # We only need the test_loader
    _, _, test_loader = get_dataloaders(batch_size=NUM_IMAGES_TO_VISUALIZE, val_split=0.1)
    
    images, labels, inks = next(iter(test_loader))
    images, labels, inks = images.to(device), labels.to(device), inks.to(device)
    
    # 3. Get Feature Maps
    logging.info("Running model and extracting feature maps...")
    # Call model with return_features=True
    logits, ink_preds, (features_b1, features_b2) = model(images, return_features=True)
    
    preds_class = torch.argmax(logits, dim=1)
    
    for i in range(NUM_IMAGES_TO_VISUALIZE):
        logging.info(f"\nVisualizing image {i+1}/{NUM_IMAGES_TO_VISUALIZE}")
        
        img_tensor = images[i]
        label_true = labels[i].item()
        label_pred = preds_class[i].item()
        ink_true = inks[i].item()
        ink_pred = ink_preds[i].item()
        
        logging.info(f"  True Label: {CLASS_NAMES[label_true]} ({label_true})")
        logging.info(f"  Pred Label: {CLASS_NAMES[label_pred]} ({label_pred})")
        logging.info(f"  True Ink: {ink_true:.4f}")
        logging.info(f"  Pred Ink: {ink_pred:.4f}")
        
        # --- Plot Original Image ---
        img_np = denormalize(img_tensor)
        plt.figure(figsize=(4, 4))
        plt.imshow(img_np, cmap='gray')
        plt.title(f"Image {i+1} - True: {CLASS_NAMES[label_true]}\nPred: {CLASS_NAMES[label_pred]}")
        plt.axis('off')
        img_save_path = os.path.join(OUTPUT_DIR, f"{MODEL_RUN_NAME}_img_{i+1}_original.png")
        plt.savefig(img_save_path)
        plt.close()
        
        # --- Plot Block 1 Feature Maps ---
        # features_b1 shape is [B, 16, 14, 14]. We select [i, :, :, :]
        f1 = features_b1[i] 
        plot_feature_maps(
            f1, 
            title=f"Image {i+1}: Block 1 Feature Maps (16 maps)",
            save_path=os.path.join(OUTPUT_DIR, f"{MODEL_RUN_NAME}_img_{i+1}_block1_maps.png"),
            grid_size=(4, 4)
        )
        
        # --- Plot Block 2 Feature Maps ---
        # features_b2 shape is [B, 32, 7, 7]. We select [i, :, :, :]
        f2 = features_b2[i]
        plot_feature_maps(
            f2, 
            title=f"Image {i+1}: Block 2 Feature Maps (32 maps)",
            save_path=os.path.join(OUTPUT_DIR, f"{MODEL_RUN_NAME}_img_{i+1}_block2_maps.png"),
            grid_size=(4, 8) # 4 rows, 8 cols to fit 32 maps
        )

    logging.info("\nVisualization finished. Check the 'output' folder for your plots.")

if __name__ == "__main__":
    run_visualization()
