import torch
import torch.nn as nn
import torch.optim as optim
import logging
import wandb 
from tqdm import tqdm # For nice progress bars
import os
from typing import Tuple

# --- Local Imports ---
# Make sure fashion_mnist_data.py and model.py are in the same directory
from fashion_mnist_data import get_dataloaders
from model import MultiTaskCNN
from utils import setup_logging, OUTPUT_DIR

# =============================================================================
# 3. Hyperparameter Tuning and wandb Logging
# =============================================================================

# This dictionary holds all hyperparameters and will be logged by wandb
config = {
    "project_name": "fashion-mnist-multitask",
    "run_name": "run_5_lambda_0.25_1", 
    "epochs": 10,
    "batch_size": 64,
    "validation_split": 0.1,
    "learning_rate": 0.001,
    "optimizer": "Adam", # "Adam" or "SGD"
    "dropout_rate": 0.25,
    "lambda1": 0.25, # Weight for classification loss
    "lambda2": 1.0, # Weight for regression loss
}

def train_one_epoch(
    model: nn.Module, 
    loader: torch.utils.data.DataLoader, 
    optimizer: optim.Optimizer, 
    criterion_class: nn.Module, 
    criterion_reg: nn.Module, 
    lambda1: float, 
    lambda2: float, 
    device: torch.device
) -> Tuple[float, float, float]:
    """Runs one full training epoch."""
    model.train()
    total_loss, total_class_loss, total_reg_loss = 0, 0, 0
    
    # Use tqdm for a progress bar
    for images, labels, inks in tqdm(loader, desc="Training"):
        images, labels, inks = images.to(device), labels.to(device), inks.to(device)
        
        # 1. Forward pass
        logits_pred, inks_pred = model(images)
        
        # 2. Calculate joint loss
        loss_class = criterion_class(logits_pred, labels)
        loss_reg = criterion_reg(inks_pred, inks)
        loss = (lambda1 * loss_class) + (lambda2 * loss_reg)
        
        # 3. Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_class_loss += loss_class.item()
        total_reg_loss += loss_reg.item()
        
    avg_loss = total_loss / len(loader)
    avg_class_loss = total_class_loss / len(loader)
    avg_reg_loss = total_reg_loss / len(loader)
    
    return avg_loss, avg_class_loss, avg_reg_loss

def validate_one_epoch(
    model: nn.Module, 
    loader: torch.utils.data.DataLoader, 
    criterion_class: nn.Module, 
    criterion_reg: nn.Module, 
    lambda1: float, 
    lambda2: float, 
    device: torch.device
) -> Tuple[float, float, float, float, float, float]:
    """Runs one full validation or test epoch."""
    model.eval()
    total_val_loss, total_class_loss, total_reg_loss = 0, 0, 0
    total_correct = 0
    total_samples = 0
    total_squared_error = 0
    total_abs_error = 0

    with torch.no_grad():
        for images, labels, inks in tqdm(loader, desc="Validating"):
            images, labels, inks = images.to(device), labels.to(device), inks.to(device)
            
            # 1. Forward pass
            logits_pred, inks_pred = model(images)
            
            # 2. Calculate joint loss
            loss_class = criterion_class(logits_pred, labels)
            loss_reg = criterion_reg(inks_pred, inks)
            loss = (lambda1 * loss_class) + (lambda2 * loss_reg)
            
            total_val_loss += loss.item()
            total_class_loss += loss_class.item()
            total_reg_loss += loss_reg.item()
            
            # 3. Calculate metrics
            # Classification: Accuracy
            _, predicted_labels = torch.max(logits_pred, 1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)
            
            # Regression: MAE and MSE (for RMSE)
            total_squared_error += ((inks_pred - inks) ** 2).sum().item()
            total_abs_error += torch.abs(inks_pred - inks).sum().item()

    # Calculate averages
    avg_loss = total_val_loss / len(loader)
    avg_class_loss = total_class_loss / len(loader)
    avg_reg_loss = total_reg_loss / len(loader)
    
    accuracy = total_correct / total_samples
    mae = total_abs_error / total_samples
    rmse = (total_squared_error / total_samples) ** 0.5
    
    return avg_loss, avg_class_loss, avg_reg_loss, accuracy, mae, rmse


def run_training():
    """Main training and validation loop."""
    
    # 1. Setup Logging and WandB
    # This will save terminal output to output/train.log
    setup_logging(log_file_name="train.log")
    
    
    wandb.init(
        project=config["project_name"], 
        name=config["run_name"], 
        config=config
    )
    
    # 3. Get DataLoaders
    logging.info("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config["batch_size"],
        val_split=config["validation_split"]
    )
    
    # 4. Initialize Model, Loss, and Optimizer
    logging.info("Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    model = MultiTaskCNN(dropout_rate=config["dropout_rate"]).to(device)
    
    # Loss functions
    criterion_class = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    
    # Optimizer
    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    else: # Default to SGD
        optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
        
    logging.info(f"Starting training for {config['epochs']} epochs...")
    logging.info(f"Hyperparameters: {config}")

    # 5. Training Loop
    for epoch in range(config["epochs"]):
        logging.info(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
        
        # --- Train ---
        avg_loss, avg_class_loss, avg_reg_loss = train_one_epoch(
            model, train_loader, optimizer, criterion_class, criterion_reg, 
            config["lambda1"], config["lambda2"], device
        )
        logging.info(f"Train: Total Loss={avg_loss:.4f}, "
                     f"Class Loss={avg_class_loss:.4f}, "
                     f"Reg Loss={avg_reg_loss:.4f}")
        
        # 3. Log training metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train/total_loss": avg_loss,
            "train/class_loss": avg_class_loss,
            "train/reg_loss": avg_reg_loss,
        })

        # --- Validate ---
        val_loss, val_class_loss, val_reg_loss, acc, mae, rmse = validate_one_epoch(
            model, val_loader, criterion_class, criterion_reg, 
            config["lambda1"], config["lambda2"], device
        )
        logging.info(f"Validate: Total Loss={val_loss:.4f}, "
                     f"Accuracy={acc:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
        
        # 4. Log validation metrics to wandb
        wandb.log({
            "epoch": epoch,
            "val/total_loss": val_loss,
            "val/class_loss": val_class_loss,
            "val/reg_loss": val_reg_loss,
            "val/accuracy": acc,
            "val/mae": mae,
            "val/rmse": rmse,
        })

    # 6. Final Test Evaluation
    # After all epochs, run on the test set once
    logging.info("\nTraining finished. Running on test set...")
    test_loss, test_class_loss, test_reg_loss, test_acc, test_mae, test_rmse = validate_one_epoch(
        model, test_loader, criterion_class, criterion_reg, 
        config["lambda1"], config["lambda2"], device
    )
    logging.info(f"Test: Accuracy={test_acc:.4f}, MAE={test_mae:.4f}, RMSE={test_rmse:.4f}")
    
    #7. Log final test metrics to wandb
    wandb.log({
        "test/accuracy": test_acc,
        "test/mae": test_mae,
        "test/rmse": test_rmse,
    })
    
    # 8. Save Model
    model_save_path = os.path.join(OUTPUT_DIR, f"{config['run_name']}.pth")
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved to {model_save_path}")
    
    # 9. Finish wandb run
    # wandb.finish()
    logging.info("Run finished.")

if __name__ == "__main__":
    run_training()

