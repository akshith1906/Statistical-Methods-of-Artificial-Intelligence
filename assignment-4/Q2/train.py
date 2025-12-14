# train.py
import os, json, random
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import ColorNet
from dataset import get_loaders

# ------------ config ------------
USE_WANDB = True
WANDB_PROJECT = "smai-colorization-q2"
DEFAULT_CENTROIDS = "color_centroids.npy"
# --------------------------------

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_optimizer(model: nn.Module, name: str, lr: float):
    name = name.lower()
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)                     # [B,24,32,32]
        loss = criterion(logits, y)
        bs = x.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(1, n)

@torch.no_grad()
def sanity_check(loader, centroids_path: str):
    C = np.load(centroids_path)
    mx, mn = C.max(), C.min()
    print(f"[centroids] shape={C.shape} min={mn:.4f} max={mx:.4f}")
    if mx > 1.5:
        C = C / 255.0
        print("[centroids] normalized to 0..1")
    print(f"[centroids] (0..1) min={C.min():.4f} max={C.max():.4f}")

    x, y = next(iter(loader))
    hist = torch.bincount(y.flatten(), minlength=24).cpu().numpy()
    uniq = torch.unique(y).numel()
    print(f"[targets] unique labels: {int(uniq)} / 24")
    print(f"[targets] histogram: {hist.tolist()}")

def run_one(cfg: Dict):
    set_seed(cfg.get("seed", 1337))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_loaders(
        batch_size=cfg["batch_size"],
        centroids_path=cfg["centroids_path"],
        root=cfg.get("data_root", "./data"),
        num_workers=cfg.get("num_workers", 2),
        download=cfg.get("download", False),
    )

    sanity_check(train_loader, cfg["centroids_path"])

    model = ColorNet(
        NIC=cfg.get("NIC", 1),
        NF=cfg.get("NF", 32),
        NC=cfg.get("NC", 24),
        kernel_size=cfg.get("kernel_size", 3),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, cfg["optimizer"], cfg["learning_rate"])

    if USE_WANDB:
        import wandb
        wandb.init(project=WANDB_PROJECT, config=cfg)
        wandb.watch(model, log="gradients", log_freq=200)

    best_val = float("inf")
    os.makedirs(cfg.get("save_dir", "checkpoints"), exist_ok=True)
    best_path = os.path.join(cfg.get("save_dir", "checkpoints"), "best_model.pth")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running, n = 0.0, 0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch:03d}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            bs = x.size(0)
            running += loss.item() * bs
            n += bs
        train_loss = running / max(1, n)
        val_loss = evaluate(model, val_loader, criterion, device)

        if USE_WANDB:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[{epoch:03d}/{cfg['epochs']}] train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_loss = evaluate(model, test_loader, criterion, device)
    if USE_WANDB:
        wandb.log({"test_loss": test_loss})
    print(f"[TEST] loss={test_loss:.4f}")
    return test_loss

def main():
    cfg = {
        "seed": 1337,
        "NIC": 1,
        "NF": 32,
        "NC": 24,
        "kernel_size": 3,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "epochs": 25,
        "optimizer": "Adam",       # or "SGD"
        "centroids_path": DEFAULT_CENTROIDS,
        "data_root": "./data",
        "num_workers": 2,
        "save_dir": ".",
        "download": False          # set True if CIFAR-10 not present
    }

    # simple JSON override via env var if you want:  export Q2_CFG_JSON='{"epochs":5}'
    if os.getenv("Q2_CFG_JSON"):
        try:
            cfg.update(json.loads(os.environ["Q2_CFG_JSON"]))
        except Exception as e:
            print("Failed to parse Q2_CFG_JSON:", e)

    run_one(cfg)

if __name__ == "__main__":
    main()
