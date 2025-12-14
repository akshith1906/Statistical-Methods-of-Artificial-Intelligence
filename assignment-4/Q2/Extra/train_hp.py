# train_hp.py
import os, argparse, random
from typing import Dict
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from model import ColorNet
from dataset import get_loaders


# =============== Utilities & Metrics ===============

def set_seed(seed: int = 1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def make_optimizer(model: nn.Module, name: str, lr: float):
    name = name.lower()
    if name == "adam": return optim.Adam(model.parameters(), lr=lr)
    if name == "sgd":  return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")

@torch.no_grad()
def pixel_acc_from_logits(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item()

@torch.no_grad()
def miou_from_logits(logits: torch.Tensor, target: torch.Tensor, n_classes: int) -> float:
    pred = logits.argmax(dim=1)
    miou_sum, n_valid = 0.0, 0
    for c in range(n_classes):
        p = (pred == c); t = (target == c)
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union > 0:
            miou_sum += inter / union
            n_valid += 1
    return miou_sum / max(1, n_valid)

@torch.no_grad()
def eval_epoch(model: nn.Module, loader, device, n_classes: int) -> Dict[str, float]:
    ce = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    loss_sum, n_img, acc_sum, miou_sum, batches = 0.0, 0, 0.0, 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += loss.item(); n_img += x.size(0)
        acc_sum  += pixel_acc_from_logits(logits, y)
        miou_sum += miou_from_logits(logits, y, n_classes)
        batches  += 1
    return {
        "loss": loss_sum / max(1, n_img),     # avg per image
        "pixel_acc": acc_sum / max(1, batches),
        "miou": miou_sum / max(1, batches),
    }

@torch.no_grad()
def make_example_images(model: nn.Module, loader, centroids_path: str, device, max_images=6):
    """Return list of wandb.Image triptychs (input | predicted | ground-truth)."""
    import wandb
    C = np.load(centroids_path)
    C = (C.astype(np.float32) / 255.0) if C.max() > 1.5 else C.astype(np.float32)

    x, y = next(iter(loader))                         # one batch
    x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
    pred = model(x).argmax(dim=1)

    gray = x.detach().cpu().numpy().squeeze(1)        # [B,32,32] in 0..1
    pidx = pred.detach().cpu().numpy().astype(np.int64)
    tidx = y.detach().cpu().numpy().astype(np.int64)
    prgb = C[pidx]                                    # [B,32,32,3]
    trgb = C[tidx]

    imgs = []
    B = gray.shape[0]; n = min(max_images, B)
    for i in range(n):
        g = (np.clip(gray[i], 0, 1) * 255).astype(np.uint8)
        p = (np.clip(prgb[i], 0, 1) * 255).astype(np.uint8)
        t = (np.clip(trgb[i], 0, 1) * 255).astype(np.uint8)
        g_img = Image.fromarray(g, mode="L").convert("RGB")
        p_img = Image.fromarray(p, mode="RGB")
        t_img = Image.fromarray(t, mode="RGB")
        W, H = g_img.size
        trip = Image.new("RGB", (W * 3, H), (255, 255, 255))
        trip.paste(g_img, (0, 0)); trip.paste(p_img, (W, 0)); trip.paste(t_img, (2 * W, 0))
        imgs.append(wandb.Image(trip, caption="input | predicted | ground-truth"))
    return imgs

def pretty_name(cfg: Dict) -> str:
    opt = cfg["optimizer"].lower()
    lr  = cfg["learning_rate"]
    lr_str = f"{lr:.0e}" if lr < 1e-2 else f"{lr:.3f}"
    return f"NF{cfg['NF']}-k{cfg['kernel_size']}-b{cfg['batch_size']}-{opt}-lr{lr_str}"


# =============== Single-Run Train/Eval (no init here) ===============

def run_one(cfg: Dict, run):
    """
    Train/eval a single config sampled by the sweep.
    Assumes wandb.init() is ALREADY active and 'run' is provided by agent_worker().
    """
    import wandb
    set_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_loaders(
        batch_size=cfg["batch_size"],
        centroids_path=cfg["centroids_path"],
        root=cfg.get("data_root", "./data"),
        num_workers=cfg.get("num_workers", 2),
        download=cfg.get("download", False),
    )

    # Make sure your ColorNet uses padding=kernel_size//2 internally for same-shape convs.
    model = ColorNet(NIC=1, NF=cfg["NF"], NC=cfg["NC"], kernel_size=cfg["kernel_size"]).to(device)
    crit  = nn.CrossEntropyLoss()
    opt   = make_optimizer(model, cfg["optimizer"], cfg["learning_rate"])

    # Give the run a nice name after init
    try:
        run.name = pretty_name(cfg)
        run.tags = list(set(list(run.tags or []) + ["sweep","q2", f"NF{cfg['NF']}", f"k{cfg['kernel_size']}", cfg["optimizer"].lower()]))
    except Exception:
        pass

    best_val = float("inf")
    best_state = None

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_sum, n = 0.0, 0
        for x, y in tqdm(train_loader, desc=f"train e{epoch:02d}", leave=False):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward(); opt.step()
            epoch_sum += loss.item() * x.size(0); n += x.size(0)
        train_loss = epoch_sum / max(1, n)

        val_metrics = eval_epoch(model, val_loader, device, cfg["NC"])
        run.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_pixel_acc": val_metrics["pixel_acc"],
            "val_miou": val_metrics["miou"],
        })

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Restore best and test
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = eval_epoch(model, test_loader, device, cfg["NC"])
    run.summary["best_val_loss"] = best_val
    run.summary["test_loss"] = test_metrics["loss"]
    run.summary["test_pixel_acc"] = test_metrics["pixel_acc"]
    run.summary["test_miou"] = test_metrics["miou"]

    # Example images (best model)
    try:
        imgs = make_example_images(model, test_loader, cfg["centroids_path"], device, max_images=6)
        run.log({"examples": imgs})
    except Exception as e:
        print("[warn] example image logging failed:", e)

    # Save + upload best checkpoint artifact
    try:
        import wandb
        ckpt_dir = os.path.join("sweep_ckpts", run.id); os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
        torch.save(model.state_dict(), ckpt_path)
        art = wandb.Artifact(name=f"best_model_{run.id}", type="model")
        art.add_file(ckpt_path)
        run.log_artifact(art)
        run.summary["best_ckpt_path"] = ckpt_path
    except Exception as e:
        print("[warn] checkpoint artifact upload failed:", e)

    print(f"[wandb] run url: {run.url}")


# =============== Sweep Launcher (EXACTLY 20 runs) ===============

def build_sweep_config(args) -> Dict:
    return {
        "method": "random",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"values": [1e-4, 3e-4, 1e-3, 3e-3]},
            "batch_size":    {"values": [32, 64, 128]},
            "NF":            {"values": [8, 16, 32]},
            "kernel_size":   {"values": [3, 5]},
            "optimizer":     {"values": ["Adam", "SGD"]},
            # fixed params
            "epochs":        {"value": args.epochs},
            "NC":            {"value": 24},
            "centroids_path":{"value": args.centroids_path},
            "data_root":     {"value": args.data_root},
            "num_workers":   {"value": args.num_workers},
            "download":      {"value": args.download},
        },
    }

def agent_worker(args):
    """Initialize wandb first (so wandb.config exists), then run one config."""
    import wandb
    run = wandb.init(project=args.project, job_type="sweep-train")
    try:
        cfg = dict(wandb.config)  # safe now that we've init'ed
        run_one(cfg, run=run)
    finally:
        run.finish()

def main():
    import wandb
    ap = argparse.ArgumentParser(description="W&B sweep (exactly 20 configs) with images & artifacts.")
    ap.add_argument("--project", type=str, default="smai-colorization-hp-eval-20")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--centroids_path", type=str, default="color_centroids.npy")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--api_key", type=str, default="")
    args = ap.parse_args()

    if args.api_key:
        os.environ["WANDB_API_KEY"] = args.api_key

    sweep_config = build_sweep_config(args)
    sweep_id = wandb.sweep(sweep_config, project=args.project)
    print(f"[sweep] id: {sweep_id}  (project: {args.project})")

    # EXACTLY 20 configs
    wandb.agent(sweep_id, function=lambda: agent_worker(args), count=20)

if __name__ == "__main__":
    main()
