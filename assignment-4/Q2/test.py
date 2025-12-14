# test.py
import os, argparse, json, random
from typing import Dict
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from model import ColorNet
from dataset import get_loaders   # reuses your CIFAR loaders

# ---------- utilities ----------
def set_seed(seed: int = 1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, criterion, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(1, n)

@torch.no_grad()
def save_examples(
    model: torch.nn.Module,
    loader,
    centroids_path: str,
    device: torch.device,
    out_dir: str,
    max_images: int = 10,
):
    os.makedirs(out_dir, exist_ok=True)

    # load centroids & scale to 0..1 if needed
    C = np.load(centroids_path)
    if C.max() > 1.5:
        C = C.astype(np.float32) / 255.0
    else:
        C = C.astype(np.float32)

    # take one batch from the test loader
    gray, target = next(iter(loader))            # gray: [B,1,32,32] in 0..1, target: [B,32,32]
    gray   = gray.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    logits = model(gray)                         # [B,24,32,32]
    pred   = torch.argmax(logits, dim=1)        # [B,32,32]

    # to numpy
    gray_np   = gray.detach().cpu().numpy().squeeze(1)          # [B,32,32], 0..1
    pred_np   = pred.detach().cpu().numpy().astype(np.int64)    # [B,32,32]
    target_np = target.detach().cpu().numpy().astype(np.int64)  # [B,32,32]

    # map classes to RGB via centroids
    pred_rgb   = C[pred_np]     # [B,32,32,3] float 0..1
    target_rgb = C[target_np]   # [B,32,32,3] float 0..1

    def to_rgb_uint8(x):
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    # save up to N images
    B = gray_np.shape[0]
    n = min(max_images, B)
    for i in range(n):
        g = to_rgb_uint8(gray_np[i])                    # [32,32] -> make RGB
        p = to_rgb_uint8(pred_rgb[i])                   # [32,32,3]
        t = to_rgb_uint8(target_rgb[i])                 # [32,32,3]

        g_img = Image.fromarray(g, mode="L").convert("RGB")
        p_img = Image.fromarray(p, mode="RGB")
        t_img = Image.fromarray(t, mode="RGB")

        # simple header strip (no text; avoids font deps)
        def add_header(img_rgb: Image.Image, header_px=8):
            W, H = img_rgb.size
            header = Image.new("RGB", (W, header_px), (255, 255, 255))
            out = Image.new("RGB", (W, H + header_px), (255, 255, 255))
            out.paste(header, (0, 0))
            out.paste(img_rgb, (0, header_px))
            return out

        g_img = add_header(g_img)
        p_img = add_header(p_img)
        t_img = add_header(t_img)

        # concat horizontally: [Input | Predicted | GroundTruth]
        W, H = g_img.size
        trip = Image.new("RGB", (W * 3, H), (255, 255, 255))
        trip.paste(g_img, (0, 0))
        trip.paste(p_img, (W, 0))
        trip.paste(t_img, (2 * W, 0))

        trip.save(os.path.join(out_dir, f"sample_{i:03d}_trip.png"))
        g_img.save(os.path.join(out_dir, f"sample_{i:03d}_input.png"))
        p_img.save(os.path.join(out_dir, f"sample_{i:03d}_pred.png"))
        t_img.save(os.path.join(out_dir, f"sample_{i:03d}_gt.png"))

    print(f"[test] wrote {n} samples to: {out_dir}")

def run_test(cfg: Dict):
    set_seed(cfg.get("seed", 1337))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # only need test loader; get_loaders gives all three
    _, _, test_loader = get_loaders(
        batch_size=cfg["batch_size"],
        centroids_path=cfg["centroids_path"],
        root=cfg.get("data_root", "./data"),
        num_workers=cfg.get("num_workers", 2),
        download=cfg.get("download", False),
    )

    # build model and load weights
    model = ColorNet(
        NIC=cfg.get("NIC", 1),
        NF=cfg.get("NF", 32),
        NC=cfg.get("NC", 24),
        kernel_size=cfg.get("kernel_size", 3),
    ).to(device)

    ckpt = cfg["checkpoint"]
    assert os.path.exists(ckpt), f"Checkpoint not found: {ckpt}"
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    print(f"[test] loaded checkpoint: {ckpt}")

    # compute test loss (optional)
    criterion = nn.CrossEntropyLoss()
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"[test] loss={test_loss:.4f}")

    # save example images
    save_examples(
        model=model,
        loader=test_loader,
        centroids_path=cfg["centroids_path"],
        device=device,
        out_dir=cfg["out_dir"],
        max_images=cfg["save_examples"],
    )

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate best model and save example images.")
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to best_model.pth saved during training")
    ap.add_argument("--centroids_path", type=str, default="color_centroids.npy")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./test_outputs")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--save_examples", type=int, default=10)
    ap.add_argument("--NIC", type=int, default=1)
    ap.add_argument("--NF", type=int, default=32)
    ap.add_argument("--NC", type=int, default=24)
    ap.add_argument("--kernel_size", type=int, default=3)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--cfg_json", type=str, default="", help="Optional JSON to override cfg")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = {
        "seed": args.seed,
        "NIC": args.NIC,
        "NF": args.NF,
        "NC": args.NC,
        "kernel_size": args.kernel_size,
        "batch_size": args.batch_size,
        "centroids_path": args.centroids_path,
        "data_root": args.data_root,
        "num_workers": args.num_workers,
        "download": args.download,
        "checkpoint": args.checkpoint,
        "out_dir": args.out_dir,
        "save_examples": args.save_examples,
    }
    if args.cfg_json:
        try:
            cfg.update(json.loads(args.cfg_json))
        except Exception as e:
            print("Failed to parse --cfg_json:", e)
    run_test(cfg)

if __name__ == "__main__":
    main()
