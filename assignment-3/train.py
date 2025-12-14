import os, time, json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from .losses import BCELoss
from .utils import ensure_dir, binary_accuracy

def train(
    model, X, Y,
    batch_size=128,
    grad_accum_steps=1,
    max_epochs=200,
    patience=10,
    rel_improve_thresh=0.01,
    runs_dir="runs",
    run_name=None,
    seed=0,
):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idx_all = np.arange(N)

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = run_name or f"run-{ts}"
    run_dir = os.path.join(runs_dir, run_name)
    ensure_dir(run_dir)

    losses, accs, samples_seen = [], [], []
    sample_counter = 0
    epoch_losses = []

    for epoch in trange(max_epochs, desc="epochs"):
        rng.shuffle(idx_all)
        running_loss_sum = 0.0
        running_count = 0
        micro = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_idx = idx_all[start:end]
            xb = X[batch_idx]
            yb = Y[batch_idx]

            loss, ypred = model.train_step(xb, yb)
            running_loss_sum += loss * xb.shape[0]
            running_count += xb.shape[0]

            if isinstance(model.loss_fn, BCELoss):
                acc = binary_accuracy(ypred, yb)
            else:
                acc = binary_accuracy(ypred, yb)

            sample_counter += xb.shape[0]
            losses.append(loss)
            accs.append(acc)
            samples_seen.append(sample_counter)

            micro += 1
            if micro % grad_accum_steps == 0:
                model.update(grad_accum_scale=1.0 / grad_accum_steps)

        if micro % grad_accum_steps != 0:
            steps_left = micro % grad_accum_steps
            model.update(grad_accum_scale=1.0 / steps_left)

        epoch_loss = running_loss_sum / max(running_count, 1)
        epoch_losses.append(epoch_loss)

        if len(epoch_losses) > patience:
            prev = epoch_losses[-(patience + 1)]
            curr = epoch_losses[-1]
            if curr >= (1.0 - rel_improve_thresh) * prev:
                break

    plt.figure()
    plt.plot(samples_seen, losses, label="loss")
    plt.xlabel("Samples Seen")
    plt.ylabel("Training Loss")
    plt.title("Loss vs Samples")
    plt.tight_layout()
    plot_path = os.path.join(run_dir, "loss_vs_samples.png")
    plt.savefig(plot_path)
    plt.close()

    weights_path = os.path.join(run_dir, "model_weights.npz")
    model.save_to(weights_path)

    meta = dict(
        run_dir=run_dir,
        plot_path=plot_path,
        weights_path=weights_path,
        max_epochs=max_epochs,
        epochs_run=len(epoch_losses),
        final_epoch_loss=epoch_losses[-1] if epoch_losses else None,
        patience=patience,
        rel_improve_thresh=rel_improve_thresh,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        samples_seen=sample_counter
    )
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return meta, (samples_seen, losses, accs)
