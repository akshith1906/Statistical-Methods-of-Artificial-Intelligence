# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# -------- helpers --------
def _load_centroids(path: str) -> np.ndarray:
    C = np.load(path)
    if C.max() > 1.5:     # stored as 0..255
        C = C.astype(np.float32) / 255.0
    else:
        C = C.astype(np.float32)
    assert C.shape == (24, 3), f"centroids must be (24,3), got {C.shape}"
    return C

def _rgb_to_labels(rgb_hwc: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # rgb_hwc: [H,W,3] in 0..1, float32
    # centroids: [24,3] in 0..1
    # returns [H,W] int64 in 0..23
    d2 = ((rgb_hwc[..., None, :] - centroids[None, None, :, :]) ** 2).sum(axis=-1)  # [H,W,24]
    lab = np.argmin(d2, axis=-1).astype(np.int64)                                   # [H,W]
    return lab

# -------- dataset --------
class ColorizationCIFAR10(Dataset):
    """
    Produces:
      - grayscale input: torch.float32 [1,32,32] in 0..1
      - target_map: torch.long [32,32] in {0..23}
    Targets are computed from the ORIGINAL RGB before any normalization.
    """
    def __init__(self, root: str, train: bool, centroids_path: str, download: bool = False):
        self.ds = datasets.CIFAR10(root=root, train=train, download=download,
                                   transform=None, target_transform=None)
        self.centroids = _load_centroids(centroids_path)

        # transforms to make grayscale AFTER we’ve saved RGB
        self.to_tensor = transforms.ToTensor()                   # 0..1 float
        self.to_gray = transforms.Grayscale(num_output_channels=1)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        pil_rgb, _ = self.ds[idx]                                # PIL RGB
        # make a torch tensor 0..1
        rgb_t = self.to_tensor(pil_rgb)                          # [3,32,32]
        # save a copy for target computation (HWC numpy)
        rgb_hwc = rgb_t.permute(1, 2, 0).numpy().astype(np.float32)  # [32,32,3], 0..1

        # grayscale input (still 0..1)
        gray_t = self.to_gray(pil_rgb)                           # PIL(1,32,32)
        gray_t = self.to_tensor(gray_t)                          # [1,32,32] 0..1

        # target labels from original RGB
        target_map = _rgb_to_labels(rgb_hwc, self.centroids)     # [32,32] int64
        target_t = torch.from_numpy(target_map).long()           # [32,32]

        return gray_t, target_t

# -------- loaders --------
def get_loaders(batch_size: int,
                centroids_path: str,
                root: str = "./data",
                num_workers: int = 2,
                download: bool = False):
    train_set = ColorizationCIFAR10(root=root, train=True,  centroids_path=centroids_path, download=download)
    test_set  = ColorizationCIFAR10(root=root, train=False, centroids_path=centroids_path, download=download)

    # split a small val from train (e.g., 45k/5k)
    val_size = 5000
    train_size = len(train_set) - val_size
    train_subset, val_subset = torch.utils.data.random_split(train_set, [train_size, val_size],
                                                             generator=torch.Generator().manual_seed(1337))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,     batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
