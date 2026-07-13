#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference:
Dong, C., Loy, C.C., Tang, X. (2016). Accelerating the Super-Resolution Convolutional Neural Network. 
In: Leibe, B., Matas, J., Sebe, N., Welling, M. (eds) Computer Vision – ECCV 2016. ECCV 2016. 
Lecture Notes in Computer Science(), vol 9906. Springer, Cham.
https://doi.org/10.1007/978-3-319-46475-6_25


FSRCNN / FSRCNN-s Super-resolution/ Spectral enhancement:
    - Input : Sentinel-2 composites (NDVI, GNDVI, NDRE, NDWI, NDVIre, CIre, etc.)
    - Target : Drone composites (same bands, same 0.5 m grid)


Two variants of FSRCNN are implemented:
    - FSRCNN standard: d=56, s=12, m=4
    - FSRCNN-s (fast): d=32, s=5, m=1

Important notes:
    - Assume that the Sentinel and Drone GeoTIFFs are already cropped to the same AOI and
      resampled to the same resolution (0.5 m) and matrix size.
"""

import os
import time
from pathlib import Path
import random
import math
import numpy as np
import rasterio

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, random_split

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# Optional metrics (use evaluate_metrics.py if available)
try:
    from evaluate_metrics import (
        read_raster,
        compute_band_metrics,
        spectral_angle_mapper,
        spectral_information_divergence,
        ergas,
        format_table as metrics_format_table,
    )
    HAS_METRICS = True
except Exception:
    HAS_METRICS = False

# ---------------- Configuration ----------------
# List of Sentinel-Drone pairs (adjust with your paths)
TRAIN_PAIRS = [
    (
        r"D:\Nueva carpeta\2025\Fusion de datos\Articulo 1\output_recorte_prev_remuestreo\sentinel\20250813_comp_sentinel_2m.tif",
        r"D:\Nueva carpeta\2025\Fusion de datos\Articulo 1\output_recorte_prev_remuestreo\dron\20250813_comp_dron_2m.tif"
    ),
    
]

OUT_DIR = Path(r"D:\Nueva carpeta\2025\Fusion de datos\fsrcnn_model")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# The spatial resolution and number of pairs for naming the model
def get_spatial_resolution(image_path):
    """Extract the spatial resolution (in meters) from a GeoTIFF."""
    try:
        with rasterio.open(image_path) as src:
            res_x = abs(src.transform.a)
            res_y = abs(src.transform.e)
            return (res_x + res_y) / 2.0  # Average if not square
    except Exception:
        return None

def format_resolution_string(res_m):
    """Format resolution in meters to string for filename (0.5 -> '0p5m', 1.0 -> '1m')."""
    if res_m is None:
        return "unknown"
    if res_m == int(res_m):
        return f"{int(res_m)}m"
    else:
        return f"{res_m}m".replace(".", "p")

# Detect resolution of the first pair (Drone) and count the number of dates used
_resolution = get_spatial_resolution(TRAIN_PAIRS[0][1]) if TRAIN_PAIRS else None
_res_str = format_resolution_string(_resolution)
_num_dates = len(TRAIN_PAIRS)
_dates_str = f"{_num_dates}date" if _num_dates == 1 else f"{_num_dates}dates"

# Model name includes the spatial resolution and the number of training dates
MODEL_PATH = OUT_DIR / f"fsrcnn_fsrcnn_dron_{_res_str}_{_dates_str}.pth"

# Optional shifts for manually coregistering the drone (in meters).
# Sign: cols>0 moves the drone to the right; rows>0 moves it up.
DRON_SHIFT_COLS_M = 0.0
DRON_SHIFT_ROWS_M = 0.0

# Training hyperparameters
PATCH_SIZE = 96
PATCHES_PER_IMAGE = 500  # Reduced for greater diversity with large patches
BATCH_SIZE = 16
NUM_EPOCHS = 1000
LEARNING_RATE = 0.001
GRAD_CLIP_NORM = 1.0  # Restored: optimal value demonstrated
LR_STEP_SIZE = 100
LR_GAMMA = 0.9
CHECKPOINT_INTERVAL = 10
MAX_PATCH_RETRIES = 100
SEED = 2025
NUM_WORKERS = 0
VAL_FRACTION = 0.1

# Physical range of indices
NORMALIZE_BANDS = False
INDEX_MIN = -1.0
INDEX_MAX = 1.0

# Number of bands (adjust if the input composition bands are changed)
NUM_CHANNELS = 7

# Spatial scale factor
UPSCALE_FACTOR = 1

# Select variant: "fsrcnn" or "fsrcnn-s"
MODEL_VARIANT = "fsrcnn"   # or "fsrcnn-s"

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP     = DEVICE.type == "cuda"
PIN_MEMORY  = DEVICE.type == "cuda"
torch.backends.cudnn.benchmark = True
AMP_DEVICE_TYPE = DEVICE.type


def clip_to_index_range(arr: np.ndarray) -> np.ndarray:
    """Clips index values to the physical window [-1, 1] respecting NaNs."""
    return np.clip(arr, INDEX_MIN, INDEX_MAX)

def apply_pixel_shift(arr: np.ndarray, shift_rows_m: float = 0.0, shift_cols_m: float = 0.0, res_y: float = 1.0, res_x: float = 1.0) -> np.ndarray:
    """
    Shift a stack (C,H,W) with displacement in meters, converting to pixels (subpixel) and interpolating.
    shift_cols_m>0 -> right; shift_rows_m>0 -> up.
    res_x/res_y: pixel size (m) of the drone.
    """
    if abs(shift_rows_m) < 1e-9 and abs(shift_cols_m) < 1e-9:
        return arr

    px_rows = -(shift_rows_m / res_y) if res_y != 0 else 0.0  # rows grow downwards, sign inverted to move up
    px_cols = shift_cols_m / res_x if res_x != 0 else 0.0

    try:
        from scipy.ndimage import shift as nd_shift
        out = np.empty_like(arr)
        for i in range(arr.shape[0]):
            out[i] = nd_shift(
                arr[i],
                shift=(px_rows, px_cols),
                order=1,
                mode="nearest",
                prefilter=False,
            )
        return out
    except Exception:
        sr = int(round(px_rows))
        sc = int(round(px_cols))
        _, h, w = arr.shape
        pad_top = max(sr, 0)
        pad_bottom = max(-sr, 0)
        pad_left = max(sc, 0)
        pad_right = max(-sc, 0)
        arr_pad = np.pad(arr, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
        r0 = pad_top - min(sr, 0)
        c0 = pad_left - min(sc, 0)
        return arr_pad[:, r0:r0 + h, c0:c0 + w]


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


def init_training_plot():
    """
    Sets up an interactive figure to monitor Loss and PSNR.
    Returns (fig, ax_loss, ax_psnr) or None if matplotlib is not available.
    """
    if not HAS_MPL:
        return None
    plt.ion()
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.set_title("FSRCNN Training Progress")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax2.set_ylabel("PSNR (dB)", color="tab:orange")
    fig.tight_layout()
    return fig, ax1, ax2


def update_training_plot(handles, epochs, train_losses, train_psnrs, eval_losses=None, eval_psnrs=None):
    """
    Refreshes the interactive figure with the accumulated lists (train and eval).
    """
    if handles is None:
        return
    fig, ax1, ax2 = handles
    ax1.cla()
    ax2.cla()
    ax1.set_title("FSRCNN Training Progress")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax2.set_ylabel("PSNR (dB)", color="tab:orange")
    ax1.plot(epochs, train_losses, color="tab:blue", label="Loss (train)")
    if eval_losses is not None:
        ax1.plot(epochs, eval_losses, color="tab:green", linestyle="--", label="Loss (eval)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax2.plot(epochs, train_psnrs, color="tab:orange", label="PSNR (train)")
    if eval_psnrs is not None:
        ax2.plot(epochs, eval_psnrs, color="tab:red", linestyle="--", label="PSNR (eval)")
    fig.legend(loc="upper right")
    fig.canvas.draw()
    fig.canvas.flush_events()

# ---------------- MODELOS FSRCNN ----------------
class FSRCNN(nn.Module):
    """
    General implementation of FSRCNN adapted to multiple channels.

    Parameters:
        in_channels   : number of input channels (your indices)
        d             : number of filters in the feature extraction layer
        s             : number of filters in the shrinking / expanding layers
        m             : number of mapping layers (3×3) in the "neck"
        upscale_factor: scale factor (here 1, because you already work at 0.5 m)
    """
    def __init__(self, in_channels=7, d=56, s=12, m=4, upscale_factor=1):
        super(FSRCNN, self).__init__()
        self.upscale_factor = upscale_factor

        # 1) Feature extraction (5x5)
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=5, padding=2),
            nn.PReLU()
        )

        # 2) Shrinking (1x1)
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU()
        )

        # 3) Mapping (m times 3x3)
        mapping_layers = []
        for _ in range(m):
            mapping_layers.append(
                nn.Conv2d(s, s, kernel_size=3, padding=1)
            )
            mapping_layers.append(nn.PReLU())
        self.mapping = nn.Sequential(*mapping_layers)

        # 4) Expanding (1x1)
        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU()
        )

        # 5) Reconstruction:
        #    - If upscale_factor>1 we use the classic FSRCNN deconvolution.
        #    - If upscale_factor=1 we use a standard conv to avoid
        #      negative padding and maintain the same resolution.
        if upscale_factor == 1:
            self.reconstruction = nn.Conv2d(
                d, in_channels,
                kernel_size=3,
                padding=1
            )
        else:
            self.reconstruction = nn.ConvTranspose2d(
                d, in_channels,
                kernel_size=9,
                stride=upscale_factor,
                padding=4,
                output_padding=upscale_factor - 1
            )

        # He initialization for stability
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: tensor (B, C, H, W)
        output: tensor (B, C, H', W') where H',W' = H*factor (here same if factor=1)
        """
        out = self.feature_extraction(x)
        out = self.shrinking(out)
        out = self.mapping(out)
        out = self.expanding(out)
        out = self.reconstruction(out)  # si factor=1, H',W'≈H,W

        # If you want residual and the same resolution:
        if self.upscale_factor == 1 and out.shape == x.shape:
            out = out + x
        return out


def build_fsrcnn_model(variant: str, in_channels: int, upscale_factor: int = 1) -> FSRCNN:
    """
    Creates the requested variant:
        - "fsrcnn"   : d=56, s=12, m=4
        - "fsrcnn-s" : d=32, s=5,  m=1
    """
    variant = variant.lower()
    if variant == "fsrcnn":
        return FSRCNN(in_channels=in_channels, d=56, s=12, m=4, upscale_factor=upscale_factor)
    elif variant == "fsrcnn-s":
        return FSRCNN(in_channels=in_channels, d=32, s=5, m=1, upscale_factor=upscale_factor)
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'fsrcnn' or 'fsrcnn-s'.")


# ---------------- DATASET ----------------
class SentinelDronDataset(Dataset):
    """
    Sentinel–Dron patches dataset:
        - Read pairs of GeoTIFF (same size, CRS, and transform).
        - Extract random patches for training.
        - Sentinel = input (X), Dron = target (Y).
    """
    def __init__(self, pairs, patch_size=64, num_patches_per_image=200,
                 normalize=NORMALIZE_BANDS, max_invalid_retry=MAX_PATCH_RETRIES):
        self.pairs = pairs
        self.patch_size = patch_size
        self.num_patches = num_patches_per_image * len(pairs)
        self.normalize = normalize
        self.max_invalid_retry = max_invalid_retry

        self.data = []
        for s_path, d_path in pairs:
            with rasterio.open(s_path) as s_src, rasterio.open(d_path) as d_src:
                s_arr = s_src.read().astype(np.float32)  # (C,H,W)
                d_arr = d_src.read().astype(np.float32)

                # Replace nodata with NaN
                if s_src.nodata is not None:
                    s_arr = np.where(s_arr == s_src.nodata, np.nan, s_arr)
                if d_src.nodata is not None:
                    d_arr = np.where(d_arr == d_src.nodata, np.nan, d_arr)

                # Apply shift in meters -> pixels (subpixel) to the drone with interpolation
                res_x = abs(d_src.transform.a)
                res_y = abs(d_src.transform.e)
                d_arr = apply_pixel_shift(
                    d_arr,
                    shift_rows_m=DRON_SHIFT_ROWS_M,
                    shift_cols_m=DRON_SHIFT_COLS_M,
                    res_y=res_y,
                    res_x=res_x,
                )
                # Clip to physical index range
                s_arr = clip_to_index_range(s_arr)
                d_arr = clip_to_index_range(d_arr)

                if normalize:
                    # Note: if you enable this, you must denormalize during inference if
                    # you want to recover the physical NDVI. For now, we leave it False.
                    for arr in (s_arr, d_arr):
                        for b in range(arr.shape[0]):
                            v = arr[b]
                            mask = np.isfinite(v)
                            if np.any(mask):
                                mu = np.nanmean(v[mask])
                                sd = np.nanstd(v[mask]) + 1e-6
                                arr[b] = (v - mu) / sd

                self.data.append((s_arr, d_arr))

        if not self.data:
            raise ValueError("No Sentinel–Dron pairs loaded.")

        self.H = self.data[0][0].shape[1]
        self.W = self.data[0][0].shape[2]

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        ps = self.patch_size
        max_row = self.H - ps
        max_col = self.W - ps
        if max_row <= 0 or max_col <= 0:
            raise ValueError("The image is smaller than the patch. Reduce PATCH_SIZE.")

        for _ in range(self.max_invalid_retry):
            pair_idx = random.randint(0, len(self.data) - 1)
            s_arr, d_arr = self.data[pair_idx]

            r0 = random.randint(0, max_row)
            c0 = random.randint(0, max_col)

            s_patch = s_arr[:, r0:r0+ps, c0:c0+ps]
            d_patch = d_arr[:, r0:r0+ps, c0:c0+ps]

            valid = np.isfinite(s_patch) & np.isfinite(d_patch)
            if not np.any(valid):
                continue

            s_patch = np.nan_to_num(s_patch, nan=0.0)
            d_patch = np.nan_to_num(d_patch, nan=0.0)
            mask = valid.astype(np.float32)

            s_t = torch.from_numpy(s_patch).float()
            d_t = torch.from_numpy(d_patch).float()
            m_t = torch.from_numpy(mask).float()
            return s_t, d_t, m_t

        raise RuntimeError("Failed to extract a valid patch after several attempts.")


def create_dataloader(dataset, shuffle=True, drop_last=True):
    kwargs = dict(
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        drop_last=drop_last,
        pin_memory=PIN_MEMORY
    )
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


# ---------------- ENTRENAMIENTO ----------------
def psnr_torch(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    mse_val = mse.item() if isinstance(mse, torch.Tensor) else float(mse)
    if mse_val == 0:
        return torch.tensor(float("inf"))
    return 10 * torch.log10((max_val ** 2) / mse)


def masked_mse(pred, target, mask, eps=1e-8):
    """
    Masked MSE (only valid pixels defined by mask: 1=valid, 0=nodata).
    """
    valid = (mask > 0.5).float()
    denom = valid.sum()
    if denom.item() == 0:
        return torch.tensor(0.0, device=pred.device)
    diff2 = (pred - target) ** 2
    return (diff2 * valid).sum() / (denom + eps)

def train_fsrcnn():
    set_seed(SEED)

    train_start_time = time.perf_counter()

    dataset = SentinelDronDataset(
        TRAIN_PAIRS,
        patch_size=PATCH_SIZE,
        num_patches_per_image=PATCHES_PER_IMAGE,
        normalize=NORMALIZE_BANDS,
        max_invalid_retry=MAX_PATCH_RETRIES
    )
    # Split train/val maintaining the proportion of patches (or no val if VAL_FRACTION<=0)
    if VAL_FRACTION <= 0:
        train_ds = dataset
        val_loader = None
    else:
        val_len = max(1, int(len(dataset) * VAL_FRACTION))
        train_len = max(1, len(dataset) - val_len)
        if train_len + val_len > len(dataset):
            val_len = len(dataset) - train_len
        train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(SEED))
        val_loader = create_dataloader(val_ds, shuffle=False, drop_last=False)

    train_loader = create_dataloader(train_ds, shuffle=True, drop_last=True)

    model = build_fsrcnn_model(
        variant=MODEL_VARIANT,
        in_channels=NUM_CHANNELS,
        upscale_factor=UPSCALE_FACTOR
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',      # Minimize the loss
        factor=0.75,      # Reduce LR to 75%
        patience=20,     # Initial (adjusts dynamically: 15/25/40 depending on improvement)
        verbose=True,    # Show when LR is reduced
        min_lr=1e-12      # Minimum LR
    )
    scaler = GradScaler(enabled=USE_AMP, device=AMP_DEVICE_TYPE if USE_AMP else "cpu")

    print(f"Entrenando {MODEL_VARIANT} en {DEVICE} con {len(dataset)} patches totales...")
    best_loss = float("inf")
    prev_best_loss = float("inf")  # Para rastrear mejora
    epoch_axis = []
    loss_history, psnr_history = [], []
    eval_loss_history = [] if VAL_FRACTION > 0 else None
    eval_psnr_history = [] if VAL_FRACTION > 0 else None
    plot_handles = init_training_plot()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        num_batches = 0

        for s_batch, d_batch, m_batch in train_loader:
            s_batch = s_batch.to(DEVICE, non_blocking=PIN_MEMORY)
            d_batch = d_batch.to(DEVICE, non_blocking=PIN_MEMORY)
            m_batch = m_batch.to(DEVICE, non_blocking=PIN_MEMORY)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=AMP_DEVICE_TYPE, enabled=USE_AMP):
                out = model(s_batch)
                loss = masked_mse(out, d_batch, m_batch)

            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            # Metrics
            epoch_loss += loss.item()
            # Approximate PSNR (range ~[-1,1], peak ~2) using mask
            with torch.no_grad():
                mse_masked = masked_mse(out, d_batch, m_batch)
                mse_val = mse_masked.item() if isinstance(mse_masked, torch.Tensor) else float(mse_masked)
                if mse_val > 0:
                    batch_psnr = 10.0 * math.log10((2.0 ** 2) / mse_val)
                else:
                    batch_psnr = float("inf")
                epoch_psnr += batch_psnr

            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        avg_psnr = epoch_psnr / max(1, num_batches)
        current_lr = optimizer.param_groups[0]["lr"]
        # Evaluation (no gradients, only if there is val)
        eval_loss = None
        eval_psnr = None
        if val_loader is not None:
            model.eval()
            eval_loss_acc = 0.0
            eval_psnr_acc = 0.0
            eval_batches = 0
            with torch.no_grad():
                for s_batch, d_batch, m_batch in val_loader:
                    s_batch = s_batch.to(DEVICE, non_blocking=PIN_MEMORY)
                    d_batch = d_batch.to(DEVICE, non_blocking=PIN_MEMORY)
                    m_batch = m_batch.to(DEVICE, non_blocking=PIN_MEMORY)

                    with autocast(device_type=AMP_DEVICE_TYPE, enabled=USE_AMP):
                        out = model(s_batch)
                        loss_eval = masked_mse(out, d_batch, m_batch)

                    eval_loss_acc += loss_eval.item()
                    mse_masked = masked_mse(out, d_batch, m_batch)
                    mse_val = mse_masked.item() if isinstance(mse_masked, torch.Tensor) else float(mse_masked)
                    if mse_val > 0:
                        batch_psnr = 10.0 * math.log10((2.0 ** 2) / mse_val)
                    else:
                        batch_psnr = float("inf")
                    eval_psnr_acc += batch_psnr
                    eval_batches += 1
            eval_loss = eval_loss_acc / max(1, eval_batches)
            eval_psnr = eval_psnr_acc / max(1, eval_batches)
        if val_loader is not None:
            print(
                f"[Epoch {epoch}/{NUM_EPOCHS}] "
                f"Loss: {avg_loss:.6f} | PSNR: {avg_psnr:.2f} dB | "
                f"EvalLoss: {eval_loss:.6f} | EvalPSNR: {eval_psnr:.2f} dB | "
                f"LR: {current_lr:.2e}"
            )
        else:
            print(
                f"[Epoch {epoch}/{NUM_EPOCHS}] "
                f"Loss: {avg_loss:.6f} | PSNR: {avg_psnr:.2f} dB | "
                f"LR: {current_lr:.2e}"
            )

        epoch_axis.append(epoch)
        loss_history.append(avg_loss)
        psnr_history.append(avg_psnr)
        if val_loader is not None:
            eval_loss_history.append(eval_loss)
            eval_psnr_history.append(eval_psnr)
        update_training_plot(
            plot_handles,
            epoch_axis,
            loss_history,
            psnr_history,
            eval_losses=eval_loss_history if val_loader is not None else None,
            eval_psnrs=eval_psnr_history if val_loader is not None else None
        )

        # Patience based on magnitude of improvement
        if avg_loss < best_loss - 1e-6:
            improvement = prev_best_loss - avg_loss
            if improvement > 0.001:
                scheduler.patience = 15
                print(f"  → Large improvement ({improvement:.6f}), patience=15")
            elif improvement > 0.0001:
                scheduler.patience = 25
                print(f"  → Medium improvement ({improvement:.6f}), patience=25")
            else:
                scheduler.patience = 40
                print(f"  → Small improvement ({improvement:.6f}), patience=40")
            prev_best_loss = best_loss

        scheduler.step(avg_loss)

        epoch_time = time.perf_counter() - epoch_start_time

        # Save model when it improves or every certain number of epochs
        save_model = False
        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            save_model = True
        elif epoch % CHECKPOINT_INTERVAL == 0 or epoch == NUM_EPOCHS:
            save_model = True

        if save_model:
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Model saved in: {MODEL_PATH} (best_loss={best_loss:.6f}) | Epoch time: {epoch_time:.2f} s")

    total_train_time = time.perf_counter() - train_start_time
    print(f"Total training time FSRCNN_v2: {total_train_time:.2f} s")
    print("Training finished.")


# ---------------- INFERENCIA ----------------
@torch.inference_mode()
def apply_fsrcnn_to_full_image(sentinel_path, out_path):
    """
    Apply trained FSRCNN to a full Sentinel image.
    Maintains dimensions (factor=1) and corrects towards Drone.
    Preserves band names if they exist.
    """
    infer_start_time = time.perf_counter()
    model = build_fsrcnn_model(
        variant=MODEL_VARIANT,
        in_channels=NUM_CHANNELS,
        upscale_factor=UPSCALE_FACTOR
    ).to(DEVICE)

    # Load weights safely; use weights_only=True if available (PyTorch >= 2.4)
    try:
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    except TypeError:
        # Compatibility with older PyTorch versions without weights_only
        state = torch.load(MODEL_PATH, map_location=DEVICE)
    loaded = False
    if isinstance(state, dict):
        # Support common structures: {'state_dict': ...} or {'model_state_dict': ...}
        if 'state_dict' in state:
            state = state['state_dict']
        elif 'model_state_dict' in state:
            state = state['model_state_dict']
        try:
            model.load_state_dict(state)
            loaded = True
        except Exception:
            loaded = False

    if not loaded:
        # if a full saved module was passed
        if hasattr(state, 'state_dict'):
            try:
                model.load_state_dict(state.state_dict())
                loaded = True
            except Exception:
                loaded = False

    if not loaded:
        raise RuntimeError(f"Failed to load valid weights from: {MODEL_PATH}")

    model.eval()

    with rasterio.open(sentinel_path) as src:
        prof = src.profile.copy()
        arr = src.read().astype(np.float32)  # (C,H,W)
        nodata_in = src.nodata

        if nodata_in is not None:
            arr = np.where(arr == nodata_in, np.nan, arr)
        arr = clip_to_index_range(arr)
        valid_mask = np.isfinite(arr)

        band_descriptions = src.descriptions or []
        if not band_descriptions or all(b is None or b == "" for b in band_descriptions):
            band_descriptions = [f"Band_{i+1}" for i in range(arr.shape[0])]

        H, W = arr.shape[1], arr.shape[2]
        inp = torch.from_numpy(np.nan_to_num(arr, nan=0.0)).unsqueeze(0).to(DEVICE)  # (1,C,H,W)

        with autocast(device_type=AMP_DEVICE_TYPE, enabled=USE_AMP):
            out = model(inp)

        out_np = out.squeeze(0).cpu().numpy().astype(np.float32)
        out_np = clip_to_index_range(out_np)
        out_np = np.where(valid_mask, out_np, np.nan)
        out_np = np.nan_to_num(out_np, nan=-9999.0)

        def _tile_mult16(val, cap=512):
            val = min(cap, val)
            return max(16, (val // 16) * 16)

        tile_w = _tile_mult16(W)
        tile_h = _tile_mult16(H)

        prof.update(
            driver="GTiff",
            dtype="float32",
            count=out_np.shape[0],
            nodata=-9999.0,
            compress="deflate",
            tiled=True,
            blockxsize=tile_w,
            blockysize=tile_h
        )

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()  # borra archivos previos corruptos
        with rasterio.open(out_path.as_posix(), "w", **prof) as dst:
            dst.write(out_np)
            for i in range(out_np.shape[0]):
                desc = band_descriptions[i] if i < len(band_descriptions) else f"Band_{i+1}"
                dst.set_band_description(i+1, desc)

        print(f"FSRCNN image ({MODEL_VARIANT}) saved in: {out_path}")
        print(f"FSRCNN_v2 inference time: {time.perf_counter() - infer_start_time:.2f} s")

    # Automatic metrics if the evaluator is available
    if HAS_METRICS:
        try:
            pred, _ = read_raster(out_path)
            ref_path = TRAIN_PAIRS[0][1] if TRAIN_PAIRS else None
            if ref_path is None:
                print("No reference found for automatic metrics.")
                return
            ref, _ = read_raster(Path(ref_path))
            stats, ref_means = compute_band_metrics(pred, ref, data_range=2.0)
            sam_mean, sam_median = spectral_angle_mapper(pred, ref)
            sid_mean = spectral_information_divergence(pred, ref)
            ergas_val = ergas(stats, ref_means, scale_ratio=1.0)

            print("\nQuick metrics vs drone (TRAIN_PAIRS[0]):")
            print(metrics_format_table(stats))
            print("\n--- Global metrics ---")
            print(f"SAM (mean/median) [degrees]: {sam_mean:.3f} / {sam_median:.3f}")
            print(f"SID (mean): {sid_mean:.6f}")
            print(f"ERGAS: {ergas_val:.3f}")
        except Exception as e:
            print(f"Failed to calculate automatic metrics: {e}")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    # 1) Train the chosen model (fsrcnn or fsrcnn-s)
    train_fsrcnn()

    # 2) Apply to a Sentinel image (for example the first in the list)
    sentinel_test = TRAIN_PAIRS[0][0]
    # Generate output name with resolution and number of training dates
    out_sr_path = OUT_DIR / f"fsrcnn_{MODEL_VARIANT}_{_res_str}_{_dates_str}.tif"
    apply_fsrcnn_to_full_image(sentinel_test, out_sr_path)









