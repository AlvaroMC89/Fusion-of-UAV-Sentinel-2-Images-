#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference:
J. Kim, J. K. Lee and K. M. Lee, "Accurate Image Super-Resolution Using Very Deep Convolutional Networks,
" 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 2016, 
pp. 1646-1654, doi: 10.1109/CVPR.2016.182.

VDSR – Very Deep Super-Resolution adapted for:
    - Input    : Sentinel-2 compositions (NDVI, GNDVI, NDRE, NDWI, NDVIre, CIre) at 0.5 m
    - Target   : drone compositions at 0.5 m (same bands, same grid)

The model learns a NON-LINEAR MAPPING:
    Sentinel_0p5m  -->  Drone_0p5m

Architecture:
    - 1 initial conv       (C -> 64, 3x3, ReLU)
    - 18 intermediate convs (64 -> 64, 3x3, ReLU)
    - 1 final conv         (64 -> C, 3x3, without ReLU)
    - Residual: out = x + net(x)

Trained with MSE between (x + f(x)) and y (drone).
"""

import os
import time
from pathlib import Path
import random
import numpy as np
import rasterio

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

import math

# Nuevo: imports para evaluación y visualización
import matplotlib.pyplot as plt
from evaluate_metrics import (
    read_raster,
    compute_band_metrics,
    spectral_angle_mapper,
    spectral_information_divergence,
    ergas,
    format_table,
    save_csv,
)


# ---------------- Configuration ----------------
# List of Sentinel–Drone pairs (you can add more dates)
TRAIN_PAIRS = [
    # (sentinel_path, drone_path)
    (
         r"D:\Nueva carpeta\2025\Fusion de datos\Articulo 1\output_recorte_prev_remuestreo\sentinel\20250813_comp_sentinel_2m.tif",
        r"D:\Nueva carpeta\2025\Fusion de datos\Articulo 1\output_recorte_prev_remuestreo\dron\20250813_comp_dron_2m.tif"

    ),
   
]

# Split into train/test
TEST_RATIO = 0.2
ALL_PAIRS = TRAIN_PAIRS.copy()
SEED = 2025
random.seed(SEED)
random.shuffle(ALL_PAIRS)
num_total = len(ALL_PAIRS)

# Automatically extract resolution from filenames (e.g., '0p5m','0.5m','2m')
import re
from collections import Counter

def extract_resolution_from_string(s: str):
    if not s:
        return None
    m = re.search(r"(\d+(?:[.,]\d+|p\d+)?m)", s, flags=re.IGNORECASE)
    if not m:
        return None
    res = m.group(1).lower()
    res = res.replace('p', '.').replace(',', '.')
    return res


def get_resolution_from_pairs(pairs):
    candidates = []
    for s_path, d_path in pairs:
        for p in (s_path, d_path):
            try:
                stem = Path(p).stem
            except Exception:
                stem = str(p)
            r = extract_resolution_from_string(stem)
            if r:
                candidates.append(r)
            else:
                r = extract_resolution_from_string(str(p))
                if r:
                    candidates.append(r)
    if not candidates:
        return 'unknown'
    return Counter(candidates).most_common(1)[0][0]

RESOLUTION = get_resolution_from_pairs(ALL_PAIRS)

num_val = max(1, int(num_total * TEST_RATIO)) if num_total > 1 else 0
VAL_PAIRS = ALL_PAIRS[:num_val]
TRAIN_PAIRS = ALL_PAIRS[num_val:]

# Folder to save model and outputs
OUT_DIR = Path(r"D:\Nueva carpeta\2025\Fusion de datos\vdsr_model")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Dynamic model name integrating resolution and number of dates
MODEL_FILENAME = f"vdsr_res{RESOLUTION}_dates{num_total}.pth"
MODEL_PATH = OUT_DIR / MODEL_FILENAME

# Hyperparameters for training
PATCH_SIZE = 96
PATCHES_PER_IMAGE = 500
BATCH_SIZE = 16
NUM_EPOCHS = 1000
LEARNING_RATE = 0.001
GRAD_CLIP_NORM = 1.0
LR_STEP_SIZE = 50
LR_GAMMA = 0.5
CHECKPOINT_INTERVAL = 5
MAX_PATCH_RETRIES = 20
NUM_WORKERS = 4
INDEX_MIN = -1.0
INDEX_MAX = 1.0
NORMALIZE_BANDS =False

# Number of channels 
NUM_CHANNELS = 7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
PIN_MEMORY = DEVICE.type == "cuda"
torch.backends.cudnn.benchmark = True


def clip_to_index_range(arr: np.ndarray) -> np.ndarray:
    """
    Clips index values to the physical window [-1, 1] respecting NaNs.
    """
    return np.clip(arr, INDEX_MIN, INDEX_MAX)


def set_seed(seed: int = SEED):
    """
    Sets the seeds for better reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ---------------- VDSR MODEL ----------------
class VDSR(nn.Module):
    def __init__(self, num_channels=1, num_layers=20):
        super(VDSR, self).__init__()

        # Initial layer
        layers = [
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        ]

        # Intermediate layers
        for _ in range(num_layers - 2):
            layers.append(
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
            )
            layers.append(nn.ReLU(inplace=True))

        # Final layer
        layers.append(
            nn.Conv2d(64, num_channels, kernel_size=3, padding=1, bias=True)
        )

        self.net = nn.Sequential(*layers)

        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: tensor (B, C, H, W)
        VDSR learns the residual: out = x + f(x)
        """
        residual = self.net(x)
        return x + residual


# ---------------- DATASET ----------------
class SentinelDronDataset(Dataset):
    """
    Sentinel–Drone patches dataset.
    Reads GeoTIFF pairs (same size, CRS, and transform).
    Extracts random patches for training.
    """
    def __init__(self, pairs, patch_size=64, num_patches_per_image=200,
                 normalize=True, max_invalid_retry=MAX_PATCH_RETRIES):
        """
        pairs: list of (sentinel_path, drone_path)
        num_patches_per_image: how many patches to extract from each pair per epoch
        """
        self.pairs = pairs
        self.patch_size = patch_size
        self.num_patches = num_patches_per_image * len(pairs)
        self.normalize = normalize
        self.max_invalid_retry = max_invalid_retry

        # Pre-load full images into RAM (if not huge)
        self.data = []
        for s_path, d_path in pairs:
            with rasterio.open(s_path) as s_src, rasterio.open(d_path) as d_src:
                s_arr = s_src.read().astype(np.float32)  # (C,H,W)
                d_arr = d_src.read().astype(np.float32)

                # replace nodata with NaN if exists
                if s_src.nodata is not None:
                    s_arr = np.where(s_arr == s_src.nodata, np.nan, s_arr)
                if d_src.nodata is not None:
                    d_arr = np.where(d_arr == d_src.nodata, np.nan, d_arr)

                s_arr = clip_to_index_range(s_arr)
                d_arr = clip_to_index_range(d_arr)

                # Option: normalize per band (mean 0, std 1)
                if normalize:
                    for b in range(s_arr.shape[0]):
                        v = s_arr[b]
                        mask = np.isfinite(v)
                        if np.any(mask):
                            mu = np.nanmean(v[mask])
                            sd = np.nanstd(v[mask]) + 1e-6
                            s_arr[b] = (v - mu) / sd

                    for b in range(d_arr.shape[0]):
                        v = d_arr[b]
                        mask = np.isfinite(v)
                        if np.any(mask):
                            mu = np.nanmean(v[mask])
                            sd = np.nanstd(v[mask]) + 1e-6
                            d_arr[b] = (v - mu) / sd

                self.data.append((s_arr, d_arr))

        if not self.data:
            raise ValueError("No Sentinel–Drone pairs loaded.")

        # assume all have the same size
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
            # Choose a random pair
            pair_idx = random.randint(0, len(self.data) - 1)
            s_arr, d_arr = self.data[pair_idx]  # (C,H,W)

            # Choose valid patch coordinates
            r0 = random.randint(0, max_row)
            c0 = random.randint(0, max_col)

            s_patch = s_arr[:, r0:r0+ps, c0:c0+ps]
            d_patch = d_arr[:, r0:r0+ps, c0:c0+ps]

            # If there are many NaNs, retry
            if not np.isfinite(s_patch).any() or not np.isfinite(d_patch).any():
                continue

            # Replace NaN with 0 (since we normalized, 0 ~ mean)
            s_patch = np.nan_to_num(s_patch, nan=0.0)
            d_patch = np.nan_to_num(d_patch, nan=0.0)

            # to torch tensors
            s_t = torch.from_numpy(s_patch)  # (C,ps,ps)
            d_t = torch.from_numpy(d_patch)

            return s_t.float(), d_t.float()

        raise RuntimeError("Failed to extract a valid patch after several attempts.")


# ---------------- TRAINING ----------------
def create_dataloader(dataset, shuffle=True, drop_last=True):
    """
    Constructs a DataLoader optimized for GPU (pin_memory) when available.
    """
    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        drop_last=drop_last,
        pin_memory=PIN_MEMORY
    )
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def compute_psnr(mse, data_range=INDEX_MAX - INDEX_MIN):
    if mse == 0:
        return float('inf')
    return 10.0 * math.log10((data_range ** 2) / mse)


# Interactive plotting functions
try:
    plt.ion()
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def init_training_plot():
    """
    Initializes an interactive figure to monitor Loss and PSNR.
    Returns (fig, ax_loss, ax_psnr) or None if matplotlib is not available.
    """
    if not HAS_MPL:
        return None
    try:
        plt.ion()
    except Exception:
        pass
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.set_title("VDSR Training Progress")
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
    try:
        fig, ax1, ax2 = handles
    except Exception:
        return

    ax1.cla()
    ax2.cla()
    ax1.set_title("VDSR Training Progress")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax2.set_ylabel("PSNR (dB)", color="tab:orange")

    # Plot loss
    if hasattr(epochs, '__len__') and len(epochs) > 0 and len(train_losses) > 0:
        try:
            ax1.plot(epochs, train_losses, color="tab:blue", label="Loss (train)")
        except Exception:
            pass
    if eval_losses is not None and len(eval_losses) > 0:
        try:
            ax1.plot(epochs, eval_losses, color="tab:green", linestyle="--", label="Loss (val)")
        except Exception:
            pass

    ax1.grid(True, linestyle="--", alpha=0.4)

    # Plot PSNR
    if len(train_psnrs) > 0:
        try:
            ax2.plot(epochs, train_psnrs, color="tab:orange", label="PSNR (train)")
        except Exception:
            pass
    if eval_psnrs is not None and len(eval_psnrs) > 0:
        try:
            ax2.plot(epochs, eval_psnrs, color="tab:red", linestyle="--", label="PSNR (val)")
        except Exception:
            pass

    try:
        fig.legend(loc="upper right")
    except Exception:
        pass

    try:
        fig.canvas.draw()
        fig.canvas.flush_events()
    except Exception:
        pass


def train_vdsr():
    set_seed(SEED)
    if len(TRAIN_PAIRS) == 0:
        raise ValueError("No training pairs available after splitting train/val. Make sure you have enough dates.")

    train_start_time = time.perf_counter()

    # Initialize interactive figure
    handles = init_training_plot()

    # Create full dataset from TRAIN_PAIRS
    full_dataset = SentinelDronDataset(
        TRAIN_PAIRS,
        patch_size=PATCH_SIZE,
        num_patches_per_image=PATCHES_PER_IMAGE,
        normalize=NORMALIZE_BANDS,
        max_invalid_retry=MAX_PATCH_RETRIES
    )

    train_loader = None
    val_loader = None

    # If there are separate validation pairs, create loader for them
    if len(VAL_PAIRS) > 0:
        train_dataset = full_dataset
        train_loader = create_dataloader(train_dataset, shuffle=True, drop_last=True)

        val_dataset = SentinelDronDataset(
            VAL_PAIRS,
            patch_size=PATCH_SIZE,
            num_patches_per_image=PATCHES_PER_IMAGE // 2,
            normalize=NORMALIZE_BANDS,
            max_invalid_retry=MAX_PATCH_RETRIES
        )
        val_loader = create_dataloader(val_dataset, shuffle=False, drop_last=False)
    else:
        # No validation pairs: split patches from the same dataset into train/val
        total_patches = len(full_dataset)
        val_count = max(1, int(total_patches * TEST_RATIO)) if total_patches > 1 else 0
        train_count = total_patches - val_count
        if val_count > 0 and train_count > 0:
            train_subset, val_subset = torch.utils.data.random_split(
                full_dataset,
                [train_count, val_count],
                generator=torch.Generator().manual_seed(SEED),
            )
            train_loader = create_dataloader(train_subset, shuffle=True, drop_last=True)
            val_loader = create_dataloader(val_subset, shuffle=False, drop_last=False)
        else:
            # Not possible to create validation (dataset too small), use all for training
            train_loader = create_dataloader(full_dataset, shuffle=True, drop_last=True)
            val_loader = None

    model = VDSR(num_channels=NUM_CHANNELS, num_layers=20).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    # use the new torch.amp API to avoid FutureWarning
    try:
        scaler = torch.amp.GradScaler(device_type='cuda', enabled=USE_AMP)
    except Exception:
        # simple fallback
        scaler = torch.amp.GradScaler() if USE_AMP else torch.amp.GradScaler(enabled=False)

    # ensure train_dataset is defined for the message
    try:
        td_len = len(train_dataset)
    except Exception:
        td_len = len(full_dataset)
    print(f"Training on {DEVICE} with {td_len} total patches (val_pairs={len(VAL_PAIRS)})...")
    best_loss = float("inf")

    # History for plotting
    train_losses = []
    val_losses = []
    train_psnrs = []
    val_psnrs = []

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for s_batch, d_batch in train_loader:
            s_batch = s_batch.to(DEVICE, non_blocking=PIN_MEMORY)  # Sentinel (input)
            d_batch = d_batch.to(DEVICE, non_blocking=PIN_MEMORY)  # Dron (target)

            optimizer.zero_grad(set_to_none=True)

            # VDSR predicts Y ~ Dron from X=Sentinel
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                out = model(s_batch)
                loss = criterion(out, d_batch)

            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        train_losses.append(avg_loss)

        # Quick training PSNR calculation (per-batch average using MSE)
        train_psnr = None
        try:
            train_psnr = compute_psnr(avg_loss)
        except Exception:
            train_psnr = float('nan')
        train_psnrs.append(train_psnr)

        # Validation
        val_loss_epoch = float('nan')
        val_psnr_epoch = float('nan')
        if val_loader is not None:
            model.eval()
            v_loss = 0.0
            v_batches = 0
            v_mse_accum = 0.0
            with torch.no_grad():
                for s_batch, d_batch in val_loader:
                    s_batch = s_batch.to(DEVICE, non_blocking=PIN_MEMORY)
                    d_batch = d_batch.to(DEVICE, non_blocking=PIN_MEMORY)
                    with torch.amp.autocast('cuda', enabled=USE_AMP):
                        out = model(s_batch)
                        loss = criterion(out, d_batch)
                    v_loss += loss.item()
                    v_batches += 1

                    # MSE per batch (en CPU numpy)
                    mse_batch = torch.mean((out - d_batch) ** 2).item()
                    v_mse_accum += mse_batch

            if v_batches > 0:
                val_loss_epoch = v_loss / v_batches
                val_losses.append(val_loss_epoch)
                val_psnr_epoch = compute_psnr(v_mse_accum / v_batches)
                val_psnrs.append(val_psnr_epoch)

        else:
            # Keep lists synchronized for plotting
            val_losses.append(float('nan'))
            val_psnrs.append(float('nan'))

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch}/{NUM_EPOCHS}] Loss: {avg_loss:.6f} | Val: {val_loss_epoch if not np.isnan(val_loss_epoch) else 'N/A'} | LR: {current_lr:.2e}")

        scheduler.step()

        epoch_time = time.perf_counter() - epoch_start_time

        # Save model when improved or periodically
        prev_best = best_loss
        save_model = False
        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            save_model = True
        elif epoch % CHECKPOINT_INTERVAL == 0 or epoch == NUM_EPOCHS:
            save_model = True

        if save_model:
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Model saved to: {MODEL_PATH} (best_loss={best_loss:.6f}) | Epoch time: {epoch_time:.2f} s")

    # Save final plots
    epochs = list(range(1, len(train_losses) + 1))
    if HAS_MPL:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        import numpy as _np
        e = _np.array(epochs)
        t_losses = _np.array(train_losses, dtype=float)
        v_losses = _np.array(val_losses, dtype=float)
        if t_losses.size > 0:
            plt.plot(e, t_losses, label='train_loss')
        if v_losses.size > 0:
            # align lengths
            plt.plot(e, v_losses[: e.size], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        t_ps = _np.array(train_psnrs, dtype=float)
        v_ps = _np.array(val_psnrs, dtype=float)
        if t_ps.size > 0:
            plt.plot(e, t_ps, label='train_psnr')
        if v_ps.size > 0:
            plt.plot(e, v_ps[: e.size], label='val_psnr')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        plt.legend()
        plt.grid(True)

        plot_path = OUT_DIR / f"training_metrics_res{RESOLUTION}_dates{num_total}.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Final metrics plot saved to: {plot_path}")

    total_train_time = time.perf_counter() - train_start_time
    print(f"Total VDSR training time: {total_train_time:.2f} s")
    print("Training finished.")


# ---------------- INFERENCIA ----------------
@torch.inference_mode()
def apply_vdsr_to_full_image(sentinel_path, out_path, model_path=None, evaluate=True, csv_out=None):
    """
    Apply trained VDSR to a full Sentinel composition (0.5 m).
    Generates a "super-resolved" drone-like image as output.
    If evaluate=True, evaluation is performed using evaluate_metrics.py comparing with the reference (if provided).
    """
    infer_start_time = time.perf_counter()
    model = VDSR(num_channels=NUM_CHANNELS, num_layers=20).to(DEVICE)
    load_path = model_path if model_path is not None else MODEL_PATH
    model.load_state_dict(torch.load(load_path, map_location=DEVICE))
    model.eval()

    with rasterio.open(sentinel_path) as src:
        prof = src.profile.copy()
        arr = src.read().astype(np.float32)  # (C,H,W)

        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        arr = clip_to_index_range(arr)

        if NORMALIZE_BANDS:
            # Normalize same as in training (here simplified: per band)
            for b in range(arr.shape[0]):
                v = arr[b]
                mask = np.isfinite(v)
                if np.any(mask):
                    mu = np.nanmean(v[mask])
                    sd = np.nanstd(v[mask]) + 1e-6
                    arr[b] = (v - mu) / sd
                else:
                    arr[b] = np.zeros_like(v)

        H, W = arr.shape[1], arr.shape[2]
        # Process by tiles if the image is very large (here simple: all at once)
        inp = torch.from_numpy(np.nan_to_num(arr, nan=0.0)).unsqueeze(0).to(DEVICE)  # (1,C,H,W)

        with torch.amp.autocast('cuda', enabled=USE_AMP):
            out = model(inp)  # (1,C,H,W)

        out_np = out.squeeze(0).cpu().numpy().astype(np.float32)
        out_np = clip_to_index_range(out_np)

        # Simplified de-normalization: here you could also save mu/sigma
        # per band of the original Sentinel if you want to return exactly to the original scale.
        # For now we leave it in "normalized scale."

        # Replace NaN with nodata
        out_np = np.nan_to_num(out_np, nan=-9999.0)

        prof.update(
            dtype="float32",
            count=out_np.shape[0],
            nodata=-9999.0,
            compress="deflate",
            tiled=True,
            blockxsize=512,
            blockysize=512
        )

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path.as_posix(), "w", **prof) as dst:
            dst.write(out_np)
            # optional: band names
            for i in range(out_np.shape[0]):
                dst.set_band_description(i+1, f"Band_{i+1}")

        print(f"Imagen VDSR-SR guardada en: {out_path}")
        print(f"Tiempo de inferencia VDSR: {time.perf_counter() - infer_start_time:.2f} s")

    
    # Evaluation with evaluate_metrics if a reference with a similar suffix is available
    if evaluate:
        # try to locate reference with the same stem in known drone folders
        ref_candidates = []
        # search in ALL_PAIRS for matching sentinel or drone name
        for s_path, d_path in ALL_PAIRS + VAL_PAIRS + TRAIN_PAIRS:
            if Path(s_path).stem == Path(sentinel_path).stem:
                ref_candidates.append(d_path)
            if Path(d_path).stem == Path(sentinel_path).stem:
                ref_candidates.append(d_path)

        if ref_candidates:
            ref_path = ref_candidates[0]
            try:
                # use evaluate_metrics functions to display metrics
                pred_arr, pred_desc = read_raster(out_path)
                ref_arr, ref_desc = read_raster(Path(ref_path))
                stats, ref_means = compute_band_metrics(pred_arr, ref_arr, data_range=INDEX_MAX-INDEX_MIN)
                descriptions = pred_desc if any(pred_desc) else ref_desc
                print(format_table(stats, descriptions))

                sam_mean, sam_median = spectral_angle_mapper(pred_arr, ref_arr)
                sid_mean = spectral_information_divergence(pred_arr, ref_arr)
                ergas_val = ergas(stats, ref_means, scale_ratio=1.0)

                print("\n--- Global metrics (post-inference) ---")
                print(f"SAM (mean/median) [degrees]: {sam_mean:.3f} / {sam_median:.3f}" if np.isfinite(sam_mean) else "SAM: no valid data")
                print(f"SID (mean): {sid_mean:.6f}" if np.isfinite(sid_mean) else "SID: no valid data")
                print(f"ERGAS: {ergas_val:.3f}" if np.isfinite(ergas_val) else "ERGAS: no valid data")

                if csv_out:
                    save_csv(stats, Path(csv_out), descriptions)
                    print(f"Metrics saved in: {csv_out}")

            except Exception as e:
                print(f"Could not evaluate the inferred image: {e}")
        else:
            print("No automatic reference found for evaluation. Provide the reference manually if you have it.")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    # 1) Train:
    train_vdsr()

    # 2) Apply to a Sentinel image (can be a training one or another):
    # Try to infer using the first available Sentinel and evaluate
    try:
        if ALL_PAIRS:
            sentinel_sample = ALL_PAIRS[0][0]
            out_inf_path = OUT_DIR / f"{num_total}_vdsr_{RESOLUTION}.tif"
            metrics_csv = OUT_DIR / f"{num_total}_vdsr_metrics.csv"
            apply_vdsr_to_full_image(sentinel_sample, out_inf_path, model_path=MODEL_PATH, evaluate=True, csv_out=str(metrics_csv))
    except Exception as e:
        print(f"Could not execute automatic inference: {e}")
