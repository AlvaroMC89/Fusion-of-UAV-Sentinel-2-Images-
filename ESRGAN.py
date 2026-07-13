#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference:
Wang, X. et al. (2019). ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks. 
In: Leal-Taixé, L., Roth, S. (eds) Computer Vision – ECCV 2018 Workshops. ECCV 2018. 
Lecture Notes in Computer Science(), vol 11133. Springer, Cham. 
https://doi.org/10.1007/978-3-030-11021-5_5

ESRGAN minimal implementation adapted for multiband vegetation-index inputs.
- Generator: RRDBNet (simplified)
- Discriminator: Patch-style CNN
- Loss: L1 content + perceptual (optional VGG) + adversarial (LSGAN)

Notes:
Designed to read multiband Sentinel–Drone pairs (same grid) and train on patches.

Saves models with resolution and number of dates in the filename.

Includes an inference function that integrates evaluate_metrics.py for post-inference evaluation.

This script is a starting point; hyperparameters and architecture can be adjusted.
"""

from pathlib import Path
import random
import re
from collections import Counter
import math
import os
import time
import numpy as np
import rasterio
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

try:
    from torchvision import models
    HAS_TORCHVISION = True
except Exception:
    HAS_TORCHVISION = False

try:
    from evaluate_metrics import read_raster, compute_band_metrics, spectral_angle_mapper, spectral_information_divergence, ergas, format_table, save_csv
    HAS_METRICS = True
except Exception:
    HAS_METRICS = False

import matplotlib
matplotlib.use('Agg')  # non-interactive backend suitable for scripts
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
# PAIRS (edit according to folder)
TRAIN_PAIRS = [
    (
        r"D:\Nueva carpeta\2025\Fusion de datos\Articulo 1\output_recorte_prev_remuestreo\sentinel\20250813_comp_sentinel_2m.tif",
        r"D:\Nueva carpeta\2025\Fusion de datos\Articulo 1\output_recorte_prev_remuestreo\dron\20250813_comp_dron_2m.tif"
    )
]

TEST_RATIO = 0.2
# change to validation ratio (used to split train/val)
VAL_RATIO = 0.15
ALL_PAIRS = TRAIN_PAIRS.copy()
SEED = 2025
random.seed(SEED)
random.shuffle(ALL_PAIRS)
num_total = len(ALL_PAIRS)

# extract resolution helper
def extract_resolution_from_string(s: str):
    if not s:
        return None
    m = re.search(r"(\d+(?:[.,]\d+|p\d+)?m)", s, flags=re.IGNORECASE)
    if not m:
        return None
    res = m.group(1).lower()
    res = res.replace('p', '.').replace(',', '.')
    return res

from collections import Counter

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

# replace previous num_val logic to use VAL_RATIO
num_val = max(1, int(num_total * VAL_RATIO)) if num_total > 1 else 0
VAL_PAIRS = ALL_PAIRS[:num_val]
TRAIN_PAIRS = ALL_PAIRS[num_val:]

OUT_DIR = Path(r"D:\\Nueva carpeta\\2025\\Fusion de datos\\gan_model")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GEN_FILENAME = f"esrgan_gen_res{RESOLUTION}_dates{num_total}.pth"
DIS_FILENAME = f"esrgan_dis_res{RESOLUTION}_dates{num_total}.pth"
GEN_PATH = OUT_DIR / GEN_FILENAME
DIS_PATH = OUT_DIR / DIS_FILENAME

# training params
PATCH_SIZE = 96
PATCHES_PER_IMAGE = 300
BATCH_SIZE = 8
NUM_EPOCHS = 200
LR = 1e-4
NUM_CHANNELS = 7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = DEVICE.type == 'cuda'
PIN_MEMORY = DEVICE.type == 'cuda'

# ESRGAN-specific hyperparams (paper-like defaults)
UPSCALE = 1  # ESRGAN in paper performs x4 upsampling; set to 1 here because inputs are same resolution. Set to 4 to perform super-resolution.
NB_RRDB = 23  # number of RRDB blocks in the original ESRGAN (heavy)
NF = 64       # number of feature maps
RESIDUAL_SCALING = 0.2  # residual scaling inside RRDB
PRETRAIN_EPOCHS = 100  # first train generator only (L1) like in the paper
PERCEPTUAL_WEIGHT = 0.01
ADVERSARIAL_WEIGHT = 0.005

# ---------------- MODELS ----------------
class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels=NF, growth=32, residual_scaling=RESIDUAL_SCALING):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels+growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels+2*growth, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channels+3*growth, in_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.res_scale = residual_scaling

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.conv4(torch.cat([x, x1, x2, x3], 1))
        return x + self.res_scale * x4

class RRDB(nn.Module):
    def __init__(self, in_channels=NF, growth=32, residual_scaling=RESIDUAL_SCALING):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(in_channels, growth, residual_scaling)
        self.rdb2 = ResidualDenseBlock(in_channels, growth, residual_scaling)
        self.rdb3 = ResidualDenseBlock(in_channels, growth, residual_scaling)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + RESIDUAL_SCALING * out

class RRDBNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, nf=NF, nb=NB_RRDB, upscale=UPSCALE):
        super().__init__()
        self.upscale = upscale
        self.conv_first = nn.Conv2d(in_channels, nf, 3, 1, 1)
        # trunk of RRDB blocks
        self.trunk = nn.Sequential(*[RRDB(nf) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)

        # reconstruction head
        self.recon = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, out_channels * (upscale ** 2) if upscale > 1 else out_channels, 3, 1, 1)
        )

        # if upscaling, use pixel shuffle
        if upscale > 1:
            self.pixel_shuffle = nn.PixelShuffle(upscale)
        else:
            self.pixel_shuffle = None

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk(fea)
        trunk = self.trunk_conv(trunk)
        fea = fea + trunk
        out = self.recon(fea)
        if self.pixel_shuffle is not None:
            out = self.pixel_shuffle(out)
        # residual learning
        # if upscaling, need to upsample input to add residual
        if self.upscale > 1:
            inp_up = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
            return inp_up + out
        else:
            return x + out

# Discriminator remains similar but keep capacity
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, nf=64):
        super().__init__()
        layers = []
        def conv_block(in_c, out_c, stride=1):
            return [nn.Conv2d(in_c, out_c, 3, stride, 1), nn.LeakyReLU(0.2, inplace=True)]
        layers += conv_block(in_channels, nf, 1)
        layers += conv_block(nf, nf, 2)
        layers += conv_block(nf, nf*2, 1)
        layers += conv_block(nf*2, nf*2, 2)
        layers += conv_block(nf*2, nf*4, 1)
        layers += conv_block(nf*4, nf*4, 2)
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(nf*4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# perceptual feature extractor (VGG) - 
class VGGFeatureExtractor(nn.Module):
    def __init__(self, use_cuda=False):
        super().__init__()
        if not HAS_TORCHVISION:
            raise RuntimeError('torchvision required for perceptual loss')
        vgg = models.vgg19(pretrained=True).features
        # ESRGAN uses features before activation in the original paper; torchvision provides layers with ReLU.
        # Exact "pre-activation" features are not directly accessible from torchvision pre-built model; 
        # we use relu-based features as a practical approximation.
        # Also VGG expects 3-channel RGB; for multiband inputs we will use the first 3 bands if available. 
        # If NUM_CHANNELS!=3 the perceptual loss is only applied to the RGB subset.
        self.feature = nn.Sequential(*list(vgg.children())[:35]).eval()  # up to relu5_4 (approx)
        for p in self.feature.parameters():
            p.requires_grad = False
        if use_cuda:
            self.feature = self.feature.to(DEVICE)

    def forward(self, x):
        # x expected in range [0,1] or normalized appropriately; many implementations normalize to ImageNet stats.
        # Here we assume x in [-1,1] or [0,1]; adapt if necessary before calling.
        return self.feature(x)

# ---------------- Dataset ----------------
class SentinelDronDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, patch_size=64, num_patches_per_image=100, normalize=False, max_retry=10):
        self.pairs = pairs
        self.patch_size = patch_size
        self.num_patches = num_patches_per_image * len(pairs)
        self.normalize = normalize
        self.max_retry = max_retry
        self.data = []
        for s_path, d_path in pairs:
            with rasterio.open(s_path) as s_src, rasterio.open(d_path) as d_src:
                s = s_src.read().astype(np.float32)
                d = d_src.read().astype(np.float32)
                if s_src.nodata is not None:
                    s = np.where(s == s_src.nodata, np.nan, s)
                if d_src.nodata is not None:
                    d = np.where(d == d_src.nodata, np.nan, d)
                # if image is smaller than requested patch size, pad with NaNs so later we can replace with zeros
                h = s.shape[1]
                w = s.shape[2]
                pad_h = max(0, patch_size - h)
                pad_w = max(0, patch_size - w)
                if pad_h > 0 or pad_w > 0:
                    s = np.pad(s, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=np.nan)
                    d = np.pad(d, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=np.nan)
                self.data.append((s, d))
        if not self.data:
            raise ValueError('No pairs loaded')
        # use dimensions from first (padded) sample
        self.C = self.data[0][0].shape[0]
        self.H = self.data[0][0].shape[1]
        self.W = self.data[0][0].shape[2]

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        ps = self.patch_size
        # allow zero range when image equals patch size
        max_r = max(0, self.H - ps)
        max_c = max(0, self.W - ps)
        for _ in range(self.max_retry):
            pidx = random.randint(0, len(self.data) - 1)
            s, d = self.data[pidx]
            r0 = 0 if max_r == 0 else random.randint(0, max_r)
            c0 = 0 if max_c == 0 else random.randint(0, max_c)
            s_patch = s[:, r0:r0 + ps, c0:c0 + ps]
            d_patch = d[:, r0:r0 + ps, c0:c0 + ps]
            # require at least some finite pixels in both patches
            if (not np.isfinite(s_patch).any()) or (not np.isfinite(d_patch).any()):
                continue
            s_patch = np.nan_to_num(s_patch, nan=0.0)
            d_patch = np.nan_to_num(d_patch, nan=0.0)
            s_t = torch.from_numpy(s_patch).float()
            d_t = torch.from_numpy(d_patch).float()
            return s_t, d_t

        # fallback: return center patch (with NaNs replaced) instead of raising to avoid DataLoader worker failure
        pidx = random.randint(0, len(self.data) - 1)
        s, d = self.data[pidx]
        r0 = max(0, (self.H - ps) // 2)
        c0 = max(0, (self.W - ps) // 2)
        s_patch = s[:, r0:r0 + ps, c0:c0 + ps]
        d_patch = d[:, r0:r0 + ps, c0:c0 + ps]
        s_patch = np.nan_to_num(s_patch, nan=0.0)
        d_patch = np.nan_to_num(d_patch, nan=0.0)
        s_t = torch.from_numpy(s_patch).float()
        d_t = torch.from_numpy(d_patch).float()
        return s_t, d_t

# ---------------- UTIL ----------------

def create_dataloader(dataset, shuffle=True, drop_last=True):
    kwargs = dict(batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=4, drop_last=drop_last, pin_memory=PIN_MEMORY)
    return torch.utils.data.DataLoader(dataset, **kwargs)

# ---------------- TRAIN ----------------

def train_esrgan():
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_start_time = time.perf_counter()

    if len(TRAIN_PAIRS) == 0:
        raise ValueError('No training pairs available')

    # create dataset
    full_ds = SentinelDronDataset(TRAIN_PAIRS, patch_size=PATCH_SIZE, num_patches_per_image=PATCHES_PER_IMAGE)

    if len(VAL_PAIRS) > 0:
        train_ds = full_ds
        train_loader = create_dataloader(train_ds, shuffle=True)
        val_ds = SentinelDronDataset(VAL_PAIRS, patch_size=PATCH_SIZE, num_patches_per_image=PATCHES_PER_IMAGE//2)
        val_loader = create_dataloader(val_ds, shuffle=False, drop_last=False)
    else:
        total = len(full_ds)
        val_count = max(1, int(total*TEST_RATIO)) if total>1 else 0
        train_count = total - val_count
        if val_count>0 and train_count>0:
            train_subset, val_subset = torch.utils.data.random_split(full_ds, [train_count, val_count], generator=torch.Generator().manual_seed(SEED))
            train_loader = create_dataloader(train_subset, shuffle=True)
            val_loader = create_dataloader(val_subset, shuffle=False, drop_last=False)
        else:
            train_loader = create_dataloader(full_ds, shuffle=True)
            val_loader = None

    # models - use paper-like sizes but keep configurable
    gen = RRDBNet(in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS, nf=NF, nb=NB_RRDB, upscale=UPSCALE).to(DEVICE)
    dis = Discriminator(in_channels=NUM_CHANNELS if UPSCALE==1 else NUM_CHANNELS, nf=NF).to(DEVICE)

    # losses
    l1_loss = nn.L1Loss().to(DEVICE)
    mse_loss = nn.MSELoss().to(DEVICE)
    feat_extractor = None
    if HAS_TORCHVISION and NUM_CHANNELS >= 3:
        try:
            feat_extractor = VGGFeatureExtractor(use_cuda=(DEVICE.type=='cuda'))
        except Exception:
            feat_extractor = None

    opt_g = optim.Adam(gen.parameters(), lr=LR, betas=(0.9, 0.999))
    opt_d = optim.Adam(dis.parameters(), lr=LR, betas=(0.9, 0.999))

    scheduler_g = optim.lr_scheduler.StepLR(opt_g, step_size=100, gamma=0.5)
    scheduler_d = optim.lr_scheduler.StepLR(opt_d, step_size=100, gamma=0.5)

    scaler_g = torch.amp.GradScaler(enabled=USE_AMP)
    scaler_d = torch.amp.GradScaler(enabled=USE_AMP)

    best_g_loss = float('inf')

    # prepare loss tracking
    train_losses_g = []
    val_losses_g = []
    train_losses_d = []
    val_epoch_indices = []
    loss_plot_path = OUT_DIR / 'training_loss_plot.png'

    def save_loss_plot():
        plt.figure(figsize=(8,5))
        plt.plot(train_losses_g, label='Train G', color='tab:blue')
        # plot validation losses at recorded iteration indices
        if val_losses_g and val_epoch_indices:
            plt.plot(val_epoch_indices, val_losses_g, 'o-', label='Val G', color='tab:orange')
        plt.xlabel('Iteration (approx)')
        plt.ylabel('Generator loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(loss_plot_path)
        plt.close()

    # ---------- Phase 1: pretrain generator with L1 (paper recommends pretraining) ----------
    print('Phase 1: pretraining generator (L1) for', PRETRAIN_EPOCHS, 'epochs')
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        epoch_start_time = time.perf_counter()
        gen.train()
        epoch_loss = 0.0
        batches = 0
        for s_batch, d_batch in train_loader:
            s = s_batch.to(DEVICE)
            d = d_batch.to(DEVICE)
            opt_g.zero_grad()
            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                fake = gen(s)
                loss_g = l1_loss(fake, d)
            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()
            epoch_loss += loss_g.item()
            batches += 1
            # track per-batch approx
            train_losses_g.append(loss_g.item())
        avg = epoch_loss / max(1, batches)
        print(f'[Pretrain {epoch}/{PRETRAIN_EPOCHS}] L1: {avg:.6f} | Epoch time: {time.perf_counter() - epoch_start_time:.2f} s')
        save_loss_plot()
        if epoch % 20 == 0:
            torch.save(gen.state_dict(), GEN_PATH)

    # ---------- Phase 2: adversarial training (perceptual + RaGAN) ----------
    print('Phase 2: adversarial training for remaining epochs')
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.perf_counter()
        gen.train(); dis.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        batches = 0
        for s_batch, d_batch in train_loader:
            s = s_batch.to(DEVICE)
            d = d_batch.to(DEVICE)

            # generate
            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                fake = gen(s)

            # -------- Discriminator update (Relativistic average GAN) --------
            opt_d.zero_grad()
            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                pred_real = dis(d)
                pred_fake = dis(fake.detach())
                # RaGAN discriminator loss (LSGAN variant)
                real_mean = torch.mean(pred_real)
                fake_mean = torch.mean(pred_fake)
                loss_d_real = mse_loss(pred_real - fake_mean, torch.ones_like(pred_real))
                loss_d_fake = mse_loss(pred_fake - real_mean, torch.zeros_like(pred_fake))
                loss_d = 0.5 * (loss_d_real + loss_d_fake)
            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            # -------- Generator update (RaGAN + perceptual + content) --------
            opt_g.zero_grad()
            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                pred_fake_for_g = dis(fake)
                pred_real_for_g = dis(d)
                real_mean_g = torch.mean(pred_real_for_g)
                fake_mean_g = torch.mean(pred_fake_for_g)
                # RaGAN generator adversarial loss (LSGAN variant)
                loss_g_adv = 0.5 * (mse_loss(pred_fake_for_g - real_mean_g, torch.ones_like(pred_fake_for_g)) + mse_loss(pred_real_for_g - fake_mean_g, torch.zeros_like(pred_real_for_g)))
                # content
                loss_g_content = l1_loss(fake, d)
                # perceptual
                loss_g_perc = 0.0
                if feat_extractor is not None and NUM_CHANNELS >= 3:
                    try:
                        # feed VGG with first 3 bands scaled to [0,1] if necessary
                        inp_fake_vgg = fake[:, :3, :, :]
                        inp_real_vgg = d[:, :3, :, :]
                        # simple normalization to [0,1] assuming inputs in arbitrary scale; user should adapt to ImageNet norm if used
                        f_fake = feat_extractor((inp_fake_vgg + 1.0) / 2.0) if fake.min() >= -1.0 else feat_extractor(inp_fake_vgg)
                        f_real = feat_extractor((inp_real_vgg + 1.0) / 2.0) if d.min() >= -1.0 else feat_extractor(inp_real_vgg)
                        loss_g_perc = F.l1_loss(f_fake, f_real)
                    except Exception:
                        loss_g_perc = 0.0
                g_loss = loss_g_content + PERCEPTUAL_WEIGHT * loss_g_perc + ADVERSARIAL_WEIGHT * loss_g_adv
            scaler_g.scale(g_loss).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += loss_d.item()
            batches += 1
            train_losses_g.append(g_loss.item())
            train_losses_d.append(loss_d.item())

        # validation pass
        val_g = None
        if val_loader is not None:
            gen.eval()
            val_loss_acc = 0.0
            val_batches = 0
            with torch.no_grad():
                for vs, vd in val_loader:
                    vs = vs.to(DEVICE)
                    vd = vd.to(DEVICE)
                    with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                        v_pred = gen(vs)
                        v_loss = l1_loss(v_pred, vd)
                    val_loss_acc += v_loss.item()
                    val_batches += 1
            val_g = val_loss_acc / max(1, val_batches)
            val_losses_g.append(val_g)
            # record the current training iteration index for plotting (end of epoch)
            val_epoch_indices.append(len(train_losses_g))
            print(f"[Val {epoch}] L1: {val_g:.6f}")

        scheduler_g.step(); scheduler_d.step()
        avg_g = epoch_g_loss / max(1, batches)
        avg_d = epoch_d_loss / max(1, batches)
        print(f"[Epoch {epoch}/{NUM_EPOCHS}] G_loss: {avg_g:.6f} | D_loss: {avg_d:.6f} | Epoch time: {time.perf_counter() - epoch_start_time:.2f} s")

        save_loss_plot()

        # save checkpoints
        if avg_g < best_g_loss - 1e-6 or epoch % 10 == 0 or epoch == NUM_EPOCHS:
            best_g_loss = min(best_g_loss, avg_g)
            # save single checkpoint files (overwrite previous) to avoid accumulating files
            torch.save(gen.state_dict(), GEN_PATH)
            torch.save(dis.state_dict(), DIS_PATH)
            print(f"Models saved: {GEN_PATH.name}, {DIS_PATH.name}")

    total_train_time = time.perf_counter() - train_start_time
    print(f'Total training time ESRGAN: {total_train_time:.2f} s')
    print('Training of ESRGAN finished')

# ---------------- INFERENCE ----------------
# tiled inference helper to avoid OOM on large images / large models
def tiled_infer_model_on_numpy(gen_model, arr_np, device, tile_size=512, overlap=32, use_amp=USE_AMP):
    """
    Run inference on a numpy array (C,H,W) by tiling to avoid OOM.
    Returns numpy array (C,H,W) float32
    """
    gen_model.eval()
    C, H, W = arr_np.shape
    out_sum = np.zeros((C, H, W), dtype=np.float64)
    out_count = np.zeros((1, H, W), dtype=np.float64)

    stride = tile_size - overlap
    ys = list(range(0, H, stride))
    xs = list(range(0, W, stride))

    with torch.no_grad():
        for y in ys:
            for x in xs:
                y0 = y
                x0 = x
                y1 = min(y0 + tile_size, H)
                x1 = min(x0 + tile_size, W)
                if y1 - y0 < tile_size and y0 != 0:
                    y0 = max(0, y1 - tile_size)
                if x1 - x0 < tile_size and x0 != 0:
                    x0 = max(0, x1 - tile_size)

                tile = arr_np[:, y0:y1, x0:x1]
                t = torch.from_numpy(tile).unsqueeze(0).to(device)
                # if model is in half precision, cast input
                if next(gen_model.parameters()).dtype == torch.float16:
                    t = t.half()
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    out_t = gen_model(t)
                out_tile = out_t.squeeze(0).cpu().float().numpy()

                out_sum[:, y0:y1, x0:x1] += out_tile
                out_count[:, y0:y1, x0:x1] += 1.0

    out_count[out_count == 0] = 1.0
    out = (out_sum / out_count).astype(np.float32)
    return out

@torch.inference_mode()
def apply_esrgan_to_full_image(sentinel_path, out_path, gen_path=None, evaluate=True, csv_out=None, tile_size=512, overlap=32):
    """Load generator and apply to full sentinel image. If full-image forward OOMs, falls back to tiled inference.
    tile_size/overlap control the fallback behavior.
    """
    infer_start_time = time.perf_counter()
    # construct generator with training NB_RRDB to match checkpoints
    gen = RRDBNet(in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS, nf=NF, nb=NB_RRDB, upscale=UPSCALE).to(DEVICE)
    load_path = gen_path if gen_path is not None else GEN_PATH

    # load checkpoint on CPU first to avoid GPU deserialization spikes
    state = torch.load(load_path, map_location='cpu')
    try:
        gen.load_state_dict(state)
    except Exception:
        # permissive load if shapes/names differ
        gen.load_state_dict(state, strict=False)

    # move model to device; use half precision on CUDA when available to reduce memory
    if DEVICE.type == 'cuda':
        gen = gen.half().to(DEVICE)
    else:
        gen = gen.to(DEVICE)

    gen.eval()

    with rasterio.open(sentinel_path) as src:
        prof = src.profile.copy()
        arr = src.read().astype(np.float32)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        inp_np = np.nan_to_num(arr, nan=0.0)

    # try full-image forward first
    try:
        inp_t = torch.from_numpy(inp_np).unsqueeze(0).to(DEVICE)
        if DEVICE.type == 'cuda' and next(gen.parameters()).dtype == torch.float16:
            inp_t = inp_t.half()
        with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            out = gen(inp_t)
        out_np = out.squeeze(0).cpu().numpy().astype(np.float32)
    except RuntimeError as e:
        # if OOM or similar, fallback to tiled inference
        print('Full-image inference failed, falling back to tiled inference:', e)
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        out_np = tiled_infer_model_on_numpy(gen, inp_np, DEVICE, tile_size=tile_size, overlap=overlap, use_amp=USE_AMP)

    out_np = np.nan_to_num(out_np, nan=-9999.0)

    prof.update(dtype='float32', count=out_np.shape[0], nodata=-9999.0, compress='deflate', tiled=True, blockxsize=512, blockysize=512)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path.as_posix(), 'w', **prof) as dst:
        dst.write(out_np)
        for i in range(out_np.shape[0]):
            dst.set_band_description(i+1, f'Band_{i+1}')
    print(f'ESRGAN output saved to: {out_path}')
    print(f'Inference time ESRGAN: {time.perf_counter() - infer_start_time:.2f} s')

    if evaluate and HAS_METRICS:
        try:
            # try to locate ref 
            ref_candidates = []
            for s_path, d_path in ALL_PAIRS + VAL_PAIRS + TRAIN_PAIRS:
                if Path(s_path).stem == Path(sentinel_path).stem:
                    ref_candidates.append(d_path)
                if Path(d_path).stem == Path(sentinel_path).stem:
                    ref_candidates.append(d_path)
            if ref_candidates:
                ref = ref_candidates[0]
                pred_arr, pred_desc = read_raster(out_path)
                ref_arr, ref_desc = read_raster(Path(ref))
                stats, ref_means = compute_band_metrics(pred_arr, ref_arr, data_range=1.0)
                print(format_table(stats))
                sam_mean, sam_median = spectral_angle_mapper(pred_arr, ref_arr)
                sid_mean = spectral_information_divergence(pred_arr, ref_arr)
                ergas_val = ergas(stats, ref_means, scale_ratio=1.0)
                print(f"SAM mean: {sam_mean:.3f}, SID: {sid_mean:.6f}, ERGAS: {ergas_val:.3f}")
                if csv_out:
                    save_csv(stats, Path(csv_out), pred_desc)
            else:
                print('No reference found for evaluation')
        except Exception as e:
            print('Evaluation failed:', e)

# ---------------- MAIN ----------------
if __name__ == '__main__':
    train_esrgan()
    try:
        if ALL_PAIRS:
            sentinel_sample = ALL_PAIRS[0][0]
            out_inf = OUT_DIR / f"esrgan_out_{num_total}_{RESOLUTION}.tif"
            apply_esrgan_to_full_image(sentinel_sample, out_inf, gen_path=GEN_PATH, evaluate=True, csv_out=str(OUT_DIR / f"esrgan_metrics_{num_total}.csv"))
    except Exception as e:
        print('Auto inference failed:', e)
