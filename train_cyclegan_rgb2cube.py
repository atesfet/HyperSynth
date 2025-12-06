#!/usr/bin/env python
import argparse
import os
import json
import math
import random
from collections import defaultdict
from datetime import datetime
import functools

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Data handling
# -----------------------------
class HSIRGBCycleSample:
    def __init__(self, patient_id, roi_id, cube_path, rgb_path, wl_path):
        self.patient_id = patient_id
        self.roi_id = roi_id
        self.cube_path = cube_path
        self.rgb_path = rgb_path
        self.wl_path = wl_path


def discover_samples(data_root):
    """
    Traverse data_root and collect all (patient, ROI, cube, rgb, wavelengths) tuples.

    Assumes:
        patient dirs: P1, P2, ..., P13
        ROI dirs inside each patient
        within each ROI dir: cube.npy, original_rgb.png, wavelengths.npy
    """
    samples = []
    patient_to_rois = defaultdict(list)

    for patient in sorted(os.listdir(data_root)):
        patient_path = os.path.join(data_root, patient)
        if not os.path.isdir(patient_path):
            continue

        for roi in sorted(os.listdir(patient_path)):
            roi_path = os.path.join(patient_path, roi)
            if not os.path.isdir(roi_path):
                continue

            cube_path = os.path.join(roi_path, "cube.npy")
            rgb_path = os.path.join(roi_path, "original_rgb.png")
            wl_path = os.path.join(roi_path, "wavelengths.npy")

            if not (os.path.isfile(cube_path) and
                    os.path.isfile(rgb_path) and
                    os.path.isfile(wl_path)):
                continue

            sample = HSIRGBCycleSample(
                patient_id=patient,
                roi_id=roi,
                cube_path=cube_path,
                rgb_path=rgb_path,
                wl_path=wl_path,
            )
            samples.append(sample)
            patient_to_rois[patient].append(roi)

    return samples, patient_to_rois


def split_by_patient(patient_to_rois, train_ratio=0.7, val_ratio=0.2, seed=42):
    """
    Split patients into train/val/test according to given ratios.
    Ensures patient-level separation.
    """
    patients = sorted(list(patient_to_rois.keys()))
    rng = random.Random(seed)
    rng.shuffle(patients)

    n_patients = len(patients)
    n_train = int(round(train_ratio * n_patients))
    n_val = int(round(val_ratio * n_patients))
    n_train = max(1, min(n_train, n_patients - 2))
    n_val = max(1, min(n_val, n_patients - n_train - 1))
    _ = n_patients - n_train - n_val  # n_test (implicit)

    train_patients = patients[:n_train]
    val_patients = patients[n_train:n_train + n_val]
    test_patients = patients[n_train + n_val:]

    splits = {
        "train": train_patients,
        "val": val_patients,
        "test": test_patients,
    }
    return splits


class HSIRGBCycleDataset(Dataset):
    def __init__(
        self,
        samples,
        patch_size=None,
        random_crop=True,
        augment=True,
        normalize_rgb=True,
        normalize_cube="per_cube",  # or "none"
    ):
        """
        samples: list[HSIRGBCycleSample]
        patch_size: int or None, side length of square crop
        random_crop: if True, use random crop, else center crop
        augment: if True, apply random flips
        normalize_rgb: convert to [0,1]
        normalize_cube: "per_cube" or "none"
        """
        self.samples = samples
        self.patch_size = patch_size
        self.random_crop = random_crop
        self.augment = augment
        self.normalize_rgb = normalize_rgb
        self.normalize_cube = normalize_cube

    def __len__(self):
        return len(self.samples)

    def _load_rgb(self, path):
        img = Image.open(path).convert("RGB")
        rgb = np.array(img, dtype=np.float32)
        if self.normalize_rgb:
            rgb = rgb / 255.0
        # H, W, 3 -> 3, H, W
        rgb = np.transpose(rgb, (2, 0, 1))
        return rgb

    def _load_cube(self, path):
        cube = np.load(path).astype(np.float32)  # H, W, C
        if self.normalize_cube == "per_cube":
            c_min = cube.min()
            c_max = cube.max()
            if c_max > c_min:
                cube = (cube - c_min) / (c_max - c_min)
        # H, W, C -> C, H, W
        cube = np.transpose(cube, (2, 0, 1))
        return cube

    def _load_wavelengths(self, path):
        wl = np.load(path).astype(np.float32)  # (C,)
        return wl

    def _random_or_center_crop(self, rgb, cube):
        if self.patch_size is None:
            return rgb, cube

        _, H, W = rgb.shape
        ps = self.patch_size
        if H < ps or W < ps:
            ps = min(H, W)
        if self.random_crop:
            y = random.randint(0, H - ps)
            x = random.randint(0, W - ps)
        else:
            y = (H - ps) // 2
            x = (W - ps) // 2

        rgb = rgb[:, y:y + ps, x:x + ps]
        cube = cube[:, y:y + ps, x:x + ps]
        return rgb, cube

    def _maybe_augment(self, rgb, cube):
        if not self.augment:
            return rgb, cube

        # Horizontal flip
        if random.random() < 0.5:
            rgb = np.flip(rgb, axis=2).copy()
            cube = np.flip(cube, axis=2).copy()
        # Vertical flip
        if random.random() < 0.5:
            rgb = np.flip(rgb, axis=1).copy()
            cube = np.flip(cube, axis=1).copy()

        return rgb, cube

    def __getitem__(self, idx):
        sample = self.samples[idx]
        rgb = self._load_rgb(sample.rgb_path)
        cube = self._load_cube(sample.cube_path)
        wl = self._load_wavelengths(sample.wl_path)  # (C,)

        rgb, cube = self._random_or_center_crop(rgb, cube)
        rgb, cube = self._maybe_augment(rgb, cube)

        rgb = torch.from_numpy(rgb)  # 3,H,W
        cube = torch.from_numpy(cube)  # C,H,W
        wl = torch.from_numpy(wl)  # C

        return {
            "rgb": rgb,
            "cube": cube,
            "wavelengths": wl,
            "patient_id": sample.patient_id,
            "roi_id": sample.roi_id,
        }


def build_dataloaders(
    data_root,
    batch_size=4,
    patch_size=256,
    num_workers=4,
    seed=42,
):
    all_samples, patient_to_rois = discover_samples(data_root)
    splits = split_by_patient(patient_to_rois, 0.7, 0.2, seed=seed)

    patient_split_map = {}
    for split_name, pts in splits.items():
        for p in pts:
            patient_split_map[p] = split_name

    train_samples, val_samples, test_samples = [], [], []
    for s in all_samples:
        split = patient_split_map[s.patient_id]
        if split == "train":
            train_samples.append(s)
        elif split == "val":
            val_samples.append(s)
        else:
            test_samples.append(s)

    # Datasets
    train_dataset = HSIRGBCycleDataset(
        train_samples,
        patch_size=patch_size,
        random_crop=True,
        augment=True,
    )
    val_dataset = HSIRGBCycleDataset(
        val_samples,
        patch_size=patch_size,
        random_crop=False,
        augment=False,
    )
    test_dataset = HSIRGBCycleDataset(
        test_samples,
        patch_size=None,     # full images for visualization
        random_crop=False,
        augment=False,
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, splits, patient_to_rois


# -----------------------------
# CycleGAN Models
# -----------------------------
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type="reflect", norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetGenerator(nn.Module):
    """Standard CycleGAN ResNet generator, but with configurable in/out channels."""
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_layer=nn.InstanceNorm2d,
        use_dropout=False,
        n_blocks=9,
        padding_type="reflect",
    ):
        assert n_blocks >= 0
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # Downsampling
        n_downsampling = 2
        mult = 1
        for _ in range(n_downsampling):
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                          stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]
            mult *= 2

        # ResNet blocks
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=use_dropout)]

        # Upsampling
        for _ in range(n_downsampling):
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
            mult = mult // 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator."""
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


# -----------------------------
# Metrics
# -----------------------------
def mae_metric(pred, target):
    return torch.mean(torch.abs(pred - target)).item()


def rmse_metric(pred, target, eps=1e-8):
    return torch.sqrt(torch.mean((pred - target) ** 2) + eps).item()


def psnr_metric(pred, target, max_val=1.0, eps=1e-8):
    mse = torch.mean((pred - target) ** 2).item()
    if mse < eps:
        return float("inf")
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def sam_metric(pred, target, eps=1e-8):
    """
    Spectral Angle Mapper (average over all pixels).

    pred, target: (B, C, H, W)
    """
    B, C, H, W = pred.shape
    pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)  # (N, C)
    targ_flat = target.permute(0, 2, 3, 1).reshape(-1, C)

    dot = torch.sum(pred_flat * targ_flat, dim=1)
    pred_norm = torch.norm(pred_flat, dim=1)
    targ_norm = torch.norm(targ_flat, dim=1)
    denom = pred_norm * targ_norm + eps
    cos_sim = torch.clamp(dot / denom, -1.0, 1.0)
    angles = torch.acos(cos_sim)  # radians
    return torch.mean(angles).item()


def evaluate_generator(dataloader, G_R2C, device):
    G_R2C.eval()
    mae_list, rmse_list, psnr_list, sam_list = [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            rgb = batch["rgb"].to(device)
            cube = batch["cube"].to(device)

            pred_cube = G_R2C(rgb)
            mae_list.append(mae_metric(pred_cube, cube))
            rmse_list.append(rmse_metric(pred_cube, cube))
            psnr_list.append(psnr_metric(pred_cube, cube))
            sam_list.append(sam_metric(pred_cube, cube))

    G_R2C.train()
    return {
        "MAE": float(np.mean(mae_list)) if mae_list else None,
        "RMSE": float(np.mean(rmse_list)) if rmse_list else None,
        "PSNR": float(np.mean(psnr_list)) if psnr_list else None,
        "SAM": float(np.mean(sam_list)) if sam_list else None,
    }


# -----------------------------
# Training Loop
# -----------------------------
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # Create dated subdirectory inside out_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, timestamp)
    ensure_dir(run_dir)
    print(f"Run directory: {run_dir}")

    log_dir = os.path.join(run_dir, "logs")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    vis_dir = os.path.join(run_dir, "visualizations")
    ensure_dir(log_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(vis_dir)

    # Data
    (
        train_loader,
        val_loader,
        test_loader,
        patient_splits,
        patient_to_rois
    ) = build_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Save split info for reproducibility
    split_info = {
        "patient_splits": patient_splits,
        "patient_to_rois": dict(patient_to_rois),
    }
    with open(os.path.join(run_dir, "splits.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    # Models
    G_R2C = ResnetGenerator(
        input_nc=3,
        output_nc=args.num_bands,
        ngf=args.ngf,
        n_blocks=args.n_res_blocks,
    )
    G_C2R = ResnetGenerator(
        input_nc=args.num_bands,
        output_nc=3,
        ngf=args.ngf,
        n_blocks=args.n_res_blocks,
    )
    D_C = NLayerDiscriminator(input_nc=args.num_bands, ndf=args.ndf)
    D_R = NLayerDiscriminator(input_nc=3, ndf=args.ndf)

    G_R2C.apply(weights_init_normal)
    G_C2R.apply(weights_init_normal)
    D_C.apply(weights_init_normal)
    D_R.apply(weights_init_normal)

    G_R2C.to(device)
    G_C2R.to(device)
    D_C.to(device)
    D_R.to(device)

    # Losses
    criterion_GAN = nn.MSELoss()   # LSGAN
    criterion_cycle = nn.L1Loss()
    criterion_sup = nn.L1Loss()    # supervised RGB->cube loss

    # Optimizers
    optimizer_G = torch.optim.Adam(
        list(G_R2C.parameters()) + list(G_C2R.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )
    optimizer_D_C = torch.optim.Adam(D_C.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_R = torch.optim.Adam(D_R.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Schedulers (optional)
    def lambda_rule(epoch):
        return 1.0 - max(0, epoch + 1 - args.lr_decay_start) / float(args.epochs - args.lr_decay_start + 1)

    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D_C = torch.optim.lr_scheduler.LambdaLR(optimizer_D_C, lr_lambda=lambda_rule)
    scheduler_D_R = torch.optim.lr_scheduler.LambdaLR(optimizer_D_R, lr_lambda=lambda_rule)

    # Training history
    history = {
        "epoch": [],
        "G_loss": [],
        "D_C_loss": [],
        "D_R_loss": [],
        "train_MAE": [],
        "val_MAE": [],
        "val_RMSE": [],
        "val_PSNR": [],
        "val_SAM": [],
    }
    best_val_mae = float("inf")

    for epoch in range(args.epochs):
        G_R2C.train()
        G_C2R.train()
        D_C.train()
        D_R.train()

        epoch_G_loss = 0.0
        epoch_D_C_loss = 0.0
        epoch_D_R_loss = 0.0
        epoch_train_mae = 0.0
        n_batches = 0

        for batch in train_loader:
            rgb = batch["rgb"].to(device)   # (B,3,H,W)
            cube = batch["cube"].to(device) # (B,C,H,W)

            # --------------------
            #  Train Generators
            # --------------------
            optimizer_G.zero_grad()

            # RGB -> Cube
            fake_cube = G_R2C(rgb)
            pred_fake_cube = D_C(fake_cube)
            valid_c = torch.ones_like(pred_fake_cube)
            loss_G_R2C = criterion_GAN(pred_fake_cube, valid_c)

            # Cube -> RGB
            fake_rgb = G_C2R(cube)
            pred_fake_rgb = D_R(fake_rgb)
            valid_r = torch.ones_like(pred_fake_rgb)
            loss_G_C2R = criterion_GAN(pred_fake_rgb, valid_r)

            # Cycle losses
            rec_rgb = G_C2R(fake_cube)
            rec_cube = G_R2C(fake_rgb)

            loss_cycle_rgb = criterion_cycle(rec_rgb, rgb)
            loss_cycle_cube = criterion_cycle(rec_cube, cube)
            loss_cycle_total = (loss_cycle_rgb + loss_cycle_cube) * args.lambda_cyc

            # Supervised L1 loss for RGB->Cube
            loss_sup = criterion_sup(fake_cube, cube) * args.lambda_sup

            # Total generator loss
            loss_G = loss_G_R2C + loss_G_C2R + loss_cycle_total + loss_sup
            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator C
            # -----------------------
            optimizer_D_C.zero_grad()

            pred_real_cube = D_C(cube)
            valid_c = torch.ones_like(pred_real_cube)
            loss_D_C_real = criterion_GAN(pred_real_cube, valid_c)

            pred_fake_cube = D_C(fake_cube.detach())
            fake_c = torch.zeros_like(pred_fake_cube)
            loss_D_C_fake = criterion_GAN(pred_fake_cube, fake_c)

            loss_D_C = 0.5 * (loss_D_C_real + loss_D_C_fake)
            loss_D_C.backward()
            optimizer_D_C.step()

            # -----------------------
            #  Train Discriminator R
            # -----------------------
            optimizer_D_R.zero_grad()

            pred_real_rgb = D_R(rgb)
            valid_r = torch.ones_like(pred_real_rgb)
            loss_D_R_real = criterion_GAN(pred_real_rgb, valid_r)

            pred_fake_rgb = D_R(fake_rgb.detach())
            fake_r = torch.zeros_like(pred_fake_rgb)
            loss_D_R_fake = criterion_GAN(pred_fake_rgb, fake_r)

            loss_D_R = 0.5 * (loss_D_R_real + loss_D_R_fake)
            loss_D_R.backward()
            optimizer_D_R.step()

            # Accumulate
            epoch_G_loss += loss_G.item()
            epoch_D_C_loss += loss_D_C.item()
            epoch_D_R_loss += loss_D_R.item()
            epoch_train_mae += mae_metric(fake_cube.detach(), cube)
            n_batches += 1

        scheduler_G.step()
        scheduler_D_C.step()
        scheduler_D_R.step()

        if n_batches > 0:
            epoch_G_loss /= n_batches
            epoch_D_C_loss /= n_batches
            epoch_D_R_loss /= n_batches
            epoch_train_mae /= n_batches

        # Validation metrics on RGB->Cube
        val_metrics = evaluate_generator(val_loader, G_R2C, device)
        val_mae = val_metrics["MAE"]
        val_rmse = val_metrics["RMSE"]
        val_psnr = val_metrics["PSNR"]
        val_sam = val_metrics["SAM"]

        history["epoch"].append(epoch)
        history["G_loss"].append(epoch_G_loss)
        history["D_C_loss"].append(epoch_D_C_loss)
        history["D_R_loss"].append(epoch_D_R_loss)
        history["train_MAE"].append(epoch_train_mae)
        history["val_MAE"].append(val_mae)
        history["val_RMSE"].append(val_rmse)
        history["val_PSNR"].append(val_psnr)
        history["val_SAM"].append(val_sam)

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"G_loss={epoch_G_loss:.4f}, D_C={epoch_D_C_loss:.4f}, D_R={epoch_D_R_loss:.4f}, "
            f"train_MAE={epoch_train_mae:.4f}, val_MAE={val_mae:.4f}, "
            f"val_RMSE={val_rmse:.4f}, val_PSNR={val_psnr:.2f}, val_SAM={val_sam:.4f}"
        )

        # Save history every epoch
        with open(os.path.join(log_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        # Save best model based on val MAE
        if val_mae is not None and val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(
                {
                    "epoch": epoch,
                    "G_R2C": G_R2C.state_dict(),
                    "G_C2R": G_C2R.state_dict(),
                    "D_C": D_C.state_dict(),
                    "D_R": D_R.state_dict(),
                    "optimizer_G": optimizer_G.state_dict(),
                    "optimizer_D_C": optimizer_D_C.state_dict(),
                    "optimizer_D_R": optimizer_D_R.state_dict(),
                    "best_val_mae": best_val_mae,
                },
                os.path.join(ckpt_dir, "best_model.pt"),
            )

    # Save final model
    torch.save(
        {
            "epoch": args.epochs - 1,
            "G_R2C": G_R2C.state_dict(),
            "G_C2R": G_C2R.state_dict(),
            "D_C": D_C.state_dict(),
            "D_R": D_R.state_dict(),
        },
        os.path.join(ckpt_dir, "last_model.pt"),
    )

    # Plot curves similar to UNet training curves
    plot_training_curves(history, log_dir)

    # Evaluate on test set and visualize bands
    test_metrics = evaluate_generator(test_loader, G_R2C, device)
    with open(os.path.join(log_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    print("Test metrics:", test_metrics)

    visualize_bands(
        test_loader,
        G_R2C,
        vis_dir,
        bands_to_visualize=args.bands_to_visualize,
        max_samples=args.max_vis_samples,
    )


# -----------------------------
# Plotting
# -----------------------------
def plot_training_curves(history, log_dir):
    epochs = history["epoch"]

    # Generator & Discriminator losses
    plt.figure()
    plt.plot(epochs, history["G_loss"], label="G_loss")
    plt.plot(epochs, history["D_C_loss"], label="D_C_loss")
    plt.plot(epochs, history["D_R_loss"], label="D_R_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CycleGAN Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "loss_curves.png"))
    plt.close()

    # Train / Val MAE
    plt.figure()
    plt.plot(epochs, history["train_MAE"], label="train_MAE")
    plt.plot(epochs, history["val_MAE"], label="val_MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("MAE (RGB->Cube)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "mae_curves.png"))
    plt.close()

    # Val PSNR
    plt.figure()
    plt.plot(epochs, history["val_PSNR"], label="val_PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.title("Validation PSNR (RGB->Cube)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "psnr_curves.png"))
    plt.close()

    # Val SAM
    plt.figure()
    plt.plot(epochs, history["val_SAM"], label="val_SAM")
    plt.xlabel("Epoch")
    plt.ylabel("SAM (radians)")
    plt.title("Validation Spectral Angle Mapper (RGB->Cube)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "sam_curves.png"))
    plt.close()


# -----------------------------
# Band Visualization
# -----------------------------
def visualize_bands(
    test_loader,
    G_R2C,
    out_dir,
    bands_to_visualize,
    max_samples=5,
):
    """
    For the first `max_samples` test samples, visualize:
        - original RGB
        - ground-truth band (for selected bands)
        - predicted band
        - absolute difference
    Save figures in out_dir.
    """
    ensure_dir(out_dir)
    device = next(G_R2C.parameters()).device
    G_R2C.eval()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= max_samples:
                break
            rgb = batch["rgb"].to(device)          # (1,3,H,W)
            cube = batch["cube"].to(device)        # (1,C,H,W)
            wl = batch["wavelengths"][0].cpu().numpy()  # (C,)
            patient_id = batch["patient_id"][0]
            roi_id = batch["roi_id"][0]

            pred_cube = G_R2C(rgb)  # (1,C,H,W)

            rgb_np = rgb[0].cpu().numpy()       # 3,H,W
            cube_np = cube[0].cpu().numpy()     # C,H,W
            pred_np = pred_cube[0].cpu().numpy()

            # Convert RGB back to H,W,3 for plotting
            rgb_img = np.transpose(rgb_np, (1, 2, 0))
            rgb_img = np.clip(rgb_img, 0.0, 1.0)

            for b in bands_to_visualize:
                if b < 0 or b >= cube_np.shape[0]:
                    continue

                lam = wl[b] if len(wl) > b else None
                gt_band = cube_np[b]
                pr_band = pred_np[b]
                diff_band = np.abs(gt_band - pr_band)

                # Normalize bands for visualization
                def norm_band(x):
                    x = x - x.min()
                    maxv = x.max()
                    if maxv > 1e-8:
                        x = x / maxv
                    return x

                gt_vis = norm_band(gt_band)
                pr_vis = norm_band(pr_band)
                diff_vis = norm_band(diff_band)

                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(rgb_img)
                axes[0].set_title("Original RGB")
                axes[0].axis("off")

                axes[1].imshow(gt_vis, cmap="gray")
                title_gt = f"GT Band {b}"
                if lam is not None:
                    title_gt += f" ({lam:.1f} nm)"
                axes[1].set_title(title_gt)
                axes[1].axis("off")

                axes[2].imshow(pr_vis, cmap="gray")
                axes[2].set_title(f"Pred Band {b}")
                axes[2].axis("off")

                axes[3].imshow(diff_vis, cmap="gray")
                axes[3].set_title(f"|GT-Pred| Band {b}")
                axes[3].axis("off")

                fig.suptitle(f"Patient {patient_id}, ROI {roi_id}", fontsize=12)
                plt.tight_layout()
                out_path = os.path.join(
                    out_dir,
                    f"{patient_id}_{roi_id}_sample{i}_band{b}.png",
                )
                plt.savefig(out_path)
                plt.close()


# -----------------------------
# Argparse and main
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="CycleGAN RGB->Cube (32 bands) training for band-reduced hyperspectral data."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/scratch/users/atesfet/GHSI/band-reduced-cubes",
        help="Root directory of band-reduced cubes (patient folders).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./cyclegan_rgb2cube_runs",
        help="Base output directory; a dated subdirectory will be created inside it.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU, ignore CUDA even if available.")

    # Model hyperparameters
    parser.add_argument("--num-bands", type=int, default=32, help="Number of bands in cube.")
    parser.add_argument("--ngf", type=int, default=64, help="Generator base channels.")
    parser.add_argument("--ndf", type=int, default=64, help="Discriminator base channels.")
    parser.add_argument("--n-res-blocks", type=int, default=9, help="Number of ResNet blocks in generators.")

    # Loss weights
    parser.add_argument("--lambda-cyc", type=float, default=10.0, help="Weight for cycle-consistency loss.")
    parser.add_argument("--lambda-sup", type=float, default=10.0, help="Weight for supervised L1 on RGB->Cube.")

    # Optim
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--lr-decay-start",
        type=int,
        default=100,
        help="Epoch to start linearly decaying learning rate.",
    )

    # Visualization
    parser.add_argument(
        "--bands-to-visualize",
        type=int,
        nargs="+",
        default=[0, 10, 20, 31],
        help="Band indices to visualize on test set.",
    )
    parser.add_argument(
        "--max-vis-samples",
        type=int,
        default=5,
        help="Number of test samples to visualize.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)