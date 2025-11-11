#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RGB -> Full Hyperspectral Cube (PyTorch) — robust loader for cube.npy shapes

Per-ROI required:
  - original_rgb.png
  - cube.npy            (may be HxWxK, KxHxW, or flat; this script reshapes/permutes robustly)
  - wavelengths.npy     (K,)
  - cube_metadata.npy   (optional; used if needed to infer shape)

Splits at ROI level; metrics: L1 (per-pixel), MSE, PSNR(dB), SAM(°)
Test: selected bands (embedded list) → side-by-side PNGs + SSIM (per band + mean)
All metrics saved to JSON + CSV; curves plotted (L1, PSNR, SAM)

Results root: /scratch/users/atesfet/GHSI/results-cube/<timestamp>/
"""

import os, sys, math, json, time, csv, random, argparse, contextlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------- Selected (reduced) bands for test-time comparison --------
RELEVANT_BANDS = {
    'nuclear_403nm': 403.0,
    'nuclear_448nm': 448.0,
    'nuclear_470nm': 470.0,
    'eosin_peak_525nm': 525.0,
    'cytoplasm_550nm': 550.0,
    'hematoxylin_peak_596nm': 596.0,
    'vascular_655nm': 655.0,
    'hemoglobin_660nm': 660.0,
    'nir_700nm': 700.0,
    'nir_780nm': 780.0,
    'nir_850nm': 850.0,
    'nir_894nm': 894.0,
    'water_970nm': 970.0,
}

try:
    from skimage.metrics import structural_similarity as ssim_fn
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


# =========================
# Utilities
# =========================

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def timestamp(): return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def list_roi_dirs(data_root: Path) -> List[Path]:
    out = []
    for pd in sorted(data_root.glob("P*")):
        if not pd.is_dir(): continue
        out += [d for d in sorted(pd.glob("ROI_*")) if d.is_dir()]
    return out

def save_split_csv(paths: List[Path], out_csv: Path):
    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id","roi_id","roi_path"])
        for d in paths: w.writerow([d.parent.name, d.name, str(d)])

def split_rois(roi_dirs: List[Path], seed=42, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    assert abs(train_ratio + val_ratio + test_ratio - 1) < 1e-6
    rng = random.Random(seed); rois = roi_dirs[:]; rng.shuffle(rois)
    n = len(rois); nt = int(round(n*train_ratio)); nv = int(round(n*val_ratio))
    return rois[:nt], rois[nt:nt+nv], rois[nt+nv:]

def save_list_as_json_csv(rows: List[Dict], json_path: Path, csv_path: Path):
    ensure_dir(json_path.parent)
    with open(json_path, "w") as f: json.dump(rows, f, indent=2)
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows: w.writerow({k: r.get(k, None) for k in keys})

def save_dict_as_json_csv(row: Dict, json_path: Path, csv_path: Path):
    ensure_dir(json_path.parent)
    with open(json_path, "w") as f: json.dump(row, f, indent=2)
    keys = sorted(row.keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerow(row)


# =========================
# IO helpers 
# =========================

def load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0

def _safe_load_npy(path: Path):
    return np.load(path, allow_pickle=True)

def load_wavelengths(path: Path) -> np.ndarray:
    wl = _safe_load_npy(path).astype(np.float32)
    wl = wl.reshape(-1)
    return wl

def _reshape_cube_flat(flat: np.ndarray, H: int, W: int, K: int) -> np.ndarray:
    if flat.size == H*W*K:
        return flat.reshape((H, W, K))
    # Try common alternative memory orders if needed
    if flat.size == K*H*W:
        return flat.reshape((K, H, W)).transpose(1, 2, 0)  # -> H,W,K
    raise ValueError(f"Cannot reshape flat cube of size {flat.size} into ({H},{W},{K})")

def _move_K_to_last(arr: np.ndarray, K: int) -> np.ndarray:
    # Ensure the last axis equals K (H,W,K). If already OK, return as-is.
    if arr.ndim != 3:
        raise ValueError(f"Cube must be 3D after reshape, got shape {arr.shape}")
    if arr.shape[-1] == K:
        return arr
    # Find which axis equals K
    axes = [i for i, d in enumerate(arr.shape) if d == K]
    if not axes:
        raise ValueError(f"No axis of size K={K} found in shape {arr.shape}")
    ax = axes[0]
    if ax == 0:
        return np.moveaxis(arr, 0, 2)  # K,H,W -> H,W,K
    if ax == 1:
        return np.moveaxis(arr, 1, 2)  # H,K,W -> H,W,K
    return arr  # already last

def load_cube_robust(roi_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load cube.npy robustly and return (cube[H,W,K], wavelengths[K]).
    Handles flat arrays and alternative axis orders; uses original_rgb.png, wavelengths.npy,
    and optionally cube_metadata.npy['cube_shape'] to infer dimensions.
    """
    cube_path = roi_dir / "cube.npy"
    wl_path   = roi_dir / "wavelengths.npy"
    rgb_path  = roi_dir / "original_rgb.png"
    meta_path = roi_dir / "cube_metadata.npy"

    if not cube_path.exists() or not wl_path.exists() or not rgb_path.exists():
        raise FileNotFoundError(f"Missing required files under {roi_dir}")

    wl = load_wavelengths(wl_path)  # (K,)
    K  = int(wl.shape[0])

    # Get H, W from the RGB image
    rgb = Image.open(rgb_path)
    W, H = rgb.size  # PIL: size=(width,height)

    arr = _safe_load_npy(cube_path)
    # If already 3D and last axis matches K, great
    if arr.ndim == 3 and arr.shape[-1] == K:
        cube = arr
    else:
        # Try to use metadata if present
        meta_shape = None
        if meta_path.exists():
            try:
                meta = _safe_load_npy(meta_path)
                if isinstance(meta, dict) or hasattr(meta, "item"):
                    md = meta if isinstance(meta, dict) else meta.item()
                    if isinstance(md, dict):
                        # Support any of these common keys
                        meta_shape = tuple(md.get("cube_shape") or md.get("shape") or ())
            except Exception:
                meta_shape = None

        if arr.ndim == 1:
            # Flat array: reshape using (H, W, K) if possible; else try metadata
            try:
                cube = _reshape_cube_flat(arr, H, W, K)
            except ValueError:
                if meta_shape and np.prod(meta_shape) == arr.size:
                    cube = arr.reshape(meta_shape)
                    cube = _move_K_to_last(cube, K)
                else:
                    raise
        elif arr.ndim == 3:
            # 3D but wrong axis order -> move K axis to the end
            cube = _move_K_to_last(arr, K)
        else:
            # Unknown case: try metadata
            if meta_shape and np.prod(meta_shape) == arr.size:
                cube = arr.reshape(meta_shape)
                cube = _move_K_to_last(cube, K)
            else:
                raise ValueError(f"Unsupported cube shape {arr.shape} for ROI {roi_dir}")

    cube = cube.astype(np.float32)
    cube = np.clip(cube, 0.0, 1.0)
    # Final sanity check
    if cube.shape != (H, W, K):
        raise ValueError(f"Final cube shape mismatch: got {cube.shape}, expected ({H},{W},{K}) in {roi_dir}")
    return cube, wl

def to_tensor(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 2: arr = arr[..., None]
    return torch.from_numpy(arr.transpose(2,0,1))  # HxWxC -> CxHxW

def from_tensor(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().transpose(1,2,0)  # CxHxW -> HxWxC

def save_gray_png(arr2d: np.ndarray, out_path: Path):
    arr = np.clip(arr2d, 0, 1)
    Image.fromarray((arr*255).round().astype(np.uint8), mode='L').save(out_path)

def save_side_by_side(gt: np.ndarray, pred: np.ndarray, titles: Tuple[str, str], out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(np.clip(gt,0,1), cmap='gray', vmin=0, vmax=1); axes[0].set_title(titles[0]); axes[0].axis('off')
    axes[1].imshow(np.clip(pred,0,1), cmap='gray', vmin=0, vmax=1); axes[1].set_title(titles[1]); axes[1].axis('off')
    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)

def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    if _HAS_SKIMAGE:
        return float(ssim_fn(a, b, data_range=1.0, gaussian_weights=True, use_sample_covariance=False))
    # minimal fallback
    C1, C2 = 0.01**2, 0.03**2
    x,y = a.astype(np.float64), b.astype(np.float64)
    mu_x, mu_y = x.mean(), y.mean()
    sx2, sy2 = ((x-mu_x)**2).mean(), ((y-mu_y)**2).mean()
    sxy = ((x-mu_x)*(y-mu_y)).mean()
    num = (2*mu_x*mu_y + C1)*(2*sxy + C2)
    den = (mu_x**2 + mu_y**2 + C1)*(sx2 + sy2 + C2) + 1e-12
    return float(num/den)


# =========================
# Dataset
# =========================

class RGB2CubeDataset(Dataset):
    def __init__(self, roi_dirs: List[Path]):
        assert roi_dirs, "Empty split!"
        self.roi_dirs = roi_dirs
        # Infer K robustly from the first ROI
        cube0, wl0 = load_cube_robust(roi_dirs[0])
        self.K = int(wl0.shape[0])
        # Validate required files exist
        for d in roi_dirs:
            for fn in ["original_rgb.png", "cube.npy", "wavelengths.npy", "cube_metadata.npy"]:
                if not (d / fn).exists():
                    raise FileNotFoundError(f"Missing {fn} in {d}")
        # Also verify shapes are consistent for a few samples (fast check)

    def __len__(self): return len(self.roi_dirs)

    def __getitem__(self, i):
        d = self.roi_dirs[i]
        cube, wl = load_cube_robust(d)
        rgb = load_rgb(d / "original_rgb.png")
        return {
            "rgb": to_tensor(rgb),               # 3xHxW
            "cube": to_tensor(cube),             # KxHxW
            "wavelengths": wl,                   # numpy (K,)
            "roi_path": str(d),
            "patient_id": d.parent.name,
            "roi_id": d.name
        }


# =========================
# Model (bilinear U-Net, odd-size safe)
# =========================

def _center_crop_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    _,_,h,w = src.shape; _,_,H,W = ref.shape
    dh, dw = h-H, w-W
    if dh==0 and dw==0: return src
    top = max(dh//2, 0); left = max(dw//2, 0)
    return src[:, :, top:top+H, left:left+W]

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNetBilinear(nn.Module):
    def __init__(self, in_ch=3, out_ch=826, base=48):
        super().__init__()
        b = base
        self.inc   = DoubleConv(in_ch, b)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b, 2*b))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(2*b, 4*b))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(4*b, 8*b))
        self.bot   = DoubleConv(8*b, 16*b)
        self.red3 = nn.Conv2d(16*b, 8*b, 1)
        self.red2 = nn.Conv2d(8*b, 4*b, 1)
        self.red1 = nn.Conv2d(4*b, 2*b, 1)
        self.dec3 = DoubleConv(8*b + 4*b, 8*b)
        self.dec2 = DoubleConv(4*b + 2*b, 4*b)
        self.dec1 = DoubleConv(2*b + 1*b, 2*b)
        self.outc = nn.Conv2d(2*b, out_ch, 1)

    def _up(self, x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xb = self.bot(x4)

        u3 = self._up(xb, x3); u3 = self.red3(u3)
        x  = self.dec3(torch.cat([u3, _center_crop_like(x3,u3)], dim=1))
        u2 = self._up(x, x2);  u2 = self.red2(u2)
        x  = self.dec2(torch.cat([u2, _center_crop_like(x2,u2)], dim=1))
        u1 = self._up(x, x1);  u1 = self.red1(u1)
        x  = self.dec1(torch.cat([u1, _center_crop_like(x1,u1)], dim=1))
        x  = self.outc(x)
        return torch.sigmoid(x)


# =========================
# AMP helpers
# =========================

def make_grad_scaler(device: torch.device):
    if device.type != "cuda":
        class _Noop:
            def scale(self,x): return x
            def step(self,opt): opt.step()
            def update(self): pass
        return _Noop()
    try:
        return torch.amp.GradScaler(device_type="cuda")
    except TypeError:
        return torch.cuda.amp.GradScaler()

def autocast_cuda(enabled):
    if not enabled: return contextlib.nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)


# =========================
# Metrics (cube-wide)
# =========================

def l1_loss(pred, target): return F.l1_loss(pred, target)

@torch.no_grad()
def psnr_db_from_mse(mse: float, data_range: float = 1.0) -> float:
    if mse <= 0: return float('inf')
    return 10.0 * math.log10((data_range ** 2) / mse)

@torch.no_grad()
def spectral_angle_mapper_deg(pred, target, eps=1e-8):
    """Per-pixel spectral angle; average over all pixels."""
    B,K,H,W = pred.shape
    p = pred.permute(0,2,3,1).reshape(-1,K)
    t = target.permute(0,2,3,1).reshape(-1,K)
    num = (p*t).sum(1)
    p_n = torch.linalg.norm(p,2,1)+eps; t_n = torch.linalg.norm(t,2,1)+eps
    v = (p_n>eps)&(t_n>eps)
    if v.any():
        cos = torch.clamp(num[v]/(p_n[v]*t_n[v]), -1, 1)
        return torch.arccos(cos).mean().item()*180/math.pi
    return float("nan")


# =========================
# Train / Eval
# =========================

def train_one_epoch(model,loader,opt,scaler,device):
    model.train(); tot=0; use_amp=device.type=="cuda"
    for b in loader:
        rgb=b["rgb"].to(device, non_blocking=True)
        tgt=b["cube"].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast_cuda(use_amp):
            pred=model(rgb); loss=l1_loss(pred,tgt)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tot+=loss.item()*rgb.size(0)
    return tot/len(loader.dataset)

@torch.no_grad()
def evaluate(model,loader,device):
    """Return mean L1 (per-pixel), MSE (global mean), PSNR (dB), SAM (deg)."""
    model.eval(); tl1=0; npx=0; tmse=0; nsam=0; tsum=0; use_amp=device.type=="cuda"
    for b in loader:
        rgb=b["rgb"].to(device, non_blocking=True)
        tgt=b["cube"].to(device, non_blocking=True)
        with autocast_cuda(use_amp): pred=model(rgb)
        tl1 += F.l1_loss(pred, tgt, reduction="sum").item(); npx += tgt.numel()
        tmse += F.mse_loss(pred, tgt, reduction="mean").item()  # mean over this batch
        sam = spectral_angle_mapper_deg(pred, tgt)
        if not (sam!=sam): tsum += sam; nsam += 1
    mean_l1 = tl1/max(npx,1)
    mean_mse = tmse / max(len(loader),1)
    mean_psnr = psnr_db_from_mse(mean_mse)
    mean_sam = tsum/max(nsam,1) if nsam else float('nan')
    return mean_l1, mean_mse, mean_psnr, mean_sam


# =========================
# Selected-band utilities (nearest by nm)
# =========================

def nearest_indices_for_targets(wavelengths: np.ndarray, target_wls: List[float]) -> List[int]:
    idxs = []
    for tw in target_wls:
        idx = int(np.argmin(np.abs(wavelengths - float(tw))))
        idxs.append(idx)
    return idxs

def extract_selected_from_cube(cube: np.ndarray, wl: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    names = list(RELEVANT_BANDS.keys())
    targets = [RELEVANT_BANDS[k] for k in names]
    idxs = nearest_indices_for_targets(wl, targets)
    sel = cube[:, :, idxs]  # HxWxM
    return sel, names


# =========================
# Test-time visualization + per-ROI metrics (incl. selected-band SSIM)
# =========================

@torch.no_grad()
def test_and_visualize(model,loader,device,out_dir):
    ensure_dir(out_dir); per_roi=[]; use_amp=device.type=="cuda"; model.eval()
    for b in loader:
        rgb=b["rgb"].to(device); tgt=b["cube"].to(device)
        with autocast_cuda(use_amp): pred=model(rgb)
        B = rgb.size(0)
        for i in range(B):
            pid = b["patient_id"][i]; rid = b["roi_id"][i]; rpath=b["roi_path"][i]
            idir = out_dir / pid / rid; ensure_dir(idir)

            gt_cube = from_tensor(tgt[i])     # HxWxK
            pr_cube = from_tensor(pred[i])    # HxWxK
            np.save(idir/"gt_cube.npy", gt_cube); np.save(idir/"pred_cube.npy", pr_cube)

            wl = b["wavelengths"][i]
            np.save(idir/"wavelengths.npy", wl)

            # Cube metrics (per ROI)
            l1 = float(F.l1_loss(pred[i:i+1], tgt[i:i+1], reduction="mean").item())
            mse = float(F.mse_loss(pred[i:i+1], tgt[i:i+1], reduction="mean").item())
            psnr = psnr_db_from_mse(mse)
            sam = spectral_angle_mapper_deg(pred[i:i+1], tgt[i:i+1])

            # Selected bands from both cubes
            sel_gt, band_names = extract_selected_from_cube(gt_cube, wl)
            sel_pr, _          = extract_selected_from_cube(pr_cube, wl)
            sb_dir = idir / "selected_bands"; ensure_dir(sb_dir)

            ssim_per_band = {}
            for k, name in enumerate(band_names):
                gt2d = sel_gt[:,:,k]; pr2d = sel_pr[:,:,k]
                save_side_by_side(gt2d, pr2d, (f"{name} - GT", f"{name} - Pred"), sb_dir / f"{name}_panel.png")
                ssim_per_band[name] = _ssim(gt2d, pr2d)
                save_gray_png(gt2d, sb_dir / f"{name}_GT.png")
                save_gray_png(pr2d, sb_dir / f"{name}_Pred.png")
            ssim_mean = float(np.mean(list(ssim_per_band.values()))) if ssim_per_band else float('nan')

            per_roi.append({
                "patient_id": pid,
                "roi_id": rid,
                "roi_path": rpath,
                "mean_L1": l1,
                "MSE": mse,
                "PSNR_dB": psnr,
                "SAM_deg": sam,
                "SSIM_selected_mean": ssim_mean,
                **{f"SSIM_{k}": float(v) for k,v in ssim_per_band.items()}
            })
    return per_roi


# =========================
# Main
# =========================

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data_root",type=Path,required=True,
                   help="e.g. /scratch/users/atesfet/GHSI/PKG-HistologyHSI-GB/reconstructed_cubes")
    p.add_argument("--results_root",type=Path,required=True,
                   help="e.g. /scratch/users/atesfet/GHSI/results-cube")
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--batch_size",type=int,default=1)  # full cube is heavy
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--num_workers",type=int,default=4)
    p.add_argument("--train_ratio",type=float,default=0.7)
    p.add_argument("--val_ratio",type=float,default=0.2)
    p.add_argument("--test_ratio",type=float,default=0.1)
    p.add_argument("--base_channels",type=int,default=48, help="try 32/48/64 depending on VRAM")
    args=p.parse_args()

    set_seed(args.seed)

    # Run directories
    run_id = timestamp()
    run_dir = args.results_root / run_id
    ckpt_dir = run_dir / "checkpoints"
    vis_dir = run_dir / "test_visuals"
    csv_dir = run_dir / "splits"
    logs_dir = run_dir / "logs"
    for d in [run_dir, ckpt_dir, vis_dir, csv_dir, logs_dir]: ensure_dir(d)

    # Discover & split
    roi_dirs = list_roi_dirs(args.data_root)
    assert roi_dirs, f"No ROI dirs under {args.data_root}"
    train_rois, val_rois, test_rois = split_rois(
        roi_dirs, seed=args.seed,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio
    )
    save_split_csv(train_rois, csv_dir/"train.csv")
    save_split_csv(val_rois,   csv_dir/"val.csv")
    save_split_csv(test_rois,  csv_dir/"test.csv")
    print(f"Split: train={len(train_rois)} val={len(val_rois)} test={len(test_rois)}")
    print(f"CSV splits saved under {csv_dir}")

    # Datasets / loaders
    train_ds = RGB2CubeDataset(train_rois)
    val_ds   = RGB2CubeDataset(val_rois)
    test_ds  = RGB2CubeDataset(test_rois)

    K = train_ds.K

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Device + AMP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    try:
        scaler = torch.amp.GradScaler(device_type="cuda")
    except Exception:
        scaler = torch.cuda.amp.GradScaler() if device.type=="cuda" else type("Noop",(object,),{"scale":lambda s,x:x,"step":lambda s,opt:opt.step(),"update":lambda s:None})()

    # Model / Opt
    model = UNetBilinear(in_ch=3, out_ch=K, base=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Train loop
    best_val = float("inf")
    history: List[Dict] = []
    t0 = time.time()
    for epoch in range(1, args.epochs+1):
        t_ep0 = time.time()
        model.train(); use_amp = device.type=="cuda"
        # train
        train_l1 = train_one_epoch(model, train_loader, optimizer, scaler, device)
        # val
        val_l1, val_mse, val_psnr, val_sam = evaluate(model, val_loader, device)
        dt = time.time() - t_ep0

        row = {
            "epoch": epoch,
            "train_L1": train_l1,
            "val_L1": val_l1,
            "val_MSE": val_mse,
            "val_PSNR_dB": val_psnr,
            "val_SAM_deg": val_sam,
            "epoch_seconds": dt
        }
        history.append(row)

        print(f"Epoch {epoch:03d} | {dt:.1f}s | "
              f"train L1 {train_l1:.4f} | val L1 {val_l1:.4f} | "
              f"val MSE {val_mse:.6f} | val PSNR {val_psnr:.2f} dB | val SAM {val_sam:.2f}°")

        # Save last & best checkpoints
        torch.save({"epoch":epoch,"model":model.state_dict(),
                    "optimizer":optimizer.state_dict(),"out_channels":K},
                   ckpt_dir/"last.pt")
        if val_l1 < best_val:
            best_val = val_l1
            torch.save({"epoch":epoch,"model":model.state_dict(),
                        "optimizer":optimizer.state_dict(),"out_channels":K},
                       ckpt_dir/"best.pt")

        # Persist history each epoch
        save_list_as_json_csv(history, logs_dir/"history.json", logs_dir/"history.csv")

    total_time = time.time()-t0
    print(f"Training complete in {total_time/60:.1f} min. Best val L1={best_val:.6f}")

    # Test with best checkpoint
    ckpt = torch.load(ckpt_dir/"best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_l1, test_mse, test_psnr, test_sam = evaluate(model, test_loader, device)
    print(f"Test: L1 {test_l1:.6f} | MSE {test_mse:.6f} | PSNR {test_psnr:.2f} dB | SAM {test_sam:.2f}°")

    test_summary = {
        "test_L1": test_l1,
        "test_MSE": test_mse,
        "test_PSNR_dB": test_psnr,
        "test_SAM_deg": test_sam,
        "best_epoch": ckpt.get("epoch", None),
        "num_train": len(train_rois),
        "num_val": len(val_rois),
        "num_test": len(test_rois),
        "run_dir": str(run_dir),
        "out_channels": K,
        "selected_band_names": list(RELEVANT_BANDS.keys())
    }
    save_dict_as_json_csv(test_summary, run_dir/"test_summary.json", run_dir/"test_summary.csv")

    # Test predictions & per-ROI metrics (includes selected-band SSIM & panels)
    print("Generating predictions + selected-band panels & per-ROI metrics on test set...")
    per_roi = test_and_visualize(model, test_loader, device, vis_dir)
    save_list_as_json_csv(per_roi, vis_dir/"test_metrics.json", vis_dir/"test_metrics.csv")

    # Plots
    try:
        epochs   = [h["epoch"] for h in history]
        train_L1 = [h["train_L1"] for h in history]
        val_L1   = [h["val_L1"] for h in history]
        val_SAM  = [h["val_SAM_deg"] for h in history]
        val_PSNR = [h["val_PSNR_dB"] for h in history]

        plt.figure(figsize=(6,4))
        plt.plot(epochs, train_L1, '-o', label="Train L1")
        plt.plot(epochs, val_L1,   '-o', label="Val L1")
        plt.xlabel("Epoch"); plt.ylabel("L1 Loss")
        plt.title("Training vs Validation L1"); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(run_dir/"L1_curves.png", dpi=150); plt.close()

        plt.figure(figsize=(6,4))
        plt.plot(epochs, val_SAM, '-o')
        plt.xlabel("Epoch"); plt.ylabel("SAM (deg)")
        plt.title("Validation SAM"); plt.grid(True)
        plt.tight_layout(); plt.savefig(run_dir/"SAM_curve.png", dpi=150); plt.close()

        plt.figure(figsize=(6,4))
        plt.plot(epochs, val_PSNR, '-o')
        plt.xlabel("Epoch"); plt.ylabel("PSNR (dB)")
        plt.title("Validation PSNR"); plt.grid(True)
        plt.tight_layout(); plt.savefig(run_dir/"PSNR_curve.png", dpi=150); plt.close()
        print(f"✓ Metric plots saved in {run_dir}")
    except Exception as e:
        print(f"(plotting skipped: {e})")

    print(f"All results saved under: {run_dir}")


if __name__ == "__main__":
    main()