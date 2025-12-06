#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global PCA exploration over many hyperspectral cubes.

- Discovers cubes under a root directory.
- Randomly samples spectra from each cube.
- Runs Incremental PCA over all sampled spectra.
- Computes band-importance scores from PCA loadings.
- Supports mapping from an original wavelength array down to the current
  cube bands via a band index map.

Assumptions (revised):
- Cubes are stored as .npy files with shape EXACTLY (H, W, B),
  i.e. the **last dimension is the spectral / wavelength dimension**.
- All cubes share the same number/order of spectral bands.
- wavelengths.npy has shape (B,) giving wavelength per band.
"""

import os
import glob
import argparse
import json
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm


# -------------------------
# Utility: I/O & discovery
# -------------------------

def find_cube_paths(root_dir: str, pattern: str) -> List[str]:
    """
    Find all cube .npy files under root_dir matching a glob pattern.

    pattern is interpreted relative to root_dir, e.g.:
      "P*/ROI_*/cube.npy"
    """
    search_pattern = os.path.join(root_dir, pattern)
    paths = sorted(glob.glob(search_pattern, recursive=True))
    return paths


def maybe_load_wavelengths(wavelengths_path: Optional[str]) -> Optional[np.ndarray]:
    if wavelengths_path is None:
        return None
    if not os.path.isfile(wavelengths_path):
        print(f"[WARN] wavelengths file not found at {wavelengths_path}; ignoring.")
        return None
    arr = np.load(wavelengths_path)
    if arr.ndim != 1:
        raise ValueError(f"wavelengths.npy should be 1D, got shape {arr.shape}")
    return arr


def maybe_load_band_map(band_map_path: Optional[str]) -> Optional[np.ndarray]:
    """
    Load an optional band index map.

    band_map is expected to be a 1D int array of length n_bands_cube,
    where band_map[i] is the index into the *full* wavelength array
    corresponding to cube band i.
    """
    if band_map_path is None:
        return None
    if not os.path.isfile(band_map_path):
        print(f"[WARN] band map file not found at {band_map_path}; ignoring.")
        return None
    arr = np.load(band_map_path)
    if arr.ndim != 1:
        raise ValueError(f"band_map.npy should be 1D, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"band_map.npy should be integer dtype, got {arr.dtype}")
    return arr.astype(np.int64)


# -------------------------
# Sampling from cubes
# -------------------------

def normalize_cube_spectrally(cube: np.ndarray) -> np.ndarray:
    """
    Simple per-band mean-centering.

    We assume cube has shape (H, W, B) and subtract the per-band mean
    across spatial pixels.
    """
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube (H, W, B), got shape {cube.shape}")

    H, W, B = cube.shape
    flat = cube.reshape(-1, B)  # (Npix, B)

    band_mean = flat.mean(axis=0, keepdims=True)
    flat_norm = flat - band_mean

    cube_norm = flat_norm.reshape(H, W, B)
    return cube_norm


def cube_to_spectra(
    cube: np.ndarray,
    n_samples: int,
    random_state: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    Convert a cube to a matrix of sampled spectra of shape (n_samples, B).

    Assumes cube shape is (H, W, B).
    """
    if random_state is None:
        random_state = np.random.RandomState()

    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube (H, W, B), got shape {cube.shape}")

    H, W, B = cube.shape
    spectra = cube.reshape(-1, B)  # (Npix, B)

    Npix = spectra.shape[0]
    if Npix == 0:
        raise ValueError("Cube appears to have zero pixels after reshaping.")

    n = min(n_samples, Npix)
    idx = random_state.choice(Npix, size=n, replace=False)
    return spectra[idx, :]


# -------------------------
# PCA + band importance
# -------------------------

def run_incremental_pca(
    cube_paths: List[str],
    n_components: int,
    n_samples_per_cube: int,
    batch_size: int,
    seed: int = 0,
    normalize: bool = True
) -> Tuple[IncrementalPCA, int]:
    """
    Runs IncrementalPCA on sampled spectra from all cubes.

    Returns:
      ipca: fitted IncrementalPCA object
      n_bands: number of spectral bands (B)
    """
    ipca = IncrementalPCA(n_components=n_components)
    rng = np.random.RandomState(seed)
    n_bands = None

    print(f"[INFO] Fitting IncrementalPCA on {len(cube_paths)} cubes...")
    batch = []

    for cube_path in tqdm(cube_paths, desc="Cubes (first pass)"):
        cube = np.load(cube_path)

        # Enforce (H, W, B) assumption
        if cube.ndim != 3:
            raise ValueError(
                f"Cube at {cube_path} is not 3D (H, W, B); got shape {cube.shape}"
            )

        if normalize:
            cube = normalize_cube_spectrally(cube)

        spectra = cube_to_spectra(cube, n_samples_per_cube, rng)
        if n_bands is None:
            n_bands = spectra.shape[1]
        elif spectra.shape[1] != n_bands:
            raise ValueError(
                f"Inconsistent band count: {cube_path} has {spectra.shape[1]} bands, "
                f"expected {n_bands}"
            )

        batch.append(spectra)
        total_in_batch = sum(b.shape[0] for b in batch)
        if total_in_batch >= batch_size:
            X_batch = np.vstack(batch)
            ipca.partial_fit(X_batch)
            batch = []

    if batch:
        X_batch = np.vstack(batch)
        ipca.partial_fit(X_batch)

    print("[INFO] IncrementalPCA fitting completed.")
    return ipca, n_bands


def compute_band_importance(
    ipca: IncrementalPCA,
    n_bands: int,
    wavelengths: Optional[np.ndarray],
    n_pcs_for_importance: Optional[int] = None
) -> dict:
    """
    Compute band-importance scores from PCA components.

    Band importance is a variance-weighted sum of absolute loadings
    over the first K PCs (K = n_pcs_for_importance or n_components).
    """
    comps = ipca.components_  # (n_components, n_bands)
    evr = ipca.explained_variance_ratio_  # (n_components,)
    n_components = comps.shape[0]
    if n_bands != comps.shape[1]:
        raise ValueError("n_bands mismatch with PCA components")

    if n_pcs_for_importance is None:
        n_pcs_for_importance = n_components
    K = min(n_pcs_for_importance, n_components)

    weights = evr[:K].reshape(-1, 1)     # (K, 1)
    abs_loadings = np.abs(comps[:K, :])  # (K, n_bands)
    band_importance = (abs_loadings * weights).sum(axis=0)  # (n_bands,)

    band_importance_norm = band_importance / (band_importance.sum() + 1e-12)

    band_indices = np.arange(n_bands)

    result = {
        "band_indices": band_indices,
        "wavelengths": wavelengths,
        "importance": band_importance,
        "importance_norm": band_importance_norm,
        "n_pcs_used": int(K),
    }
    return result


# -------------------------
# Visualization
# -------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_scree(ipca: IncrementalPCA, out_dir: str):
    evr = ipca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)
    xs = np.arange(1, len(evr) + 1)

    plt.figure()
    plt.plot(xs, evr, marker="o")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained variance ratio")
    plt.title("Scree plot (per-component variance)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scree_explained_variance_ratio.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(xs, cum_evr, marker="o")
    plt.xlabel("Principal Component")
    plt.ylabel("Cumulative explained variance ratio")
    plt.title("Scree plot (cumulative variance)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scree_cumulative_explained_variance.png"), dpi=200)
    plt.close()


def plot_band_importance(
    band_info: dict,
    out_dir: str,
    top_k_to_annotate: int = 20
):
    band_indices = band_info["band_indices"]
    wavelengths = band_info["wavelengths"]
    importance = band_info["importance_norm"]

    if wavelengths is not None:
        x = wavelengths
        xlabel = "Wavelength"
    else:
        x = band_indices
        xlabel = "Band index"

    # Curve over all bands
    plt.figure(figsize=(8, 4))
    plt.plot(x, importance, linestyle="-")
    plt.xlabel(xlabel)
    plt.ylabel("Normalized band importance")
    plt.title("Band importance from PCA loadings")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "band_importance_curve.png"), dpi=200)
    plt.close()

    # Bar chart for top-k bands
    ranked_idx = np.argsort(importance)[::-1]
    top_k = ranked_idx[:top_k_to_annotate]

    plt.figure(figsize=(10, 5))
    if wavelengths is not None:
        xticks = wavelengths[top_k]
    else:
        xticks = band_indices[top_k]

    plt.bar(range(len(top_k)), importance[top_k])
    plt.xticks(
        range(len(top_k)),
        [f"{v:.1f}" for v in xticks],
        rotation=45,
        ha="right"
    )
    plt.ylabel("Normalized band importance")
    plt.xlabel("Top bands (wavelength or index)")
    plt.title(f"Top {top_k_to_annotate} important bands")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "band_importance_top_k.png"), dpi=200)
    plt.close()


def plot_pc_loadings(
    ipca: IncrementalPCA,
    out_dir: str,
    wavelengths: Optional[np.ndarray],
    n_pcs_to_plot: int = 5
):
    comps = ipca.components_
    n_pcs = min(n_pcs_to_plot, comps.shape[0])
    band_indices = np.arange(comps.shape[1])

    if wavelengths is not None:
        x = wavelengths
        xlabel = "Wavelength"
    else:
        x = band_indices
        xlabel = "Band index"

    plt.figure(figsize=(10, 6))
    for i in range(n_pcs):
        plt.plot(x, comps[i], label=f"PC{i+1}")
    plt.xlabel(xlabel)
    plt.ylabel("Loading")
    plt.title("PCA loadings for leading PCs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pc_loadings_first_pcs.png"), dpi=200)
    plt.close()


# -------------------------
# Saving metrics
# -------------------------

def save_metrics(
    ipca: IncrementalPCA,
    band_info: dict,
    out_dir: str,
    wavelengths: Optional[np.ndarray],
    band_map: Optional[np.ndarray],
):
    ensure_dir(out_dir)

    # PCA core arrays
    np.save(os.path.join(out_dir, "components.npy"), ipca.components_)
    np.save(os.path.join(out_dir, "explained_variance.npy"), ipca.explained_variance_)
    np.save(
        os.path.join(out_dir, "explained_variance_ratio.npy"),
        ipca.explained_variance_ratio_
    )

    if wavelengths is not None:
        np.save(os.path.join(out_dir, "wavelengths_used.npy"), wavelengths)

    if band_map is not None:
        np.save(os.path.join(out_dir, "band_map_used.npy"), band_map)

    # Band importance arrays
    np.save(os.path.join(out_dir, "band_indices.npy"), band_info["band_indices"])
    np.save(os.path.join(out_dir, "band_importance.npy"), band_info["importance"])
    np.save(
        os.path.join(out_dir, "band_importance_norm.npy"),
        band_info["importance_norm"]
    )

    # CSV for human inspection
    csv_path = os.path.join(out_dir, "band_importance_table.csv")
    with open(csv_path, "w") as f:
        if wavelengths is not None:
            f.write("band_index,wavelength,importance,importance_norm\n")
            for idx, wl, imp, impn in zip(
                band_info["band_indices"],
                wavelengths,
                band_info["importance"],
                band_info["importance_norm"],
            ):
                f.write(f"{idx},{wl:.6f},{imp:.8e},{impn:.8e}\n")
        else:
            f.write("band_index,importance,importance_norm\n")
            for idx, imp, impn in zip(
                band_info["band_indices"],
                band_info["importance"],
                band_info["importance_norm"],
            ):
                f.write(f"{idx},{imp:.8e},{impn:.8e}\n")

    # JSON summary
    summary = {
        "n_components": int(ipca.components_.shape[0]),
        "total_explained_variance_ratio": float(ipca.explained_variance_ratio_.sum()),
        "n_pcs_used_for_importance": int(band_info["n_pcs_used"]),
        "top5_band_indices": [
            int(i) for i in np.argsort(band_info["importance_norm"])[::-1][:5]
        ],
        "band_map_used": band_map is not None,
    }
    if wavelengths is not None:
        top5_idx = np.argsort(band_info["importance_norm"])[::-1][:5]
        summary["top5_wavelengths"] = [float(wavelengths[i]) for i in top5_idx]
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


# -------------------------
# Main CLI
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="PCA-based band-importance exploration over many hyperspectral cubes."
    )
    parser.add_argument(
        "--cube-root",
        type=str,
        default="/media/atesfet/TS801/GHSI/full_spectrum_cubes",
        help="Root directory under which cubes are stored.",
    )
    parser.add_argument(
        "--cube-pattern",
        type=str,
        default="P*/ROI_*/cube.npy",
        help=(
            "Glob pattern (relative to cube-root) to find cube .npy files. "
            "Your cubes are expected at P*/ROI_*/cube.npy."
        ),
    )
    parser.add_argument(
        "--wavelengths-path",
        type=str,
        default=None,
        help="Optional path to full wavelengths.npy (1D array, length B).",
    )
    parser.add_argument(
        "--band-map-path",
        type=str,
        default=None,
        help=(
            "Optional path to band_map.npy (1D int array of length n_bands_cube). "
            "band_map[i] is the index into full wavelengths array for cube band i."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/media/atesfet/TS801/GHSI/pca-explore",
        help="Directory to save PCA metrics and visualizations.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=32,
        help="Number of PCA components to compute.",
    )
    parser.add_argument(
        "--n-samples-per-cube",
        type=int,
        default=10000,
        help="Number of random spectra to sample from each cube.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Total number of spectra per IncrementalPCA batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--n-pcs-for-importance",
        type=int,
        default=10,
        help="Number of leading PCs to use for band-importance computation.",
    )
    parser.add_argument(
        "--n-pcs-to-plot-loadings",
        type=int,
        default=5,
        help="Number of leading PCs for which to plot loadings curves.",
    )
    parser.add_argument(
        "--top-k-bands-plot",
        type=int,
        default=20,
        help="How many top bands to show in the bar-plot of band importance.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable per-band mean-centering normalization.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    print("[INFO] Discovering cubes...")
    cube_paths = find_cube_paths(args.cube_root, args.cube_pattern)
    print(f"[INFO] Found {len(cube_paths)} cube(s).")
    if len(cube_paths) == 0:
        print("[ERROR] No cubes found. Check --cube-root and --cube-pattern.")
        return

    wavelengths_full = maybe_load_wavelengths(args.wavelengths_path)
    band_map = maybe_load_band_map(args.band_map_path)

    ipca, n_bands_cube = run_incremental_pca(
        cube_paths=cube_paths,
        n_components=args.n_components,
        n_samples_per_cube=args.n_samples_per_cube,
        batch_size=args.batch_size,
        seed=args.seed,
        normalize=not args.no_normalize,
    )

    # Determine wavelengths used for the cube bands
    wavelengths_used = None
    if wavelengths_full is not None:
        if band_map is not None:
            # Validate band_map against n_bands_cube and wavelengths_full
            if len(band_map) != n_bands_cube:
                print(
                    f"[WARN] band_map length {len(band_map)} != cube n_bands {n_bands_cube}. "
                    "Ignoring band_map and falling back to direct wavelengths or band indices."
                )
                band_map = None
            elif np.any(band_map < 0) or np.any(band_map >= len(wavelengths_full)):
                print(
                    f"[WARN] band_map has indices outside [0, {len(wavelengths_full)-1}]. "
                    "Ignoring band_map and falling back to direct wavelengths or band indices."
                )
                band_map = None
            else:
                wavelengths_used = wavelengths_full[band_map]
                print(
                    f"[INFO] Using wavelengths from full array of length "
                    f"{len(wavelengths_full)} with band_map of length {len(band_map)}."
                )
        else:
            # No band_map provided: require exact length match
            if len(wavelengths_full) == n_bands_cube:
                wavelengths_used = wavelengths_full
                print(
                    f"[INFO] Using wavelengths array of length {len(wavelengths_full)} "
                    f"directly for cube bands."
                )
            else:
                print(
                    f"[WARN] wavelengths length {len(wavelengths_full)} != cube n_bands "
                    f"{n_bands_cube}, and no valid band_map provided. "
                    "To avoid misalignment, wavelengths will be ignored and "
                    "band indices will be used instead."
                )
                wavelengths_used = None
    else:
        wavelengths_used = None

    band_info = compute_band_importance(
        ipca=ipca,
        n_bands=n_bands_cube,
        wavelengths=wavelengths_used,
        n_pcs_for_importance=args.n_pcs_for_importance,
    )

    save_metrics(ipca, band_info, args.out_dir, wavelengths_used, band_map)

    plot_scree(ipca, args.out_dir)
    plot_band_importance(
        band_info,
        args.out_dir,
        top_k_to_annotate=args.top_k_bands_plot,
    )
    plot_pc_loadings(
        ipca,
        args.out_dir,
        wavelengths=wavelengths_used,
        n_pcs_to_plot=args.n_pcs_to_plot_loadings,
    )

    print(f"[INFO] Done. Outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()