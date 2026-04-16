#!/usr/bin/env python3
"""Fit k-means centroids from SGLang KV dump files for use with SGLANG_KV_CENTROIDS_PATH.

Expects files named kv_calibration_layer_<L>.pt as written by sglang-kmeans
(triton_backend) with keys 'k', 'v' (and optional 'indices').

Writes k_layer_<L>_clusters_<N>_centers.pt and v_layer_<L>_clusters_<N>_centers.pt
with tensors of shape (N, num_kv_heads * head_dim), matching MHATokenToKVPool._load_kv_centroids.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Optional, Tuple

import torch


def _layer_id_from_name(path: str) -> Optional[int]:
    m = re.search(r"kv_calibration_layer_(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else None


def _fit_one(
    k: torch.Tensor,
    v: torch.Tensor,
    n_clusters: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return centroids float32 tensors (n_clusters, flat_dim)."""
    if k.shape != v.shape:
        raise ValueError(f"k shape {k.shape} != v shape {v.shape}")
    if k.dim() != 3:
        raise ValueError(f"expected k with 3 dims (tokens, heads, head_dim), got {k.shape}")
    t, h, d = k.shape
    flat = h * d
    if t < n_clusters:
        raise ValueError(
            f"need at least n_clusters={n_clusters} tokens, got {t} in calibration file"
        )

    xk = k.reshape(t, flat).to(torch.float32).numpy()
    xv = v.reshape(t, flat).to(torch.float32).numpy()

    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise SystemExit(
            "sklearn is required: pip install scikit-learn"
        ) from e

    km_k = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=10,
        max_iter=300,
    ).fit(xk)
    km_v = KMeans(
        n_clusters=n_clusters,
        random_state=seed + 1,
        n_init=10,
        max_iter=300,
    ).fit(xv)
    ck = torch.from_numpy(km_k.cluster_centers_).to(torch.float32)
    cv = torch.from_numpy(km_v.cluster_centers_).to(torch.float32)
    return ck, cv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump-dir", required=True, help="Directory with kv_calibration_layer_*.pt")
    p.add_argument("--out-dir", required=True, help="Output directory for *_centers.pt")
    p.add_argument("--n-clusters", type=int, default=16, help="K-means clusters (default 16)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.dump_dir, "kv_calibration_layer_*.pt")))
    if not paths:
        raise SystemExit(f"No kv_calibration_layer_*.pt under {args.dump_dir}")

    for path in paths:
        lid = _layer_id_from_name(path)
        if lid is None:
            continue
        blob = torch.load(path, map_location="cpu", weights_only=True)
        k = blob["k"]
        v = blob["v"]
        ck, cv = _fit_one(k, v, args.n_clusters, args.seed)
        k_out = os.path.join(
            args.out_dir,
            f"k_layer_{lid}_clusters_{args.n_clusters}_centers.pt",
        )
        v_out = os.path.join(
            args.out_dir,
            f"v_layer_{lid}_clusters_{args.n_clusters}_centers.pt",
        )
        torch.save(ck, k_out)
        torch.save(cv, v_out)
        print(f"layer {lid}: wrote {k_out} and {v_out} shape {tuple(ck.shape)}")


if __name__ == "__main__":
    main()
