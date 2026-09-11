"""Paired check: is a 64-crop of a large domain distributed like a native 64 volume?

The Tier-2 premise is that ResFlow's training volumes are whole ResMill
domains, not windows of a larger field, so a MultiDiffusion tile is out
of distribution. Because the new sweep reuses build_jobs with the same
seed and count, both datasets realize the SAME 200,000 parameter draws
-- so we can pair by ResMill seed and vary only the domain size.

Compares, per matched pair:
  native  = the published 64x64x32 volume (whole 80^3 domain, cropped)
  crop    = a centred 64x64x32 window of the new 192x192x32 volume

If the premise holds, body-scale statistics differ systematically, and
the gap should widen with requested body width.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from resbench import anisotropy as an
from resbench import metrics

SCRATCH = os.environ["SCRATCH"]
OLD = f"{SCRATCH}/SiliciclasticReservoirs/lobe"
NEW = f"{SCRATCH}/resmill_lobes_192/lobe"
N_PAIRS = 150
RNG = np.random.default_rng(20260801)


def index_by_seed(root, n_shards_scan):
    """seed -> (shard_dir, row). Scans a prefix of shards only."""
    out = {}
    for d in sorted(glob.glob(f"{root}/shard_*"))[:n_shards_scan]:
        t = pq.read_table(f"{d}/params.parquet", columns=["seed"])
        for row, s in enumerate(t["seed"].to_pylist()):
            out.setdefault(int(s), (d, row))
    return out


def stats(vol, azimuth):
    labels, _ = metrics.label_geobodies(vol)
    sizes = metrics.geobody_sizes(vol, labels=labels)
    major = an.directional_chords(vol, azimuth)
    minor = an.directional_chords(vol, azimuth + 90.0)
    return {
        "ntg": float(vol.mean()),
        "n_bodies": int(len(sizes)),
        "median_body": float(np.median(sizes)) if len(sizes) else np.nan,
        "largest_frac": metrics.largest_fraction(vol, labels=labels),
        "mean_chord_major": float(major.mean()) if major.size else np.nan,
        "mean_chord_minor": float(minor.mean()) if minor.size else np.nan,
        "anisotropy": float(major.mean() / minor.mean())
        if major.size and minor.size else np.nan,
    }


def main():
    print("indexing seeds ...", flush=True)
    old_idx = index_by_seed(OLD, 40)
    new_idx = index_by_seed(NEW, 400)
    shared = sorted(set(old_idx) & set(new_idx))
    print(f"old={len(old_idx)} new={len(new_idx)} shared={len(shared)}", flush=True)
    if len(shared) < 20:
        raise SystemExit("too few shared seeds -- job lists differ?")

    pick = RNG.choice(len(shared), size=min(N_PAIRS, len(shared)), replace=False)
    rows = []
    for k, i in enumerate(pick):
        seed = shared[i]
        od, orow = old_idx[seed]
        nd, nrow = new_idx[seed]
        ov = np.load(f"{od}/facies.npy", mmap_mode="r")[orow]
        nv = np.load(f"{nd}/facies.npy", mmap_mode="r")[nrow]
        p = pq.read_table(f"{nd}/params_slim.parquet").slice(nrow, 1).to_pylist()[0]
        az = float(p["azimuth"])
        c = nv.shape[0] // 2
        crop = np.asarray(nv[c - 32:c + 32, c - 32:c + 32, :])
        r = {"seed": seed, "azimuth": az, "width_cells": float(p["width_cells"]),
             "asp": float(p["asp"]), "req_ntg": float(p["ntg"])}
        for tag, v in (("native", np.asarray(ov)), ("crop", crop)):
            for kk, vv in stats(v, az).items():
                r[f"{tag}_{kk}"] = vv
        rows.append(r)
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(pick)}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    out = Path(__file__).with_name("paired_marginal_check.csv")
    df.to_csv(out, index=False)

    print(f"\n{len(df)} matched pairs (identical ResMill seed and parameters)\n")
    print(f"{'statistic':>18} {'native 64^3':>12} {'crop of 192':>12} {'rel diff':>10}")
    for m in ["ntg", "n_bodies", "median_body", "largest_frac",
              "mean_chord_major", "mean_chord_minor", "anisotropy"]:
        a = df[f"native_{m}"].mean()
        b = df[f"crop_{m}"].mean()
        print(f"{m:>18} {a:>12.4f} {b:>12.4f} {(b - a) / a * 100:>9.1f}%")

    print("\nby requested body width (mean chord along major axis):")
    print(f"{'width_cells bin':>18} {'n':>5} {'native':>9} {'crop':>9} {'rel diff':>10}")
    for lo, hi in [(0, 32), (32, 48), (48, 64), (64, 100)]:
        s = df[(df.width_cells >= lo) & (df.width_cells < hi)]
        if len(s) < 3:
            continue
        a = s["native_mean_chord_major"].mean()
        b = s["crop_mean_chord_major"].mean()
        print(f"{f'[{lo},{hi})':>18} {len(s):>5} {a:>9.2f} {b:>9.2f} "
              f"{(b - a) / a * 100:>9.1f}%")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
