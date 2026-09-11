"""Rank-striped large-domain lobe generation (Tier-2 dataset).

A thin driver around ResMill's own ``build_jobs`` + ``generate_sample``
so ResMill itself stays untouched. The only behavioural difference from
``resmill.dataset.cli`` is the writer: ResFlow trains on binary facies
only, so ``poro`` / ``perm`` / ``facies_alluvsim`` are discarded instead
of written (1.4 TB -> 236 GB at count=200,000, 192x192x32).

Shards land at ``<output_dir>/lobe/shard_NNNNNN`` with a globally unique
id ``rank * 100 + shard_idx``, matching the consolidated layout that
``data_reservoirs.py`` expects, so no separate combine pass is needed.

Usage (under SLURM, one rank per core)::

    srun -n 1152 --cpu-bind=cores \
        python scripts/tier2/gen_large_lobes.py scripts/tier2/config_lobes_192.json

Runs serially when ``SLURM_NTASKS`` is unset (handy for smoke tests):

    python scripts/tier2/gen_large_lobes.py CONFIG.json --limit 4
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from resmill.dataset.generate import generate_sample
from resmill.dataset.sampling import build_jobs
from resmill.dataset.schemas import slim_columns

MAX_SHARDS_PER_RANK = 100


class FaciesShardWriter:
    """ShardWriter minus the property arrays.

    Emits ``facies.npy`` + ``params.parquet`` + ``params_slim.parquet``,
    row-aligned, assembled in a ``.tmp`` sibling and atomically renamed
    so a shard is either complete or absent (same contract as ResMill's
    ``ShardWriter``).
    """

    def __init__(self, output_dir, rank, shard_size):
        self.output_dir = Path(output_dir)
        self.rank = int(rank)
        self.shard_size = int(shard_size)
        self._shard_idx = 0
        self._facies: list = []
        self._meta: list = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add(self, facies, meta):
        self._facies.append(facies)
        self._meta.append(meta)
        if len(self._facies) >= self.shard_size:
            self._flush()

    def close(self):
        if self._facies:
            self._flush()

    def _flush(self):
        if self._shard_idx >= MAX_SHARDS_PER_RANK:
            raise RuntimeError(
                f"rank {self.rank} exceeded {MAX_SHARDS_PER_RANK} shards; "
                f"raise MAX_SHARDS_PER_RANK or the shard_size"
            )
        gid = self.rank * MAX_SHARDS_PER_RANK + self._shard_idx
        name = f"shard_{gid:06d}"
        shard_dir = self.output_dir / name
        tmp_dir = self.output_dir / (name + ".tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        np.save(tmp_dir / "facies.npy", np.stack(self._facies).astype(np.int8))

        all_keys = sorted({k for m in self._meta for k in m.keys()})
        pq.write_table(
            pa.Table.from_pydict({k: [m.get(k) for m in self._meta]
                                  for k in all_keys}),
            tmp_dir / "params.parquet",
        )

        slim_keys: set[str] = set()
        for m in self._meta:
            slim_keys.update(slim_columns(m.get("layer_type", "")))
        pq.write_table(
            pa.Table.from_pydict({
                k: [(m.get(k) if k in slim_columns(m.get("layer_type", ""))
                     else None) for m in self._meta]
                for k in sorted(slim_keys)
            }),
            tmp_dir / "params_slim.parquet",
        )

        if shard_dir.exists():
            raise FileExistsError(
                f"shard {shard_dir} already exists; pick a fresh output_dir"
            )
        os.rename(tmp_dir, shard_dir)

        self._shard_idx += 1
        self._facies.clear()
        self._meta.clear()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, default=None,
                    help="take only the first N jobs (smoke tests)")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    out_root = args.output_dir or cfg["output_dir"]
    out_root = os.path.expandvars(os.path.expanduser(out_root))

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))

    jobs = build_jobs(cfg["layers"], cfg["seed"])
    n_jobs = min(len(jobs), args.limit) if args.limit else len(jobs)
    my_indices = list(range(rank, n_jobs, world))

    if rank == 0:
        print(f"[rank 0] total_jobs={n_jobs} world={world} "
              f"jobs_per_rank~={len(my_indices)} out={out_root}", flush=True)

    # One layer type per config; keep the <layer_type>/shard_* layout.
    layer_name = next(iter(cfg["layers"]))
    writer = FaciesShardWriter(Path(out_root) / layer_name, rank,
                               cfg["shard_size"])
    failures_path = Path(out_root) / f"failures_r{rank:04d}.jsonl"
    n_failed = 0
    t0 = time.perf_counter()

    for n_done, i in enumerate(my_indices, 1):
        job = jobs[i]
        try:
            facies, _poro, _perm, _allu, meta = generate_sample(job, cfg["grid"])
            writer.add(facies, meta)
        except Exception as exc:  # keep the sweep alive; record and move on
            n_failed += 1
            with open(failures_path, "a") as fh:
                fh.write(json.dumps({"job_index": int(i),
                                     "seed": int(job["seed"]),
                                     "error": repr(exc)[:500]}) + "\n")
        if rank == 0 and n_done % 20 == 0:
            el = time.perf_counter() - t0
            rate = el / n_done
            print(f"[rank 0] {n_done}/{len(my_indices)} "
                  f"{rate:.2f}s/sample eta={rate * (len(my_indices) - n_done) / 60:.1f}min",
                  flush=True)

    writer.close()
    el = time.perf_counter() - t0
    print(f"[rank {rank}] done n={len(my_indices)} failed={n_failed} "
          f"wall={el / 60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
