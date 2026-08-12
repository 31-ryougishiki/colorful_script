#!/usr/bin/env python3
"""
Compare dump tensors between two DP groups (e.g., DP0 vs DP1).
All tensors are concatenated along the token dimension (first axis).
"""
import argparse
import glob
import os
import re

import torch


def load(path):
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu")


def rank_dirs(root, dp):
    return sorted(glob.glob(os.path.join(root, f"dp{dp}_tp*")))


def fwd_dir(rank_dir):
    if os.path.exists(os.path.join(rank_dir, "oproj_out.pt")):
        return rank_dir
    fwds = sorted(glob.glob(os.path.join(rank_dir, "fwd*")))
    if not fwds:
        return rank_dir
    for d in reversed(fwds):
        if os.path.exists(os.path.join(d, "oproj_input.pt")):
            return d
    best = max(fwds, key=lambda d: len(glob.glob(os.path.join(d, "*.pt"))))
    return best


def compare_experts(name, dirs_a, dirs_b, dp_a, dp_b):
    """Compare per-token SELECTED expert ids (integer tensor) between DP groups.
    Reports the fraction of (token,k) slots where the two DPs chose DIFFERENT
    experts — 0.0 means identical routing."""
    ra = [load(os.path.join(d, f"{name}.pt")) for d in dirs_a]
    rb = [load(os.path.join(d, f"{name}.pt")) for d in dirs_b]
    if not all(x is not None for x in ra + rb):
        print(f"[{name}] missing: DP{dp_a}={[x is not None for x in ra]} "
              f"DP{dp_b}={[x is not None for x in rb]} (re-sync)")
        return None
    A = torch.cat([x.long().reshape(-1) for x in ra], dim=0)
    B = torch.cat([x.long().reshape(-1) for x in rb], dim=0)
    n = min(A.shape[0], B.shape[0])
    same = (A[:n] == B[:n]).float().mean().item()
    mism = n - int((A[:n] == B[:n]).sum().item())
    print(f"[{name}] expert-selection match={same*100:.2f}%  mismatched_slots={mism}/{n}")
    return same


def compare_probe(name, dirs_a, dirs_b, dp_a, dp_b, tol):
    """Compare a probe by concatenating all ranks along the token dimension."""
    ra = [load(os.path.join(d, f"{name}.pt")) for d in dirs_a]
    rb = [load(os.path.join(d, f"{name}.pt")) for d in dirs_b]
    if not all(x is not None for x in ra + rb):
        print(f"[{name}] missing: DP{dp_a}={[x is not None for x in ra]} "
              f"DP{dp_b}={[x is not None for x in rb]} (re-sync)")
        return

    # Print shapes for diagnosis
    print(f"[{name}] DP{dp_a} shapes: {[x.shape for x in ra]}")
    print(f"[{name}] DP{dp_b} shapes: {[x.shape for x in rb]}")

    # Concatenate along token dimension (dim=0)
    try:
        A = torch.cat([x.float() for x in ra], dim=0)
        B = torch.cat([x.float() for x in rb], dim=0)
    except RuntimeError as e:
        print(f"[{name}] concatenation failed: {e}")
        return

    if A.shape != B.shape:
        print(f"[{name}] shape mismatch after concat: A {A.shape} vs B {B.shape} - skip")
        return

    # Flatten all non-token dimensions for per-token max diff
    A_flat = A.view(A.size(0), -1)
    B_flat = B.view(B.size(0), -1)
    n = A_flat.size(0)

    diff = (A_flat - B_flat).abs().float()
    maxdiff = diff.max().item()
    nbad = int((diff.max(dim=1)[0] > tol).sum().item())
    print(f"[{name}] maxdiff={maxdiff:.6e}  tokens={n}  bad_positions={nbad}/{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dps", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--root", default="hetero_debug")
    ap.add_argument("--tol", type=float, default=1e-2)
    args = ap.parse_args()
    dp_a, dp_b = args.dps
    dirs_a = [fwd_dir(d) for d in rank_dirs(args.root, dp_a)]
    dirs_b = [fwd_dir(d) for d in rank_dirs(args.root, dp_b)]
    if not dirs_a or not dirs_b:
        print(f"no dumps for dp{args.dps}: {dirs_a} {dirs_b}")
        return

    def fmt(d):
        return os.path.basename(os.path.dirname(d)) + '/' + os.path.basename(d)
    print(f"DP{dp_a}: {[fmt(d) for d in dirs_a]}")
    print(f"DP{dp_b}: {[fmt(d) for d in dirs_b]}")

    # Compare top-level probes
    compare_probe("oproj_out", dirs_a, dirs_b, dp_a, dp_b, args.tol)

    # Discover layer indices
    first_layer_files = sorted(glob.glob(os.path.join(dirs_a[0], "layer*_hc_pre_y.pt")))
    if not first_layer_files:
        print("No per‑layer dump files found.")
        return

    layer_indices = sorted(
        int(re.search(r"layer(\d+)_", os.path.basename(f)).group(1))
        for f in first_layer_files
    )

    probes = [
        ("hc_pre_y", "HC_PRE"),
        ("attn_in", "ATTN_IN"),
        ("attn_out", "ATTN_OUT"),
        ("mlp_in", "MLP_IN"),
        ("mlp_router_in", "MLP_ROUTER_IN"),
        ("mlp_router", "MLP_ROUTER"),
        ("mlp_fused", "MLP_FUSED"),
        ("mlp_shared", "MLP_SHARED"),
        ("mlp_routed", "MLP_ROUTED"),
        ("mlp_combined", "MLP_COMBINED"),
        ("mlp_out", "MLP_OUT"),
    ]

    for idx in layer_indices:
        for suffix, label in probes:
            name = f"layer{idx}_{suffix}"
            compare_probe(name, dirs_a, dirs_b, dp_a, dp_b, args.tol)

    # Compare per-token selected experts (moe{idx}_expert_ids, the ACTUAL
    # routing computed in AscendUnquantizedFusedMoEMethod.apply).
    moe_files = sorted(glob.glob(os.path.join(dirs_a[0], "moe*_expert_ids.pt")))
    if moe_files:
        moe_indices = sorted(
            int(re.search(r"moe(\d+)_expert_ids", os.path.basename(f)).group(1))
            for f in moe_files
        )
        for idx in moe_indices:
            compare_experts(f"moe{idx}_expert_ids", dirs_a, dirs_b, dp_a, dp_b)
    else:
        print("no moe*_expert_ids dumps found")


if __name__ == "__main__":
    main()