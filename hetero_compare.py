"""
Compare the o_proj rope input/output dumps between DP groups.

Usage:
    python hetero_compare.py [--root <dump_dir>] [--dps 0 1] [--tol 1e-2]
"""
import argparse
import glob
import os
import re

import torch


def head_overlay(rank_tensors):
    """Overlay per-rank [N, nh, H] tensors into [N, sum(nh), H]."""
    norm = []
    for t in rank_tensors:
        t = t.float()
        if t.dim() == 1:
            t = t.unsqueeze(0).unsqueeze(-1)
        elif t.dim() == 2:
            t = t.unsqueeze(1)
        norm.append(t)
    total_h = sum(t.shape[1] for t in norm)
    n_tok = norm[0].shape[0]
    h_dim = norm[0].shape[2]
    out = torch.zeros(n_tok, total_h, h_dim)
    off_h = 0
    for t in norm:
        n, nh, _ = t.shape
        assert n == n_tok, f"token count mismatch: {n} vs {n_tok}"
        out[:, off_h:off_h + nh, :] = t
        off_h += nh
    return out


def load(path):
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu")


def rank_dirs(root, dp):
    return sorted(glob.glob(os.path.join(root, f"dp{dp}_tp*")))


def extract_fwd_num(name):
    m = re.search(r'fwd(\d+)', name)
    return int(m.group(1)) if m else -1


def fwd_dir(rank_dir):
    """Choose the dump directory: prefer the fixed per-rank dir (new layout,
    all probes co-located), else fall back to fwd* subdirs (legacy layout)."""
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

    # Concat per-rank token chunks (SP slices are contiguous in token order,
    # so concat restores the full token order for both DPs).
    def concat_compare(name, mark):
        ra = [load(os.path.join(d, f"{name}.pt")) for d in dirs_a]
        rb = [load(os.path.join(d, f"{name}.pt")) for d in dirs_b]
        if not all(x is not None for x in ra + rb):
            print(f"[{name}] missing: DP{dp_a}={[x is not None for x in ra]} "
                  f"DP{dp_b}={[x is not None for x in rb]} (re-sync)")
            return
        A = torch.cat([x.float().reshape(-1, x.shape[-1]) for x in ra], dim=0)
        B = torch.cat([x.float().reshape(-1, x.shape[-1]) for x in rb], dim=0)
        n = min(A.shape[0], B.shape[0])
        diff = (A[:n] - B[:n]).abs().float()
        maxdiff = diff.max().item()
        nbad = int((diff.max(-1).values > 1e-3).sum().item())
        mark_out = f"   <== {mark}" if maxdiff > args.tol else ""
        print(f"[{name}] maxdiff={maxdiff:.6e}  tokens={n}  bad_positions={nbad}/{n}{mark_out}")

    # attn_hidden_in is full tokens, no heads -> direct rank0 compare.
    def direct_compare(name, mark):
        fa = load(os.path.join(dirs_a[0], f"{name}.pt"))
        fb = load(os.path.join(dirs_b[0], f"{name}.pt"))
        if fa is None or fb is None:
            print(f"[{name}] missing: DP{dp_a}={fa is not None} DP{dp_b}={fb is not None} (re-sync)")
            return
        n = min(fa.shape[0], fb.shape[0])
        diff = (fa[:n].float() - fb[:n].float()).abs()
        maxdiff = diff.max().item()
        mark_out = f"   <== {mark}" if maxdiff > args.tol else ""
        print(f"[{name}] maxdiff={maxdiff:.6e}  shape={tuple(fa.shape)} vs {tuple(fb.shape)}{mark_out}")

    # Per-rank heads with full tokens -> head-overlay.
    def overlay_compare(name, mark):
        ra = [load(os.path.join(d, f"{name}.pt")) for d in dirs_a]
        rb = [load(os.path.join(d, f"{name}.pt")) for d in dirs_b]
        if not all(x is not None for x in ra + rb):
            print(f"[{name}] missing: DP{dp_a}={[x is not None for x in ra]} "
                  f"DP{dp_b}={[x is not None for x in rb]} (re-sync)")
            return
        try:
            A = head_overlay(ra)
            B = head_overlay(rb)
        except AssertionError as e:
            print(f"[{name}] head_overlay failed: {e}")
            return
        n = min(A.shape[0], B.shape[0])
        diff = (A[:n] - B[:n]).abs().float()
        maxdiff = diff.max().item()
        nbad = int((diff.max(-1).values > 1e-3).sum().item())
        mark_out = f"   <== {mark}" if maxdiff > args.tol else ""
        print(f"[{name}] maxdiff={maxdiff:.6e}  heads={A.shape[1]}  tokens={n}  "
              f"bad_positions={nbad}/{n}{mark_out}")

    concat_compare("oproj_out", "O_PROJ OUTPUT DIVERGES")
    concat_compare("model_embed", "RAW EMBEDDING DIVERGES")
    concat_compare("model_embed_hc", "EMBEDDING (REPEATED, LAYER INPUT) DIVERGES")
    concat_compare("model_pre_layer0", "MODEL PRE-LAYER0 INPUT DIVERGES")
    concat_compare("model_pre_layer0_meta", "MODEL PRE-LAYER0 META (PTR)")
    concat_compare("layer0_input", "LAYER0 ENTRY INPUT (PRE-CLONE) DIVERGES")
    concat_compare("hc_residual", "HC_RESIDUAL (LAYER INPUT CLONE) DIVERGES")


if __name__ == "__main__":
    main()