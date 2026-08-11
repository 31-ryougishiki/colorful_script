"""Compare the o_proj rope input/output dumps between DP groups.

The only remaining hetero dump (VLLM_HETERO_DEBUG + VLLM_HETERO_OPROJ_DET) is
the o_proj rope at dsa_v1.py: it writes `oproj_rope_in.pt` (pre-rope,
padding-zeroed) and `oproj_rope_out.pt` (post-rope) per rank.

Each rank holds the FULL token set but a different head subset, so we head-
overlay the per-rank tensors and compare DP0 vs DP1:

- oproj_rope_in  ~0  AND oproj_rope_out  ~0  -> rope is deterministic, bug is
  elsewhere.
- oproj_rope_in  ~0  AND oproj_rope_out != 0  -> the rope OP itself diverges
  (identical input + cos -> different output).
- oproj_rope_in  != 0 -> the divergence is already in the rope input (attention
  output / o_proj_input assembly).

Usage:
    python hetero_compare.py [--root <dump_dir>] [--dps 0 1] [--tol 1e-2]
"""
import argparse
import glob
import os

import torch


def head_overlay(rank_tensors):
    """Overlay per-rank [N, nh, H] tensors (full tokens, different head subset)
    into a [N, sum(nh), H] tensor in global-head order."""
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
        assert n == n_tok, f"head_overlay requires same token count, got {n} vs {n_tok}"
        out[:, off_h:off_h + nh, :] = t
        off_h += nh
    return out


def load(path):
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu")


def rank_dirs(root, dp):
    return sorted(glob.glob(os.path.join(root, f"dp{dp}_tp*")))


def fwd_dir(rank_dir):
    """Pick the fwd dir that actually contains the (one-time) rope dumps.
    The dump lands in the first QUALIFYING forward (real prefill), which is not
    necessarily the latest fwd (warmup batches may come after it)."""
    fwds = sorted(glob.glob(os.path.join(rank_dir, "fwd*")))
    if not fwds:
        return rank_dir  # legacy flat layout
    for d in fwds:
        if os.path.exists(os.path.join(d, "oproj_rope_in.pt")) \
                and os.path.exists(os.path.join(d, "oproj_rope_out.pt")):
            return d
    return fwds[-1]  # fall back to the latest


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
    print(f"DP{dp_a}: {[os.path.basename(os.path.dirname(d)) + '/' + os.path.basename(d) for d in dirs_a]}")
    print(f"DP{dp_b}: {[os.path.basename(os.path.dirname(d)) + '/' + os.path.basename(d) for d in dirs_b]}")

    for name, mark in (("oproj_rope_in", "ROPE INPUT DIVERGES"),
                       ("oproj_rope_out", "ROPE OUTPUT DIVERGES")):
        ra = [load(os.path.join(d, f"{name}.pt")) for d in dirs_a]
        rb = [load(os.path.join(d, f"{name}.pt")) for d in dirs_b]
        if all(x is not None for x in ra + rb):
            try:
                A = head_overlay(ra)
                B = head_overlay(rb)
                n = min(A.shape[0], B.shape[0])
                d = (A[:n] - B[:n]).abs().max().item()
                nbad = int(((A[:n] - B[:n]).abs().max(-1).values > 1e-3).sum().item())
                mark_out = f"   <== {mark}" if d > args.tol else ""
                print(f"[{name}] maxdiff={d:.6e}  heads={A.shape[1]}  tokens={n}  "
                      f"bad_positions={nbad}/{n}{mark_out}")
            except Exception as e:
                print(f"[{name}] compare failed: {e!r}")
        else:
            print(f"[{name}] missing: DP{dp_a}={[x is not None for x in ra]} "
                  f"DP{dp_b}={[x is not None for x in rb]} (re-sync dsa_v1.py)")


if __name__ == "__main__":
    main()
