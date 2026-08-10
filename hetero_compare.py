"""Compare heterogeneous-debug dumps between DP groups to find the first diverging layer.

How it works:
- Each NPU rank dumps `hetero_debug/dp{dp}_tp{tp}/fwd{n}/{input_hidden,layer{i},input_ids}.pt`
  when VLLM_HETERO_DEBUG=1 (up to VLLM_HETERO_DEBUG_N batches per rank).
- DP ranks are data-parallel replicas: DP0 (TP=3) and DP1 (TP=4) must produce
  identical hidden states for the SAME input. We auto-pick the fwd index whose
  input_ids match across the two DPs, then find the first layer that diverges.
- SP splits the sequence across each DP's TP ranks, so per-layer tensors are
  per-rank chunks; concatenating the ranks' chunks in order restores the full
  sequence (token x hc_mult channels).

Usage:
    python hetero_compare.py                 # DP0 vs DP1, auto-pick matching fwd
    python hetero_compare.py --dps 0 2       # any pair
    python hetero_compare.py --dps 1 2       # two TP=4 groups (sanity: should match)
    python hetero_compare.py --fwd 0         # force a specific fwd index
"""
import argparse
import glob
import hashlib
import os
import re

import torch


def _to_tokens(x: torch.Tensor) -> torch.Tensor:
    """Flatten to [-1, hidden]. Dim 0 is the (per-rank) token dim; concatenating
    rank chunks in order restores the full sequence (x hc_mult channels)."""
    return x.float().reshape(-1, x.shape[-1])


def head_aligned(rank_tensors: list[torch.Tensor]) -> torch.Tensor:
    """Place each rank's [N, nh, H] (or 1-D [nh]) tensor into a full
    [sum(N), sum(nh), H] tensor by concatenating token chunks and head chunks in
    rank order, so differently-sharded DPs become directly comparable."""
    if rank_tensors[0].dim() == 1:
        rank_tensors = [t.float().unsqueeze(0).unsqueeze(-1) for t in rank_tensors]
    else:
        rank_tensors = [t.float() for t in rank_tensors]
    total_h = sum(t.shape[1] for t in rank_tensors)
    n_tok = sum(t.shape[0] for t in rank_tensors)
    h_dim = rank_tensors[0].shape[2]
    out = torch.zeros(n_tok, total_h, h_dim)
    off_t = off_h = 0
    for t in rank_tensors:
        n, nh, _ = t.shape
        out[off_t:off_t + n, off_h:off_h + nh, :] = t
        off_t += n
        off_h += nh
    return out


def load(path):
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu")


def rank_dirs(root: str, dp: int):
    return sorted(glob.glob(os.path.join(root, f"dp{dp}_tp*")))


def fwd_dir(rank_dir: str, fwd: int | None) -> str:
    """Pick a fwd{n}/ subdir under a rank dir: --fwd index, else the latest."""
    fwds = sorted(glob.glob(os.path.join(rank_dir, "fwd*")))
    if not fwds:
        if glob.glob(os.path.join(rank_dir, "*.pt")):
            return rank_dir  # legacy flat layout
        raise SystemExit(
            f"no .pt files under {rank_dir}; run with the right --root, or set "
            f"--fwd. Contents: {sorted(os.listdir(rank_dir))[:20]}"
        )
    if fwd is None:
        return fwds[-1]
    for d in fwds:
        if os.path.basename(d).startswith(f"fwd{fwd}_") or os.path.basename(d) == f"fwd{fwd}":
            return d
    raise SystemExit(f"--fwd {fwd} not found under {rank_dir}: {[os.path.basename(d) for d in fwds]}")


def list_fwds(rank_dir: str) -> list[int]:
    out = set()
    for d in glob.glob(os.path.join(rank_dir, "fwd*")):
        m = re.match(r"fwd(\d+)", os.path.basename(d))
        if m:
            out.add(int(m.group(1)))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dps", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--root", default="hetero_debug")
    ap.add_argument("--fwd", type=int, default=None, help="force a specific fwd index")
    ap.add_argument("--tol", type=float, default=1e-2,
                    help="max-abs-diff above this counts as a divergence")
    args = ap.parse_args()
    dp_a, dp_b = args.dps
    dirs_a, dirs_b = rank_dirs(args.root, dp_a), rank_dirs(args.root, dp_b)
    if not dirs_a or not dirs_b:
        print(f"no dumps for dp{args.dps}: {dirs_a} {dirs_b}")
        return

    # Auto-select the fwd index whose input_ids match across the two DPs.
    fwds_a = sorted(set().union(*[list_fwds(d) for d in dirs_a]))
    fwds_b = sorted(set().union(*[list_fwds(d) for d in dirs_b]))
    common = sorted(set(fwds_a) & set(fwds_b))
    if args.fwd is not None:
        chosen = args.fwd
    else:
        chosen = None
        for f in common:
            ida = load(os.path.join(fwd_dir(dirs_a[0], f), "input_ids.pt"))
            idb = load(os.path.join(fwd_dir(dirs_b[0], f), "input_ids.pt"))
            if (ida is not None and idb is not None and ida.shape == idb.shape
                    and torch.equal(ida, idb)):
                chosen = f
                break
        if chosen is None:
            print(f"available fwds: DP{dp_a}={fwds_a} DP{dp_b}={fwds_b} (common={common})")
            for f in common:
                ida = load(os.path.join(fwd_dir(dirs_a[0], f), "input_ids.pt"))
                idb = load(os.path.join(fwd_dir(dirs_b[0], f), "input_ids.pt"))
                sha = None if ida is None else int(hashlib.md5(ida.cpu().numpy().tobytes()).hexdigest(), 16) % 100000
                shb = None if idb is None else int(hashlib.md5(idb.cpu().numpy().tobytes()).hexdigest(), 16) % 100000
                print(f"  fwd{f}: input_ids len={0 if ida is None else ida.numel()} vs "
                      f"{0 if idb is None else idb.numel()}, hash={sha} vs {shb}")
            print("No fwd has matching input_ids across the two DPs. Send the SAME prompt "
                  "repeatedly so each DP gets an identical copy, then re-run.")
            return
    print(f"Comparing fwd{chosen} "
          + ("(--fwd)" if args.fwd is not None else "(auto-picked, input_ids match)"))
    dirs_a = [fwd_dir(d, chosen) for d in dirs_a]
    dirs_b = [fwd_dir(d, chosen) for d in dirs_b]
    print(f"DP{dp_a}: {[os.path.basename(os.path.dirname(d)) + '/' + os.path.basename(d) for d in dirs_a]}")
    print(f"DP{dp_b}: {[os.path.basename(os.path.dirname(d)) + '/' + os.path.basename(d) for d in dirs_b]}")

    # --- input_hidden: compare the overlapping (same-token) prefix ---
    ha = load(os.path.join(dirs_a[0], "input_hidden.pt"))
    hb = load(os.path.join(dirs_b[0], "input_hidden.pt"))
    if ha is not None and hb is not None:
        fa, fb = _to_tokens(ha), _to_tokens(hb)
        n = min(fa.shape[0], fb.shape[0])
        d = (fa[:n] - fb[:n]).abs().max().item()
        print(f"[embedding] maxdiff={d:.6e}  shape={tuple(ha.shape)} vs {tuple(hb.shape)}")

    # --- layer-0 internals (attention vs MoE) ---
    for name in ("layer_attn_in", "layer_attn_out", "layer_mlp_in", "layer_mlp_out"):
        full_a, full_b = [], []
        for d in dirs_a:
            h = load(os.path.join(d, f"{name}.pt"))
            if h is None:
                break
            full_a.append(_to_tokens(h))
        for d in dirs_b:
            h = load(os.path.join(d, f"{name}.pt"))
            if h is None:
                break
            full_b.append(_to_tokens(h))
        if not full_a or not full_b:
            continue
        A = torch.cat(full_a, dim=0)
        B = torch.cat(full_b, dim=0)
        n = min(A.shape[0], B.shape[0])
        d = (A[:n] - B[:n]).abs().max().item()
        mark = "   <== ATTENTION DIVERGES" if name == "layer_attn_out" and d > args.tol else ""
        if name == "layer_mlp_out" and d > args.tol:
            mark = "   <== MLP/MoE DIVERGES"
        nan_a = bool(A.isnan().any().item())
        nan_b = bool(B.isnan().any().item())
        extra = ""
        if nan_a or nan_b:
            # Which rank holds the NaN (concatenated in dir order), and where.
            for dp, dirs, full in ((dp_a, dirs_a, full_a), (dp_b, dirs_b, full_b)):
                if not any(t.isnan().any().item() for t in full):
                    continue
                for r, d_ in enumerate(dirs):
                    t = full[r]
                    if not t.isnan().any().item():
                        continue
                    nan_rows = torch.isnan(t).any(dim=-1)
                    pos = nan_rows.nonzero(as_tuple=True)[0].tolist()
                    extra += f"  DP{dp} rank{r}({os.path.basename(d_)}) nan_positions={pos[:20]}{'...' if len(pos) > 20 else ''}"
        print(f"[layer0 {name}] maxdiff={d:.6e}  nan_dp{dp_a}={nan_a} nan_dp{dp_b}={nan_b} "
              f"tokens={A.shape[0]} vs {B.shape[0]}{extra}{mark}")

    # --- DSA attention op inputs/outputs (head-aligned) ---
    # If attn_op_q / attn_op_sink are bit-identical but attn_op_out differs,
    # the CANN op itself is at fault for asymmetric per-rank heads.
    na = [load(os.path.join(d, "attn_op_nheads.pt")) for d in dirs_a]
    nb = [load(os.path.join(d, "attn_op_nheads.pt")) for d in dirs_b]
    if all(x is not None for x in na + nb):
        print(f"per-rank n_local_heads: DP{dp_a}={[int(x.item()) for x in na]} "
              f"DP{dp_b}={[int(x.item()) for x in nb]}")
    for name in ("attn_op_q", "attn_op_kv", "attn_op_sink", "attn_op_out"):
        ranks_a, ranks_b = [], []
        for d in dirs_a:
            h = load(os.path.join(d, f"{name}.pt"))
            if h is None:
                break
            ranks_a.append(h)
        for d in dirs_b:
            h = load(os.path.join(d, f"{name}.pt"))
            if h is None:
                break
            ranks_b.append(h)
        if not ranks_a or not ranks_b:
            continue
        A = head_aligned(ranks_a)
        B = head_aligned(ranks_b)
        n = min(A.shape[0], B.shape[0])
        d = (A[:n] - B[:n]).abs().max().item()
        mark = "   <== OP OUTPUT DIFFERS (input matched)" if name == "attn_op_out" and d > args.tol else ""
        print(f"[op {name}] maxdiff={d:.6e}  heads={A.shape[1]}  tokens={n}{mark}")

    # --- Batch/decode/prefill metadata that decides the [decode:actual] slice ---
    ma = load(os.path.join(dirs_a[0], "attn_meta.pt"))
    mb = load(os.path.join(dirs_b[0], "attn_meta.pt"))
    if ma is not None and mb is not None:
        for k in ("num_actual_tokens", "num_decode_tokens", "num_input_tokens",
                  "num_decodes", "num_prefills", "num_tokens_ctx",
                  "padded_num_tokens", "pad_size"):
            print(f"[attn_meta] {k}: DP{dp_a}={ma.get(k)}  DP{dp_b}={mb.get(k)}")
        qa, qb = ma.get("query_start_loc"), mb.get("query_start_loc")
        if qa is not None and qb is not None:
            print(f"[attn_meta] query_start_loc: DP{dp_a}={qa.tolist()}")
            print(f"[attn_meta] query_start_loc: DP{dp_b}={qb.tolist()}")
    elif ma is not None or mb is not None:
        print("(attn_meta.pt only on one side - re-sync dsa_v1.py)")

    # --- TP all_gather internals (register_custom_ops) ---
    meta_a = load(os.path.join(dirs_a[0], "tp_gather_meta.pt"))
    meta_b = load(os.path.join(dirs_b[0], "tp_gather_meta.pt"))
    if meta_a is not None and meta_b is not None:
        print(f"[tp_gather_meta] per-rank_chunk/pad_size: DP{dp_a}={meta_a.tolist()} "
              f"DP{dp_b}={meta_b.tolist()}")
    ra = load(os.path.join(dirs_a[0], "tp_group_ranks.pt"))
    rb = load(os.path.join(dirs_b[0], "tp_group_ranks.pt"))
    wa = load(os.path.join(dirs_a[0], "tp_group_world_rank.pt"))
    wb = load(os.path.join(dirs_b[0], "tp_group_world_rank.pt"))
    if ra is not None and rb is not None:
        print(f"[tp_group_ranks] DP{dp_a} world_rank={None if wa is None else wa.item()} "
              f"tp_group={ra.tolist()}  |  DP{dp_b} world_rank={None if wb is None else wb.item()} "
              f"tp_group={rb.tolist()}")
    tg_a = load(os.path.join(dirs_a[0], "tp_gather_in.pt"))
    tg_b = load(os.path.join(dirs_b[0], "tp_gather_in.pt"))
    if tg_a is not None and tg_b is not None:
        n = min(tg_a.shape[0], tg_b.shape[0])
        d = (tg_a[:n].float() - tg_b[:n].float()).abs().max().item()
        print(f"[tp_gather_in (per-rank)] maxdiff={d:.6e}  shape={tuple(tg_a.shape)} vs {tuple(tg_b.shape)}")

    # --- NEW: all_gather OUTPUT (tp_gather_out = all_gather + unpad, no slice) ---
    # 1) Cross-DP: DP0's all_gather result vs DP1's all_gather result. If they
    #    differ, the all_gather (or unpad) reconstructs the sequence differently
    #    for the two DPs => the bug is inside maybe_all_gather_and_maybe_unpad.
    # 2) Within-DP: all_gather result vs the concat-reconstruction from
    #    layer_attn_in ranks (known good). If they differ, the all_gather scrambles
    #    the rank order vs the plain torch.cat used for layer_attn_in.
    to_a = load(os.path.join(dirs_a[0], "tp_gather_out.pt"))
    to_b = load(os.path.join(dirs_b[0], "tp_gather_out.pt"))
    if to_a is not None and to_b is not None:
        n = min(to_a.shape[0], to_b.shape[0])
        d = (to_a[:n].float() - to_b[:n].float()).abs().max().item()
        nbad = int(((to_a[:n].float() - to_b[:n].float()).abs().max(-1).values > 1e-3).sum().item())
        print(f"[tp_gather_out (all_gather+unpad)] DP{dp_a} vs DP{dp_b}: maxdiff={d:.6e} "
              f"bad_positions={nbad}/{n}  shape={tuple(to_a.shape)} vs {tuple(to_b.shape)}")
        # Within-DP cross-check vs layer_attn_in concat reconstruction.
        for dp, dirs, to, tg in ((dp_a, dirs_a, to_a, tg_a), (dp_b, dirs_b, to_b, tg_b)):
            ranks = []
            for d in dirs:
                h = load(os.path.join(d, "layer_attn_in.pt"))
                if h is None:
                    break
                ranks.append(_to_tokens(h))
            if ranks and to is not None:
                concat_full = torch.cat(ranks, dim=0).float()
                m = min(concat_full.shape[0], to.shape[0])
                d2 = (concat_full[:m] - to[:m].float()).abs().max().item()
                print(f"[tp_gather_out vs concat(layer_attn_in)] DP{dp}: maxdiff={d2:.6e}  "
                      f"n={m}  (0 => all_gather restores same order as cat)")
    elif tg_a is not None and tg_b is not None:
        print("(tp_gather_out.pt not found on node - re-sync register_custom_ops.py)")

    # --- Q projection isolation (all-gather -> wq_a -> q_norm -> wq_b) ---
    # attn_hidden_in: the all-gathered full prefill hidden (wq_a input).
    hg_a = load(os.path.join(dirs_a[0], "attn_hidden_in.pt"))
    hg_b = load(os.path.join(dirs_b[0], "attn_hidden_in.pt"))
    if hg_a is not None and hg_b is not None:
        n = min(hg_a.shape[0], hg_b.shape[0])
        fa, fb = hg_a[:n].float(), hg_b[:n].float()
        per_tok = (fa - fb).abs().max(dim=-1).values
        d = per_tok.max().item()
        nbad = int((per_tok > 1e-3).sum().item())
        print(f"[attn_hidden_in (all_gather->wq_a input)] maxdiff={d:.6e}  tokens={n}  "
              f"bad_positions={nbad}  shape={tuple(hg_a.shape)} vs {tuple(hg_b.shape)}")
        # show the positions with the largest diffs (pattern: end vs interleaved)
        if nbad > 0:
            top = torch.topk(per_tok, k=min(12, nbad)).indices.sort().values.tolist()
            print(f"    largest-diff token positions: {top}")
    # qr is wq_b's input (replicated wq_a/q_norm): rank-0 token-overlap compare.
    qr_a = load(os.path.join(dirs_a[0], "attn_qr.pt"))
    qr_b = load(os.path.join(dirs_b[0], "attn_qr.pt"))
    if qr_a is not None and qr_b is not None:
        n = min(qr_a.shape[0], qr_b.shape[0])
        d = (qr_a[:n].float() - qr_b[:n].float()).abs().max().item()
        print(f"[op attn_qr (wq_b input)] maxdiff={d:.6e}  tokens={n}  "
              f"shape={tuple(qr_a.shape)} vs {tuple(qr_b.shape)}")

    # RAW wq_b output (before q_rms + rotary) - isolates the matmul vs q_rms.
    qo_a = [load(os.path.join(d, "attn_wq_b_out.pt")) for d in dirs_a]
    qo_b = [load(os.path.join(d, "attn_wq_b_out.pt")) for d in dirs_b]
    if all(x is not None for x in qo_a + qo_b):
        try:
            A = head_aligned(qo_a)
            B = head_aligned(qo_b)
            n = min(A.shape[0], B.shape[0])
            d = (A[:n] - B[:n]).abs().max().item()
            print(f"[op attn_wq_b_out (pre q_rms/rotary)] maxdiff={d:.6e}  heads={A.shape[1]}  tokens={n}")
        except Exception as e:
            print(f"[op attn_wq_b_out] compare failed: {e!r}")
    else:
        print(f"[op attn_wq_b_out] missing: DP{dp_a}={[x is not None for x in qo_a]} "
              f"DP{dp_b}={[x is not None for x in qo_b]} (re-sync dsa_v1.py)")

    # per-token scale of qr (from npu_rms_norm_dynamic_quant).
    qs_a = load(os.path.join(dirs_a[0], "attn_qr_scale.pt"))
    qs_b = load(os.path.join(dirs_b[0], "attn_qr_scale.pt"))
    if qs_a is not None and qs_b is not None:
        n = min(qs_a.numel(), qs_b.numel())
        d = (qs_a.reshape(-1)[:n].float() - qs_b.reshape(-1)[:n].float()).abs().max().item()
        print(f"[op attn_qr_scale] maxdiff={d:.6e}  shape={tuple(qs_a.shape)} vs {tuple(qs_b.shape)}")
    else:
        print(f"[op attn_qr_scale] missing: DP{dp_a}={qs_a is not None} DP{dp_b}={qs_b is not None}")
    # wq_b weights themselves: concat along dim 0 (the output-head dim).
    # Robust version: report each rank's shape and never crash on asymmetric
    # per-rank shapes (the 16384-vs-8192 crash).
    for name in ("attn_wq_b_weight", "attn_wq_b_scale"):
        ranks_a = [load(os.path.join(d, f"{name}.pt")) for d in dirs_a]
        ranks_b = [load(os.path.join(d, f"{name}.pt")) for d in dirs_b]
        print(f"[op {name}] DP{dp_a} rank shapes: "
              f"{[None if x is None else tuple(x.shape) for x in ranks_a]}")
        print(f"[op {name}] DP{dp_b} rank shapes: "
              f"{[None if x is None else tuple(x.shape) for x in ranks_b]}")
        if all(x is not None for x in ranks_a + ranks_b):
            try:
                A = torch.cat([x.float() for x in ranks_a], dim=0)
                B = torch.cat([x.float() for x in ranks_b], dim=0)
            except RuntimeError as e:
                print(f"[op {name}] CONCAT FAILED: {e}")
                # Fall back to head-aligned comparison: DP0 rank0 (bigger head
                # count) must equal the concatenation of the corresponding DP1
                # ranks (e.g. 32 heads vs 16+16). Assumes weights are
                # (input, output); compare along the last dim.
                try:
                    # DP0 rank0 vs DP1 ranks 0..k such that output sizes match.
                    if name.endswith("weight"):
                        dim = 1  # (input, output)
                    else:
                        dim = 0  # (output,)
                    sa = ranks_a[0].shape[dim] if ranks_a[0].dim() > dim else ranks_a[0].shape[0]
                    acc, k, prev = [], 0, None
                    for r in ranks_b:
                        rb = r.shape[dim] if r.dim() > dim else r.shape[0]
                        if prev is not None and k + rb > sa:
                            break
                        acc.append(r); k += rb
                        if k == sa:
                            break
                    if sum(x.shape[dim] for x in acc) == sa:
                        B0 = torch.cat([x.float() for x in acc], dim=dim)
                        A0 = ranks_a[0].float()
                        d = (A0 - B0).abs().max().item()
                        print(f"[op {name}] head-aligned DP{dp_a} rank0 vs DP{dp_b} r0..{len(acc)-1}: "
                              f"maxdiff={d:.6e}")
                    else:
                        print(f"[op {name}] head-aligned sizes don't match: {sa} vs {k}")
                except Exception as e2:
                    print(f"[op {name}] head-aligned compare failed: {e2!r}")
                continue
            if A.shape == B.shape:
                d = (A - B).abs().max().item()
                print(f"[op {name}] maxdiff={d:.6e}  shape={tuple(A.shape)}")
            else:
                print(f"[op {name}] SHAPE MISMATCH {tuple(A.shape)} vs {tuple(B.shape)}")

    # --- per-layer: full-sequence reconstruction per DP ---
    n_layers = 0
    for d in dirs_a:
        n_layers = max(n_layers, len(glob.glob(os.path.join(d, "layer*.pt"))))
    first_div = None
    for i in range(n_layers):
        full_a, full_b = [], []
        for d in dirs_a:
            h = load(os.path.join(d, f"layer{i}.pt"))
            if h is None:
                break
            full_a.append(_to_tokens(h))
        for d in dirs_b:
            h = load(os.path.join(d, f"layer{i}.pt"))
            if h is None:
                break
            full_b.append(_to_tokens(h))
        if not full_a or not full_b:
            continue
        A = torch.cat(full_a, dim=0)
        B = torch.cat(full_b, dim=0)
        n = min(A.shape[0], B.shape[0])
        nan_a = bool(A.isnan().any().item())
        nan_b = bool(B.isnan().any().item())
        d = (A[:n] - B[:n]).abs().max().item()
        mark = ""
        if d > args.tol and first_div is None:
            first_div = i
            mark = "   <== FIRST DIVERGENCE"
        print(f"layer{i}: maxdiff={d:.6e}  nan_dp{dp_a}={nan_a} nan_dp{dp_b}={nan_b} "
              f"tokens={A.shape[0]} vs {B.shape[0]}{mark}")
    if first_div is None:
        print(f"\nNo divergence in layers 0..{n_layers - 1} (maxdiff <= {args.tol}).")
    else:
        print(f"\nFirst divergence at layer {first_div}. Inspect that layer's attention/MoE.")


if __name__ == "__main__":
    main()
