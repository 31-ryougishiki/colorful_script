#!/usr/bin/env python3
"""Analyze the DIAGSTEP dual-timing (host vs device) log lines produced by
NPUModelRunner._cs_step_end (model_runner_v1.py) and the DIAG triton_q_rms
lines from ops/triton/rms_norm.py.

One run answers: for each occasional spike step, is the time
  (a) device-bound inside model_forward / q_rms_rope / moe / update_cos_sin,
  (b) host-side triton JIT recompile (CPU, device idle during it), or
  (c) host blocked on a device backlog that was enqueued *before* model_forward
      (this is the case where adding torch.npu.synchronize() before
      model_forward moves the spike onto the sync)?

Usage:
    python analyze_diagstep.py dsv4.log [--min-ms 500] [--worker DP0] [--all]
"""

import argparse
import re
import statistics
import sys

# e.g.  (Worker_DP0_EP0 pid=2188) INFO 08-07 06:41:25 [model_runner_v1.py:2410] <content>
_LINE_PREFIX = re.compile(r"^(\(.*?\))?\s*\S+\s+.*?\[(model_runner_v1|rms_norm)\.py:\d+\]\s")

# DIAGSTEP 12: tok=16171 pad=16384 prep_h=3.2 prep_d=40.1 cos_h=0.3 cos_d=2100.0
#   fwd_h=2800.5 fwd_d=2890.0 post_h=1.2 post_d=8.0 drain=50.0ms
#   layer0_h=2200.1 layer0_d=2215.6 qrms_h=2150.2 qrms_d=2152.3 moe_h=4.1 moe_d=4.5
#   memR=40960MiB dMemR=+0MiB memA=20480MiB
_DIAGSTEP = re.compile(
    r"DIAGSTEP (\d+): tok=(\d+) pad=(\d+) "
    r"prep_h=([\d.]+) prep_d=([\d.]+) cos_h=([\d.]+) cos_d=([\d.]+) "
    r"fwd_h=([\d.]+) fwd_d=([\d.]+) post_h=([\d.]+) post_d=([\d.]+) "
    r"drain=([\d.]+)ms "
    r"layer0_h=([\d.]+) layer0_d=([\d.]+) qrms_h=([\d.]+) qrms_d=([\d.]+) "
    r"moe_h=([\d.]+) moe_d=([\d.]+) memR=([\d.-]+)MiB dMemR=([\d.+-]+)MiB memA=([\d.]+)MiB"
)

# DIAG triton_q_rms new=True key=(total_batch=2097152,dim=512,BLOCK_M=16,vc=48)
#   setup=0.2ms alloc=1.1ms launch=2130.0ms
_TRITON = re.compile(
    r"DIAG triton_q_rms new=(\w+) key=\(total_batch=(\d+),dim=(\d+),BLOCK_M=(\d+),vc=(\d+)\) "
    r"setup=([\d.]+)ms alloc=([\d.]+)ms launch=([\d.]+)ms"
)

_KEYS = ["prep_h", "prep_d", "cos_h", "cos_d", "fwd_h", "fwd_d",
         "post_h", "post_d", "drain", "layer0_h", "layer0_d", "qrms_h",
         "qrms_d", "moe_h", "moe_d", "memR", "dMemR", "memA"]


# Worker tag from the console prefix, e.g. "Worker_DP0_EP0 pid=2188)".
_WORKER = re.compile(r"^\((\w+)\s")


def strip_prefix(line: str) -> str:
    m = _LINE_PREFIX.match(line)
    return line[m.end():] if m else line


def worker_of(line: str) -> str:
    m = _WORKER.match(line.strip())
    return m.group(1) if m else "?"


def parse_diagstep(line: str) -> dict | None:
    m = _DIAGSTEP.search(strip_prefix(line))
    if not m:
        return None
    g = m.groups()
    d = {"step": int(g[0]), "tok": int(g[1]), "pad": int(g[2])}
    for i, k in enumerate(_KEYS):
        d[k] = float(g[3 + i])
    return d


def parse_triton(line: str) -> dict | None:
    m = _TRITON.search(strip_prefix(line))
    if not m:
        return None
    g = m.groups()
    return {
        "new": g[0] == "True",
        "total_batch": int(g[1]), "dim": int(g[2]),
        "BLOCK_M": int(g[3]), "vc": int(g[4]),
        "setup": float(g[5]), "alloc": float(g[6]), "launch": float(g[7]),
    }


def classify(d: dict, min_ms: float) -> str:
    """Return a one-line mechanism verdict for a spike/compile step."""
    spike = max(d["fwd_d"], d["fwd_h"], d["cos_d"], d["drain"])
    # triton JIT compile candidate: host spent >500ms inside q_rms_rope.
    # Checked before the spike gate so a compile on an otherwise-normal step
    # still gets a verdict.
    if d["qrms_h"] >= 500.0:
        tag = (f"TRITON JIT RECOMPILE in q_rms_rope (host={d['qrms_h']:.0f}ms "
               f"dev={d['qrms_d']:.0f}ms drain={d['drain']:.0f}ms)")
        if d["drain"] < 500.0:
            tag += (" | device idle during compile -> pure CPU, a sync before "
                    "model_forward will NOT remove it; fix = rms_norm.py "
                    "TOTAL_BATCH->runtime or stable pad buckets")
        else:
            tag += (" | device backlog draining inside launch -> a sync before "
                    "model_forward absorbs part of it")
        return tag
    if spike < min_ms:
        return ""
    # cos region device-bound -> gather/op in update_cos_sin
    if d["cos_d"] >= min_ms:
        return (f"DEVICE-BOUND in update_cos_sin (cos_d={d['cos_d']:.0f}ms) "
                f"-> ops/rotary_embedding.py index_select chain; fwd inherits "
                f"fwd_h={d['fwd_h']:.0f}ms while device busy")
    # model_forward region
    if d["fwd_d"] >= min_ms or d["qrms_d"] >= min_ms:
        if d["moe_d"] >= min_ms:
            return (f"DEVICE-BOUND in MoE branch (moe_d={d['moe_d']:.0f}ms "
                    f"host={d['moe_h']:.0f}ms)")
        return f"DEVICE-BOUND in model_forward (fwd_d={d['fwd_d']:.0f}ms)"
    if d["fwd_h"] >= min_ms:
        if d["drain"] >= min_ms:
            return (f"HOST BLOCKED ON DEVICE BACKLOG (fwd_h={d['fwd_h']:.0f}ms, "
                    f"drain={d['drain']:.0f}ms) - work enqueued earlier drained "
                    f"at step_end; matches 'sync before model_forward absorbs it'")
        return (f"HOST-SIDE CPU work in model_forward (fwd_h={d['fwd_h']:.0f}ms, "
                f"dev={d['fwd_d']:.0f}ms, drain small) - triton compile / alloc / python")
    if d["drain"] >= min_ms:
        return (f"DEVICE backlog at step end (drain={d['drain']:.0f}ms) with "
                f"small per-region dev - side-stream work or queue backlog")
    return f"unclassified spike (spike={spike:.0f}ms)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log")
    ap.add_argument("--min-ms", type=float, default=500.0)
    ap.add_argument("--worker", default="", help="filter worker substring, e.g. DP0")
    ap.add_argument("--all", action="store_true", help="print every step, not just spikes")
    args = ap.parse_args()

    # Group by worker so DP shards don't collide on step numbers.
    steps: dict[str, dict[int, dict]] = {}
    tritons: dict[str, list[dict]] = {}
    with open(args.log, encoding="utf-8", errors="replace") as f:
        for line in f:
            if args.worker and args.worker not in line:
                continue
            w = worker_of(line)
            d = parse_diagstep(line)
            if d:
                steps.setdefault(w, {})[d["step"]] = d
                continue
            t = parse_triton(line)
            if t:
                t["worker"] = w
                tritons.setdefault(t["total_batch"], []).append(t)

    if not steps:
        print(f"[warn] no DIAGSTEP lines in {args.log} "
              f"(did the run enable callstack_tracing? worker filter too narrow?)")
        return 1

    for w in sorted(steps):
        ws = steps[w]
        ordered = sorted(ws)
        fwd_dev = [ws[s]["fwd_d"] for s in ordered]
        med = statistics.median(fwd_dev)
        # fwd_d is in thousands of ms; 5x would swallow 2-3s spikes.  Use a
        # tighter multiple + an absolute floor so triton-recompile steps show.
        threshold = max(args.min_ms, med * 1.3)

        print(f"\n=== worker {w} (median fwd_d={med:.1f}ms, "
              f"spike threshold={threshold:.1f}ms) ===")
        hdr = (f"{'step':>4}{'tok':>7}{'pad':>8}"
               f"{'cos_h':>7}{'cos_d':>8}{'fwd_h':>8}{'fwd_d':>8}{'drain':>8}"
               f"{'qrms_h':>8}{'qrms_d':>8}{'moe_d':>7}{'dMemR':>7}")
        print(hdr)
        print("-" * len(hdr))
        for s in ordered:
            d = ws[s]
            spike = max(d["fwd_d"], d["fwd_h"], d["cos_d"], d["drain"]) > threshold
            compile_cand = d["qrms_h"] >= 500.0
            if not spike and not compile_cand and not args.all:
                continue
            flag = ""
            if spike:
                flag += " <== SPIKE"
            if compile_cand:
                flag += " [COMPILE?]"
            print(f"{s:>4}{d['tok']:>7}{d['pad']:>8}"
                  f"{d['cos_h']:>7.1f}{d['cos_d']:>8.1f}{d['fwd_h']:>8.1f}"
                  f"{d['fwd_d']:>8.1f}{d['drain']:>8.1f}"
                  f"{d['qrms_h']:>8.1f}{d['qrms_d']:>8.1f}{d['moe_d']:>7.1f}"
                  f"{d['dMemR']:>+7.0f}{flag}")
            if spike or compile_cand:
                print(f"    -> {classify(d, threshold)}")

    # triton new-shape events with slow launch
    slow = [t for ts in tritons.values() for t in ts if t["launch"] > 50.0]
    if slow:
        print("\n=== triton_q_rms slow launches (launch > 50ms) ===")
        for t in sorted(slow, key=lambda x: -x["launch"]):
            print(f"  {t['worker']:>14} new={t['new']} total_batch={t['total_batch']} "
                  f"dim={t['dim']} BLOCK_M={t['BLOCK_M']} vc={t['vc']} "
                  f"setup={t['setup']:.1f}ms alloc={t['alloc']:.1f}ms "
                  f"launch={t['launch']:.1f}ms")
        slow_launch = [t for t in slow if t["launch"] >= 500.0]
        if slow_launch:
            print(f"  >>> {len(slow_launch)} launch(s) >= 500ms "
                  f"= triton JIT recompile on a new shape confirmed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
