#!/usr/bin/env python3
"""Extract and analyze the per-step CallStack log lines produced by
NPUModelRunner._cs_step_end (model_runner_v1.py) — the compact one-liner,
the "Step N end" timing, and (on layer_0 spike steps) the full span tree.

This is a drop-in replacement for the manual grep command used during the
layer_0 spike diagnosis.  It answers three questions from one log file:
  1. Which steps had a layer_0 spike (layer_0 >> median or > tree_threshold)?
  2. On spike steps, is the 2s inside q_rms_rope, and if so in which of its
     two sub-spans (triton_q_rms vs inplace_partial_rotary_mul)?
  3. The host elapsed (Step N end) vs the sum of NPU-event spans (total).

Usage:
    python extract_callstack_log.py dsv4.log
    python extract_callstack_log.py dsv4.log --min-ms 200      # spike threshold
    python extract_callstack_log.py dsv4.log --worker DP0      # filter worker
    python extract_callstack_log.py dsv4.log --tree 8          # dump step 8's tree
"""

import argparse
import re
import statistics
import sys

# vLLM console line prefix, e.g.:
#   (Worker_DP0_EP0 pid=2188) INFO 08-07 06:41:25 [model_runner_v1.py:2257] <content>
_LINE_PREFIX = re.compile(r"^(\(.*?\))?\s*\S+\s+.*?\[model_runner_v1\.py:\d+\]\s")

# Compact one-liner:
#   CallStack step 12: tok=16171 total=5927.3ms fwd=5890.2ms prep=33.9ms layer0=2215.6ms attn=2172.8ms prefill=2160.6ms q_rms_rope=2152.3ms
_COMPACT = re.compile(
    r"CallStack step (\d+): tok=(\d+) total=([\d.]+)ms fwd=([\d.]+)ms prep=([\d.]+)ms "
    r"layer0=([\d.]+)ms attn=([\d.]+)ms (\w+)=([\d.]+)ms q_rms_rope=([\d.]+)ms"
)

# Step end:
#   Step 12 end: elapsed=5879.760 ms, num_tokens=16171
_STEP_END = re.compile(r"Step (\d+) end: elapsed=([\d.]+) ms, num_tokens=(\d+)")

# Tree block header:  CallStack span timings (step 8):
_TREE_HEADER = re.compile(r"CallStack span timings \(step (\d+)\):")

# Tree node line (from _format_span_summary), short label + duration:
#   │       ├── q_rms_rope               2152.3ms ( 36%)
_NODE = re.compile(r"^(.*?)([\w/._-]+)\s+([\d.]+)ms\s*\(\s*([\d.]+)%\)")

# Compressed suffixes we care about on spike steps.
# Full path (suffix match) -> short label for the report.
_Q_RMS_TRITON_SUF = "mla_prolog/standard/q_rms_rope/triton"
_Q_RMS_ROTARY_SUF = "mla_prolog/standard/q_rms_rope/rotary"


def strip_prefix(line: str) -> str:
    m = _LINE_PREFIX.match(line)
    return line[m.end():] if m else line


def parse_compact(line: str):
    """Return dict for a compact one-liner, or None."""
    m = _COMPACT.search(strip_prefix(line))
    if not m:
        return None
    g = m.groups()
    return {
        "step": int(g[0]),
        "tok": int(g[1]),
        "total": float(g[2]),
        "fwd": float(g[3]),
        "prep": float(g[4]),
        "layer0": float(g[5]),
        "attn": float(g[6]),
        "phase": g[7],
        "phase_ms": float(g[8]),
        "q_rms_rope": float(g[9]),
    }


def parse_step_end(line: str):
    m = _STEP_END.search(strip_prefix(line))
    if not m:
        return None
    return {"step": int(m.group(1)), "elapsed": float(m.group(2)), "tok": int(m.group(3))}


def collect_tree(lines: list[str]) -> list[dict]:
    """Parse the tree lines of one block into a flat list of
    {indent_str, label, ms, pct, indent} nodes."""
    nodes = []
    for raw in lines:
        line = strip_prefix(raw)
        m = _NODE.match(line)
        if not m:
            continue
        indent, label, ms, pct = m.groups()
        nodes.append({"label": label, "ms": float(ms), "pct": float(pct),
                      "indent": indent})
    return nodes


def find_spike_subs(nodes: list[dict]) -> dict:
    """From a spike step's tree, find q_rms_rope and its triton/rotary children.

    We locate the q_rms_rope node by label, then look at the immediately
    following nodes one indent deeper that carry the triton/rotary labels.
    """
    out = {"q_rms_rope": None, "triton": None, "rotary": None, "moe": None}
    for i, n in enumerate(nodes):
        if n["label"] == "q_rms_rope" and out["q_rms_rope"] is None:
            out["q_rms_rope"] = n["ms"]
            depth = n["indent"]
            # children are the following nodes with one more indent level
            for c in nodes[i + 1:]:
                if len(c["indent"]) <= len(depth):
                    break
                if c["label"] in ("triton", "rotary"):
                    out[c["label"]] = c["ms"]
        if n["label"] in ("moe", "quant_apply", "experts") and out["moe"] is None:
            out["moe"] = n["ms"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", help="path to the vLLM console log")
    ap.add_argument("--min-ms", type=float, default=200.0,
                    help="layer_0 threshold (ms) to treat a step as a spike "
                         "(default 200, matches tree_threshold_ms)")
    ap.add_argument("--worker", default="Worker_DP0",
                    help="only consider lines from this worker (default Worker_DP0)")
    ap.add_argument("--tree", type=int, default=None,
                    help="dump the full span tree of a specific step number")
    args = ap.parse_args()

    # ---- pass 1: gather everything -------------------------------------------
    steps: dict[int, dict] = {}       # step -> merged compact/end info
    trees: dict[int, list[dict]] = {}  # step -> parsed tree nodes
    cur_step: int | None = None
    cur_lines: list[str] = []

    with open(args.log, encoding="utf-8", errors="replace") as f:
        for line in f:
            if args.worker not in line:
                continue

            sc = parse_compact(line)
            if sc:
                steps.setdefault(sc["step"], {}).update(sc)
                continue

            se = parse_step_end(line)
            if se:
                steps.setdefault(se["step"], {}).update(se)
                continue

            th = _TREE_HEADER.search(strip_prefix(line))
            if th:
                # flush previous block
                if cur_step is not None and cur_lines:
                    trees[cur_step] = collect_tree(cur_lines)
                cur_step = int(th.group(1))
                cur_lines = []
                continue
            if cur_step is not None:
                cur_lines.append(line)

    if cur_step is not None and cur_lines:
        trees[cur_step] = collect_tree(cur_lines)

    # ---- pass 2: summary table ------------------------------------------------
    if not steps:
        print(f"[warn] no CallStack step lines found for worker '{args.worker}' in {args.log}")
        return 1

    ordered = sorted(steps)
    layer0s = [steps[s].get("layer0", 0.0) for s in ordered]
    # Baseline median should exclude genuine spikes, else the median itself is
    # inflated by the anomalies. Two-pass: first find spikes vs the crude
    # min_ms floor, then recompute the median over the non-spike steps.
    # Baseline median over non-spike samples only; if too few, fall back to
    # the explicit --min-ms floor so we don't mask spikes with their own median.
    crude = [v for v in layer0s if v <= args.min_ms]
    med = statistics.median(crude) if len(crude) >= 3 else 0.0
    threshold = max(args.min_ms, med * 5.0)

    hdr = (f"{'step':>5}{'tok':>7}{'total':>9}{'elapsed':>9}{'fwd':>9}{'prep':>9}"
           f"{'layer0':>10}{'attn':>8}{'q_rms':>9}{'phase':>7}")
    print(hdr)
    print("-" * len(hdr))
    for s in ordered:
        st = steps[s]
        ms = lambda k: f"{st.get(k, 0.0):8.1f}"  # noqa: E731
        spike = " <== SPIKE" if st.get("layer0", 0.0) > threshold else ""
        print(f"{s:>5}{st.get('tok','?'):>7}"
              f"{ms('total')}{ms('elapsed')}{ms('fwd')}{ms('prep')}"
              f"{ms('layer0')}{ms('attn')}{ms('q_rms_rope')}"
              f"{st.get('phase','-'):>7}{spike}")

    print(f"\nmedian layer_0 = {med:.1f} ms | spike threshold = {threshold:.1f} ms "
          f"| {sum(1 for s in ordered if steps[s].get('layer0', 0) > threshold)} spikes")

    # ---- pass 3: spike detail (q_rms_rope triton vs rotary) -------------------
    spikes = [s for s in ordered if steps[s].get("layer0", 0.0) > threshold]
    if spikes:
        print("\n=== spike steps: where the time went ===")
        for s in spikes:
            st = steps[s]
            nodes = trees.get(s, [])
            sub = find_spike_subs(nodes) if nodes else {}
            l0 = st.get("layer0", 0.0)
            qr = st.get("q_rms_rope", 0.0)
            print(f"step {s}: layer0={l0:.1f}ms q_rms_rope={qr:.1f}ms")
            if sub.get("q_rms_rope") is not None:
                tr = sub.get("triton")
                rt = sub.get("rotary")
                print(f"    q_rms_rope(tree)={sub['q_rms_rope']:.1f}ms "
                      f"triton={tr if tr is not None else 0.0:.1f}ms "
                      f"rotary={rt if rt is not None else 0.0:.1f}ms")
                if tr is not None and rt is not None:
                    gap = sub["q_rms_rope"] - tr - rt
                    print(f"    -> unaccounted (host-side wait between them): {gap:.1f}ms")
            else:
                print("    (no q_rms_rope sub-spans in tree; spike may be elsewhere)")
                for label in ("moe", "quant_apply"):
                    if sub.get(label) is not None:
                        print(f"    hint: {label}={sub[label]:.1f}ms")

    # ---- optional: dump a specific tree ---------------------------------------
    if args.tree is not None:
        nodes = trees.get(args.tree)
        if nodes is None:
            print(f"\n[tree] no tree for step {args.tree}")
        else:
            print(f"\n=== full tree for step {args.tree} ===")
            for n in nodes:
                print(f"{n['indent']}{n['label']:<28s} {n['ms']:>8.1f}ms ({n['pct']:>4.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
