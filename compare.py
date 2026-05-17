#!/usr/bin/env python3
"""
compare.py — Compare intermediate tensor dumps between ds4 (C) and ds4f (Rust)
to locate the first layer / operation where the two implementations diverge.

QUICK START
-----------
1. Generate C dumps (DS4_METAL_GRAPH_DUMP_PREFIX):
   DS4_METAL_GRAPH_DUMP_PREFIX=/tmp/c DS4_METAL_GRAPH_DUMP_LAYER=all \\
     ./ds4 --metal --nothink -sys "" -n 1 -p "Hi"

2. Generate Rust dumps (DS4_DUMP_PREFIX):
   DS4_DUMP_PREFIX=/tmp/rs \\
     ./ds4-rs/target/release/ds4f run --nothink --system "" -n 1 -p "Hi"

3. Compare:
   python3 compare.py /tmp/c /tmp/rs --pos 0

The script prints a table of max-absolute-error per (layer, tensor), stops at
the first divergence (unless --all is given), and exits 1 if any mismatch is
found.

TENSOR NAME ALIASES
-------------------
C and Rust use slightly different names for a few tensors:
  C name                       Rust name
  ─────────────────────────── ─────────────────────────────
  ffn_moe_gate_clamped        ffn_moe_gate
  ffn_moe_up_clamped          ffn_moe_up
  ffn_moe_weighted_swiglu     ffn_moe_swiglu
  ffn_moe_topk  (.i32)        ffn_moe_topk  (.i32)

ENVIRONMENT VARIABLES (run mode, --run flag)
--------------------------------------------
  DS4_BIN   path to ds4 C binary   (default: ./ds4)
  DS4F_BIN  path to ds4f binary    (default: ./ds4-rs/target/release/ds4f)
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: pip install numpy")

# ── Constants ──────────────────────────────────────────────────────────────

N_LAYER = 43

# Ordered list of tensors within each layer (computation order).
# Each entry is either:
#   "name"                    - same name in both C and Rust
#   ("c_name", "rs_name")     - different names but same semantic content
#   ("c_name", "rs_name", "i32")  - integer tensor (raw i32 file)
LAYER_TENSORS_ORDERED = [
    "hc_attn_pre",           # attn_cur after HC weighted sum  (N_EMBD)
    "attn_norm",             # after attention RMSNorm         (N_EMBD)
    "q_lora",                # Q LoRA compression              (N_LORA_Q)
    "q_lora_norm",           # Q LoRA RMSNorm                  (N_LORA_Q)
    "KVraw",                 # KV compression                  (N_HEAD_DIM)
    "KVnorm",                # KV RMSNorm                      (N_HEAD_DIM)
    "Qraw",                  # Q before head RMSNorm           (N_HEAD*N_HEAD_DIM)
    "Qnorm",                 # Q after head RMSNorm
    "Qcur",                  # Q after RoPE
    "KVrope",                # KV after RoPE
    "KVcur",                 # KV after FP8 store/reload
    "kqv_out",               # raw attention output            (N_HEAD*N_HEAD_DIM)
    "kqv_back",              # after inverse RoPE
    "attn_out",              # after O LoRA projection         (N_EMBD)
    "hc_attn_post",          # after HC expand + residual      (N_HC*N_EMBD)
    "hc_ffn_pre",            # ffn_cur after HC weighted sum   (N_EMBD)
    "ffn_norm",              # after FFN RMSNorm               (N_EMBD)
    "ffn_moe_logits",        # router logits                   (N_EXPERT)
    "ffn_moe_probs",         # router probs (softplus_sqrt)    (N_EXPERT)
    ("ffn_moe_topk", "ffn_moe_topk", "i32"),   # selected expert IDs (N_EXPERT_USED i32s)
    "ffn_moe_weights_scaled",# normalised router weights       (N_EXPERT_USED)
    # gate/up: C dumps AFTER SwiGLU clamping; Rust dumps BEFORE → not directly comparable.
    # We skip them and compare the SwiGLU output instead.
    ("ffn_moe_weighted_swiglu", "ffn_moe_swiglu"),  # weighted SwiGLU mid (N_FF_EXP*N_EXPERT_USED)
    "ffn_moe_down",          # down projections, all experts   (N_EMBD*N_EXPERT_USED)
    "ffn_moe_out",           # accumulated routed expert out   (N_EMBD)
    "ffn_shexp",             # shared expert output            (N_EMBD)
    "hc_ffn_post",           # after HC expand + residual      (N_HC*N_EMBD)
]

# Output-head tensors (dumped at il=N_LAYER, pos=0 by convention)
OUTPUT_TENSORS_ORDERED = [
    "result_hc_pre",         # HC pre-weights (N_HC)
    "result_hc_weights",     # HC weights after normalisation (N_HC)
    "result_hc",             # HC weighted sum → output embedding (N_EMBD)
    "result_norm",           # output RMSNorm (N_EMBD)
    "result_output",         # final logits (N_VOCAB)
]

ANSI_RED    = "\033[31m"
ANSI_GREEN  = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_RESET  = "\033[0m"


# ── File helpers ───────────────────────────────────────────────────────────

def bin_path(prefix: str, name: str, layer: int, pos: int) -> Path:
    return Path(f"{prefix}_{name}-{layer}_pos{pos}.bin")

def i32_path(prefix: str, name: str, layer: int, pos: int) -> Path:
    return Path(f"{prefix}_{name}-{layer}_pos{pos}.i32")


def load_f32(path: Path) -> "np.ndarray | None":
    if not path.exists():
        return None
    return np.frombuffer(path.read_bytes(), dtype=np.float32).copy()


def load_i32(path: Path) -> "np.ndarray | None":
    if not path.exists():
        return None
    return np.frombuffer(path.read_bytes(), dtype=np.int32).copy()


def _slice_batch(arr: "np.ndarray", pos: int, row_len: int) -> "np.ndarray | None":
    """Extract row `pos` from a batch-dump array (n_tokens * row_len elements)."""
    if row_len <= 0 or len(arr) % row_len != 0:
        return None
    n_batch = len(arr) // row_len
    if pos >= n_batch:
        return None
    return arr[pos * row_len : (pos + 1) * row_len]


def load_c_sliced_f32(c_prefix: str, name: str, il: int, pos: int,
                       rs_arr: "np.ndarray | None") -> "np.ndarray | None":
    """Load C f32 tensor at layer il for token position pos.

    C batch-prefill dumps all tokens concatenated in a single pos0 file.
    If the loaded array is larger than rs_arr by an integer multiple, extract
    row `pos` from the batch.  Also falls back to the pos0 batch file when
    there is no file for the requested pos directly.
    """
    rs_len = len(rs_arr) if rs_arr is not None else 0

    # Try the exact-pos file first.
    c_arr = load_f32(bin_path(c_prefix, name, il, pos))
    if c_arr is not None:
        if rs_len > 0 and len(c_arr) > rs_len and len(c_arr) % rs_len == 0:
            # Batch file: C tagged it with pos=0 but it holds all tokens.
            sliced = _slice_batch(c_arr, pos, rs_len)
            return sliced if sliced is not None else c_arr
        return c_arr

    # Fall back to the pos0 batch file (the common case for prefill).
    if pos != 0:
        c_batch = load_f32(bin_path(c_prefix, name, il, 0))
        if c_batch is not None and rs_len > 0:
            return _slice_batch(c_batch, pos, rs_len)

    return None


def load_c_sliced_i32(c_prefix: str, name: str, il: int, pos: int,
                       rs_arr: "np.ndarray | None") -> "np.ndarray | None":
    """Same as load_c_sliced_f32 but for i32 tensors."""
    rs_len = len(rs_arr) if rs_arr is not None else 0

    c_arr = load_i32(i32_path(c_prefix, name, il, pos))
    if c_arr is not None:
        if rs_len > 0 and len(c_arr) > rs_len and len(c_arr) % rs_len == 0:
            sliced = _slice_batch(c_arr, pos, rs_len)
            return sliced if sliced is not None else c_arr
        return c_arr

    if pos != 0:
        c_batch = load_i32(i32_path(c_prefix, name, il, 0))
        if c_batch is not None and rs_len > 0:
            return _slice_batch(c_batch, pos, rs_len)

    return None


def discover_positions(prefix: str, n_layers: int = N_LAYER) -> list:
    """Scan dump files to find all positions that have data."""
    pat = re.compile(r"_([A-Za-z_]+)-(\d+)_pos(\d+)\.(bin|i32)$")
    positions = set()
    dir_ = Path(prefix).parent
    stem = Path(prefix).name
    try:
        for f in dir_.iterdir():
            if f.name.startswith(stem + "_"):
                m = pat.search(f.name)
                if m:
                    positions.add(int(m.group(3)))
    except (FileNotFoundError, PermissionError):
        pass
    return sorted(positions)


# ── Comparison helpers ─────────────────────────────────────────────────────

def compare_arrays(c_arr, rs_arr, tol: float, label: str, verbose: bool) -> bool:
    """Return True if the arrays match within tolerance, False on mismatch.
    Prints a summary line in both cases."""
    if c_arr is None or rs_arr is None:
        if verbose:
            which = "C" if c_arr is None else "Rust"
            print(f"  SKIP  {label}  (missing from {which})")
        return True  # not a failure, just missing

    if c_arr.shape != rs_arr.shape:
        # Allow comparison if one is a flattened version of the other
        min_len = min(len(c_arr.flat), len(rs_arr.flat))
        c_flat = c_arr.flat[:min_len]
        rs_flat = rs_arr.flat[:min_len]
        print(f"  WARN  {label}  shapes differ: C={c_arr.shape} Rust={rs_arr.shape}")
    else:
        c_flat = c_arr
        rs_flat = rs_arr

    diff = np.abs(c_flat - rs_flat)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    max_idx = int(diff.argmax())

    ok = max_diff <= tol
    status = (ANSI_GREEN + "  OK  " + ANSI_RESET) if ok else (ANSI_RED + " FAIL " + ANSI_RESET)
    print(f"{status} {label}  max_abs={max_diff:.3e}  mean={mean_diff:.3e}"
          f"  [idx={max_idx} c={float(c_flat.flat[max_idx]):.5g}"
          f" rs={float(rs_flat.flat[max_idx]):.5g}]")
    return ok


# ── Core comparison ────────────────────────────────────────────────────────

def compare_dumps(
    c_prefix: str, rs_prefix: str, pos: int,
    tol: float, stop_on_first: bool, verbose: bool,
    n_layers: int,
) -> bool:
    """
    Compare all layer tensors and output-head tensors.
    Returns True if everything matches within tol, False otherwise.
    """
    any_mismatch = False

    for il in range(n_layers):
        layer_header_printed = False

        for entry in LAYER_TENSORS_ORDERED:
            # Unpack entry
            if isinstance(entry, str):
                c_name, rs_name, dtype = entry, entry, "f32"
            elif len(entry) == 2:
                c_name, rs_name, dtype = entry[0], entry[1], "f32"
            else:
                c_name, rs_name, dtype = entry[0], entry[1], entry[2]

            label = f"layer={il:2d}  {c_name}" + (f" / {rs_name}" if c_name != rs_name else "")

            if dtype == "i32":
                rs_arr = load_i32(i32_path(rs_prefix, rs_name, il, pos))
                c_arr  = load_c_sliced_i32(c_prefix, c_name, il, pos, rs_arr)
                if c_arr is None or rs_arr is None:
                    if verbose:
                        which = "C" if c_arr is None else "Rust"
                        print(f"  SKIP  {label}  (missing from {which})")
                    continue
                matches = np.array_equal(c_arr, rs_arr)
                if not matches or verbose:
                    if not layer_header_printed:
                        print(f"\n── Layer {il} ──")
                        layer_header_printed = True
                    status = (ANSI_GREEN + "  OK  " + ANSI_RESET) if matches \
                             else (ANSI_RED + " FAIL " + ANSI_RESET)
                    print(f"{status} {label}  [i32 expert IDs]"
                          f"  C={c_arr.tolist()}  Rust={rs_arr.tolist()}")
                if not matches:
                    any_mismatch = True
                    if stop_on_first:
                        print(f"\n{ANSI_RED}First mismatch at layer {il}, tensor '{c_name}'{ANSI_RESET}")
                        return False
            else:
                rs_arr = load_f32(bin_path(rs_prefix, rs_name, il, pos))
                c_arr  = load_c_sliced_f32(c_prefix, c_name, il, pos, rs_arr)
                if c_arr is None and rs_arr is None:
                    continue
                if not layer_header_printed and (c_arr is not None or rs_arr is not None):
                    print(f"\n── Layer {il} ──")
                    layer_header_printed = True
                ok = compare_arrays(c_arr, rs_arr, tol, label, verbose)
                if not ok:
                    any_mismatch = True
                    if stop_on_first:
                        print(f"\n{ANSI_RED}First mismatch at layer {il}, tensor '{c_name}'{ANSI_RESET}")
                        return False

    # ── Output head ──
    print(f"\n── Output head (il={n_layers}) ──")
    # C batch prefill stores output head at pos=0 (always the last token's logits).
    # Rust stores output head at every position.  The comparison position for Rust
    # is the user-specified --pos argument.
    #
    # For a 5-token prompt: --pos 4  → C pos=0 (last batch token) vs Rust pos=4 (last prefill token)  ✓
    # For decode:           --pos 5  → C pos=0 (prefill logits) vs Rust pos=5 (decode logits)        ✓
    c_out_pos = 0
    rs_out_pos = pos if bin_path(rs_prefix, OUTPUT_TENSORS_ORDERED[0], n_layers, pos).exists() else 0

    any_found = any(
        bin_path(c_prefix, t, n_layers, c_out_pos).exists() or
        bin_path(rs_prefix, t, n_layers, rs_out_pos).exists()
        for t in OUTPUT_TENSORS_ORDERED
    )
    if any_found:
        for t in OUTPUT_TENSORS_ORDERED:
            c_arr = load_f32(bin_path(c_prefix, t, n_layers, c_out_pos))
            rs_arr = load_f32(bin_path(rs_prefix, t, n_layers, rs_out_pos))
            if c_arr is None and rs_arr is None:
                continue
            label = f"output   {t}  [C pos={c_out_pos} Rust pos={rs_out_pos}]"
            ok = compare_arrays(c_arr, rs_arr, tol, label, verbose)
            if not ok:
                any_mismatch = True
                if stop_on_first:
                    print(f"\n{ANSI_RED}First mismatch in output head, tensor '{t}'{ANSI_RESET}")
                    return False
    else:
        print("  (no output-head dumps found — run with DS4_DUMP_PREFIX / DS4_METAL_GRAPH_DUMP_PREFIX set)")

    return not any_mismatch


# ── Run mode ───────────────────────────────────────────────────────────────

def run_dumps(prompt: str, n_generate: int, sys_prompt: str,
              think: bool, c_prefix: str, rs_prefix: str,
              c_bin: str, rs_bin: str, layer: str, pos_filter: str) -> None:
    """Run both binaries with dump env vars to generate tensor dumps."""

    nothink_c  = [] if think else ["--nothink"]
    nothink_rs = [] if think else ["--nothink"]

    c_env = dict(os.environ,
                 DS4_METAL_GRAPH_DUMP_PREFIX=c_prefix,
                 DS4_METAL_GRAPH_DUMP_LAYER=layer)
    if pos_filter:
        c_env["DS4_METAL_GRAPH_DUMP_POS"] = pos_filter

    rs_env = dict(os.environ,
                  DS4_DUMP_PREFIX=rs_prefix,
                  DS4_DUMP_LAYER=layer)
    if pos_filter:
        rs_env["DS4_DUMP_POS"] = pos_filter

    c_cmd = [c_bin, "--metal", f"--ctx=16384", "-sys", sys_prompt,
             "-n", str(n_generate), "-p", prompt] + nothink_c
    rs_cmd = [rs_bin, "run", f"--ctx-size=16384", "--system", sys_prompt,
              "-n", str(n_generate), "-p", prompt] + nothink_rs

    print(f"Running C:    {' '.join(c_cmd)}")
    print(f"  env: DS4_METAL_GRAPH_DUMP_PREFIX={c_prefix} DS4_METAL_GRAPH_DUMP_LAYER={layer}")
    r = subprocess.run(c_cmd, env=c_env)
    if r.returncode != 0:
        sys.exit(f"ds4 (C) failed with exit code {r.returncode}")

    print(f"\nRunning Rust: {' '.join(rs_cmd)}")
    print(f"  env: DS4_DUMP_PREFIX={rs_prefix} DS4_DUMP_LAYER={layer}")
    r = subprocess.run(rs_cmd, env=rs_env)
    if r.returncode != 0:
        sys.exit(f"ds4f (Rust) failed with exit code {r.returncode}")


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("c_prefix",  nargs="?", help="C dump file prefix (DS4_METAL_GRAPH_DUMP_PREFIX value)")
    ap.add_argument("rs_prefix", nargs="?", help="Rust dump file prefix (DS4_DUMP_PREFIX value)")

    ap.add_argument("--pos", "-P", type=int, default=0,
                    help="Token position to compare (default 0 = first token/prefill)")
    ap.add_argument("--n-layers", type=int, default=N_LAYER,
                    help=f"Number of transformer layers (default {N_LAYER})")
    ap.add_argument("--tol", "-e", type=float, default=1e-4,
                    help="Max-absolute-error tolerance for a match (default 1e-4)")
    ap.add_argument("--all", "-a", dest="show_all", action="store_true",
                    help="Continue past first mismatch, show all layers")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Also print OK and SKIP lines")

    # --run mode: run both binaries then compare
    ap.add_argument("--run", action="store_true",
                    help="Run both ds4 and ds4f to generate dumps, then compare")
    ap.add_argument("--prompt", "-p", default="Hi",
                    help="Prompt text when using --run (default: 'Hi')")
    ap.add_argument("--n-generate", "-n", type=int, default=1,
                    help="Tokens to generate when using --run (default: 1)")
    ap.add_argument("--system", default="",
                    help="System prompt when using --run (default: empty)")
    ap.add_argument("--think", action="store_true",
                    help="Enable thinking mode when using --run")
    ap.add_argument("--layer", default="all",
                    help="Which layer to dump: 'all' or a layer number (default: all)")
    ap.add_argument("--c-bin", default=os.environ.get("DS4_BIN", "./ds4"),
                    help="Path to ds4 C binary (or DS4_BIN env var)")
    ap.add_argument("--rs-bin", default=os.environ.get("DS4F_BIN", "./ds4-rs/target/release/ds4f"),
                    help="Path to ds4f Rust binary (or DS4F_BIN env var)")

    args = ap.parse_args()

    if args.run:
        # Auto-generate dump directories
        c_prefix  = args.c_prefix  or "/tmp/ds4_c_dump"
        rs_prefix = args.rs_prefix or "/tmp/ds4_rs_dump"
        print(f"=== Generating dumps ===")
        print(f"  C   prefix: {c_prefix}")
        print(f"  Rust prefix: {rs_prefix}\n")
        pos_filter = str(args.pos) if args.layer != "all" else ""
        run_dumps(args.prompt, args.n_generate, args.system, args.think,
                  c_prefix, rs_prefix,
                  args.c_bin, args.rs_bin,
                  args.layer, pos_filter)
        print()
    else:
        if not args.c_prefix or not args.rs_prefix:
            ap.error("c_prefix and rs_prefix are required unless --run is given")
        c_prefix  = args.c_prefix
        rs_prefix = args.rs_prefix

    # Scan for available positions if pos was not given explicitly
    pos = args.pos

    # Auto-detect decode-step positions (prefer the highest common position
    # found in BOTH dumps, as that corresponds to the first generated token).
    # Only auto-detect if the user did NOT explicitly pass --pos on the command line.
    pos_explicit = "--pos" in sys.argv or "-P" in sys.argv
    if not args.run and not pos_explicit:
        c_positions  = discover_positions(c_prefix)
        rs_positions = discover_positions(rs_prefix)
        common = sorted(set(c_positions) & set(rs_positions))
        if common and args.pos == 0 and max(common) > 0:
            # The highest position present in both dumps is the decode step
            # (prefill positions are 0..N_prompt-1, decode is N_prompt+).
            pos = max(common)
            print(f"  (auto-detected pos={pos} as decode position; "
                  f"C positions={c_positions}, Rust positions={rs_positions})")
        elif not common:
            print(f"  WARNING: no common positions found between C and Rust dumps.")
            print(f"    C   positions: {c_positions}")
            print(f"    Rust positions: {rs_positions}")

    print(f"=== Comparing dumps ===")
    print(f"  C   prefix : {c_prefix}")
    print(f"  Rust prefix: {rs_prefix}")
    print(f"  pos        : {pos}")
    print(f"  tolerance  : {args.tol:.1e}")
    print(f"  stop-first : {not args.show_all}")
    print()

    ok = compare_dumps(
        c_prefix=c_prefix,
        rs_prefix=rs_prefix,
        pos=pos,
        tol=args.tol,
        stop_on_first=not args.show_all,
        verbose=args.verbose,
        n_layers=args.n_layers,
    )

    print()
    if ok:
        print(ANSI_GREEN + "All tensors match within tolerance." + ANSI_RESET)
        sys.exit(0)
    else:
        print(ANSI_RED + "Mismatch(es) found — see above." + ANSI_RESET)
        sys.exit(1)


if __name__ == "__main__":
    main()
