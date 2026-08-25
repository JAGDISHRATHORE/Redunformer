"""
Rectangular-strip purity probe -- extends the square tile purity probe to 1xN and
Nx1 strips, to settle the strip-orientation question the square probe cannot.

Reuses the validated importance-map / redundant-set / anisotropy machinery from
tile_purity_probe and only adds rectangular (Th, Tw) tiling. A strip can harvest
redundancy only where the redundant set is correlated ALONG its long axis, so the
harvestable fraction of 1xN vs Nx1 vs the equal-area square baseline shows which
orientation (if any) beats squares -- the empirical answer to whether the row
anisotropy (xi_row > xi_col) favours tall Nx1 strips or wide 1xN strips.

Convention (matches the rest of the repo): W is (out, in) = (rows, cols).
  shape (Th, Tw):  Th spans OUTPUT neurons (rows, axis 0),
                   Tw spans INPUT features (cols, axis 1).
  1xN = one output row, N input cols (harvests along the input-feature axis).
  Nx1 = N output rows, one input col (harvests along the output-neuron axis).

CPU self-test (--selftest) validates rectangular counting and orientation
selectivity on planted structure; it does not touch the GPU.

Example
  python scripts/strip_purity_probe.py --model Qwen/Qwen3-4B \
    --layers 0 9 18 27 34 35 \
    --shapes 2 4 8 16 32 1x2 1x4 1x8 1x16 1x32 2x1 4x1 8x1 16x1 32x1 \
    --primary-map wanda --primary-q 0.05 --null shuffle within_col --nperm 200 \
    --out experiments/tilesize/probe_strips
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                    # scripts/  (reuse run_pruning helpers)
sys.path.insert(0, os.path.join(HERE, "..", "src"))        # src/

from scipy.stats import binom

# Reused, shape-agnostic primitives from the square probe (importance maps,
# redundant-set thresholding, autocorrelation length, z-score, Fano).
from tile_purity_probe import (
    redundant_mask, acf_length, zscore, fano, build_maps_for_matrix,
)


# ----------------------------------------------------------------------------
# Shapes
# ----------------------------------------------------------------------------

def parse_shape(tok):
    """'1x4' -> (1, 4);  '4' -> (4, 4)."""
    t = str(tok).lower()
    if "x" in t:
        a, b = t.split("x")
        return (int(a), int(b))
    n = int(t)
    return (n, n)


def area(shape):
    return shape[0] * shape[1]


def orient(shape):
    Th, Tw = shape
    if Th == Tw:
        return "square"
    return "1xN" if Th == 1 else ("Nx1" if Tw == 1 else "rect")


# ----------------------------------------------------------------------------
# Rectangular tiling primitives (mirror tile_purity_probe, generalized to Th,Tw)
# ----------------------------------------------------------------------------

def tile_counts_rect(R, Th, Tw):
    """Per-tile redundant-weight count for a (Th, Tw) tiling of boolean R.
    R -> (nr, Th, nc, Tw) summed over the two tile axes."""
    rows, cols = R.shape
    nr, nc = rows // Th, cols // Tw
    if nr == 0 or nc == 0:
        return np.zeros((0, 0), dtype=np.int32)
    block = R[:nr * Th, :nc * Tw].astype(np.int32).reshape(nr, Th, nc, Tw)
    return block.sum(axis=(1, 3))                           # (nr, nc), 0..Th*Tw


def clean_harvest_rect(counts, Th, Tw, purity):
    """n_pure, n_tiles, redundant_in_pure for a (Th, Tw) tiling."""
    n_tiles = counts.size
    if n_tiles == 0:
        return 0, 0, 0
    thr = math.ceil(purity * Th * Tw)
    pure = counts >= thr
    return int(pure.sum()), n_tiles, int(counts[pure].sum())


def binom_null_rect(Th, Tw, q, n_tiles, purity):
    """Spatially-random null: each tile is area=Th*Tw iid Bernoulli(q)."""
    a = Th * Tw
    thr = math.ceil(purity * a)
    p_pure = float(binom.sf(thr - 1, a, q))                 # P(X >= thr)
    return n_tiles * p_pure, n_tiles * p_pure * (1.0 - p_pure), p_pure


# ---- numpy nulls (used by --selftest; CPU, no GPU) -------------------------

def shuffle_null_rect_np(R, shape, purity, nperm, rng):
    Th, Tw = shape
    rows, cols = R.shape
    thr = math.ceil(purity * Th * Tw)
    flat = R.reshape(-1).astype(np.int8)
    out = np.empty(nperm, dtype=np.int64)
    for p in range(nperm):
        c = tile_counts_rect(rng.permutation(flat).reshape(rows, cols), Th, Tw)
        out[p] = int((c >= thr).sum())
    return out


def within_col_null_rect_np(R, shape, purity, nperm, rng):
    Th, Tw = shape
    rows, cols = R.shape
    thr = math.ceil(purity * Th * Tw)
    Ri = R.astype(np.int8)
    out = np.empty(nperm, dtype=np.int64)
    for p in range(nperm):
        order = np.argsort(rng.random((rows, cols)), axis=0)
        Rp = np.take_along_axis(Ri, order, axis=0)
        out[p] = int((tile_counts_rect(Rp, Th, Tw) >= thr).sum())
    return out


# ---- torch nulls (real run; all shapes per draw, on GPU when available) -----

def _tile_counts_torch_rect(Rp, Th, Tw):
    rows, cols = Rp.shape
    nr, nc = rows // Th, cols // Tw
    return Rp[:nr * Th, :nc * Tw].reshape(nr, Th, nc, Tw).sum(dim=(1, 3))


def _shapes_for(R, shapes):
    rows, cols = R.shape
    return [s for s in shapes if area(s) > 1 and rows % s[0] == 0 and cols % s[1] == 0]


def shuffle_null_multi_rect(R, shapes, purity, nperm, rng):
    """Within-matrix (global shuffle) null for ALL shapes at once (Torch/GPU)."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, cols = R.shape
    S = _shapes_for(R, shapes)
    out = {s: np.empty(nperm, dtype=np.int64) for s in S}
    if not S:
        return out
    flat = torch.as_tensor(np.ascontiguousarray(R), dtype=torch.int32, device=dev).reshape(-1)
    n = flat.numel()
    g = torch.Generator(device=dev).manual_seed(int(rng.integers(0, 2**31 - 1)))
    thr = {s: math.ceil(purity * area(s)) for s in S}
    for p in range(nperm):
        Rp = flat[torch.randperm(n, generator=g, device=dev)].reshape(rows, cols)
        for s in S:
            out[s][p] = int((_tile_counts_torch_rect(Rp, s[0], s[1]) >= thr[s]).sum().item())
    return out


def within_col_null_multi_rect(R, shapes, purity, nperm, rng):
    """Within-COLUMN permutation null for ALL shapes at once (Torch/GPU).
    Preserves each column's redundant count; scrambles the row arrangement --
    the stringent null for Nx1 strips (which live along the output-neuron axis)."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, cols = R.shape
    S = _shapes_for(R, shapes)
    out = {s: np.empty(nperm, dtype=np.int64) for s in S}
    if not S:
        return out
    Rt = torch.as_tensor(np.ascontiguousarray(R), dtype=torch.int32, device=dev)
    g = torch.Generator(device=dev).manual_seed(int(rng.integers(0, 2**31 - 1)))
    thr = {s: math.ceil(purity * area(s)) for s in S}
    for p in range(nperm):
        order = torch.argsort(torch.rand(rows, cols, generator=g, device=dev), dim=0)
        Rp = torch.gather(Rt, 0, order)
        for s in S:
            out[s][p] = int((_tile_counts_torch_rect(Rp, s[0], s[1]) >= thr[s]).sum().item())
    return out


# ----------------------------------------------------------------------------
# Per-map analysis across all q and shapes (accumulates into `agg`)
# ----------------------------------------------------------------------------

def analyze_map_rect(imp, map_name, matrix_type, shapes, qs, nulls, nperm, purity,
                     rng, agg, aniso_accum, primary_q):
    rows, cols = imp.shape
    weights = rows * cols
    for q in qs:
        R = redundant_mask(imp, q)
        q_hat = float(R.mean())
        shuffle_null = shuffle_null_multi_rect(R, shapes, purity, nperm, rng) if "shuffle" in nulls else {}
        wcol_null = within_col_null_multi_rect(R, shapes, purity, nperm, rng) if "within_col" in nulls else {}
        for s in shapes:
            Th, Tw = s
            if rows % Th or cols % Tw:
                continue
            counts = tile_counts_rect(R, Th, Tw)
            n_pure, n_tiles, red_in_pure = clean_harvest_rect(counts, Th, Tw, purity)
            total_red = int(R.sum())
            b_mean, b_var, _ = binom_null_rect(Th, Tw, q_hat, n_tiles, purity)

            rec = agg[(map_name, round(q, 4), s)]
            rec["obs_pure"] += n_pure
            rec["n_tiles"] += n_tiles
            rec["weights"] += weights
            rec["red_in_pure"] += red_in_pure
            rec["total_red"] += total_red
            rec["binom_mean"] += b_mean
            rec["binom_var"] += b_var
            rec["cells"] += 1

            if "shuffle" in nulls and area(s) > 1 and s in shuffle_null:
                sh = shuffle_null[s]
                rec["shuffle_mean"] += float(np.mean(sh))
                rec["shuffle_var"] += float(np.var(sh))
                rec["shuffle_fano"] += fano(sh) if np.mean(sh) > 0 else 0.0
                rec["shuffle_cells"] += 1
            if "within_col" in nulls and area(s) > 1 and s in wcol_null:
                wc = wcol_null[s]
                rec["wcol_mean"] += float(np.mean(wc))
                rec["wcol_var"] += float(np.var(wc))
                rec["wcol_cells"] += 1

    if round(primary_q, 4) in [round(x, 4) for x in qs]:
        xi_row = acf_length(imp, axis=0)                    # along output neurons (rows)
        xi_col = acf_length(imp, axis=1)                    # along input features (cols)
        a = aniso_accum[(map_name, matrix_type)]
        a["xi_row"] += xi_row * (rows * cols)
        a["xi_col"] += xi_col * (rows * cols)
        a["w"] += rows * cols


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def strict_z(entry):
    if "z_within_col" in entry:
        return entry["z_within_col"]
    if "z_shuffle" in entry:
        return entry["z_shuffle"]
    return entry["z_binom"]


def summarize(agg, aniso_accum, shapes, qs, maps, nulls, primary_map, primary_q,
              purity, z_thresh):
    report = {"per_map": {}, "anisotropy_map_weighted": {}}

    for m in maps:
        report["per_map"][m] = {}
        for q in qs:
            qk = round(q, 4)
            curve = []
            for s in shapes:
                rec = agg.get((m, qk, s))
                if rec is None or rec["n_tiles"] == 0:
                    continue
                obs = rec["obs_pure"]
                n_tiles = rec["n_tiles"]
                harvest = obs / n_tiles                      # equal-size tiles
                harvest_eff = (rec["red_in_pure"] / rec["total_red"]) if rec["total_red"] else 0.0
                entry = {
                    "shape": f"{s[0]}x{s[1]}",
                    "Th": s[0], "Tw": s[1], "area": area(s), "orient": orient(s),
                    "harvestable": harvest,
                    "harvest_efficiency": harvest_eff,
                    "obs_pure": obs,
                    "n_tiles": n_tiles,
                    "z_binom": zscore(obs, rec["binom_mean"], rec["binom_var"]),
                }
                if rec.get("shuffle_cells"):
                    entry["z_shuffle"] = zscore(obs, rec["shuffle_mean"], rec["shuffle_var"])
                if rec.get("wcol_cells"):
                    entry["z_within_col"] = zscore(obs, rec["wcol_mean"], rec["wcol_var"])
                curve.append(entry)
            report["per_map"][m][str(qk)] = curve

    # tile-weighted anisotropy per map
    for m in maps:
        tw = {"xr": 0.0, "xc": 0.0, "w": 0.0}
        for (mm, _mtype), a in aniso_accum.items():
            if mm == m:
                tw["xr"] += a["xi_row"]; tw["xc"] += a["xi_col"]; tw["w"] += a["w"]
        if tw["w"] > 0:
            xr, xc = tw["xr"] / tw["w"], tw["xc"] / tw["w"]
            report["anisotropy_map_weighted"][m] = {
                "xi_row": xr, "xi_col": xc, "A": (xc / xr) if xr > 0 else float("inf"),
            }

    # --- orientation comparison at the primary map/q ----------------------
    curve = report["per_map"].get(primary_map, {}).get(str(round(primary_q, 4)), [])
    by_shape = {e["shape"]: e for e in curve}

    def best(pred):
        cand = [e for e in curve if pred(e) and (math.isfinite(strict_z(e)) and strict_z(e) >= z_thresh
                                                 or not math.isfinite(strict_z(e)))]
        # among significant shapes, the one with the largest harvestable
        pool = cand if cand else [e for e in curve if pred(e)]
        return max(pool, key=lambda e: e["harvestable"]) if pool else None

    best_1xN = best(lambda e: e["orient"] == "1xN")
    best_Nx1 = best(lambda e: e["orient"] == "Nx1")
    # square baseline: the largest square present (typically 32) and its harvestable
    squares = [e for e in curve if e["orient"] == "square"]
    sq_big = max(squares, key=lambda e: e["area"]) if squares else None

    def hv(e):
        return e["harvestable"] if e else 0.0

    winner = max(
        [("1xN", hv(best_1xN)), ("Nx1", hv(best_Nx1)), ("square", hv(sq_big))],
        key=lambda x: x[1],
    )[0]

    report["orientation_verdict"] = {
        "primary_map": primary_map,
        "primary_q": primary_q,
        "z_threshold": z_thresh,
        "best_1xN": best_1xN,
        "best_Nx1": best_Nx1,
        "square_baseline": sq_big,
        "winner": winner,
        "anisotropy": report["anisotropy_map_weighted"].get(primary_map, {}),
        "note": ("probe rule: A=xi_col/xi_row >> 1 favours 1xN, << 1 favours Nx1; "
                 "harvestable is the empirical arbiter above."),
    }
    return report


def print_verdict(report):
    v = report["orientation_verdict"]
    a = v.get("anisotropy", {})

    def fmt(e):
        if not e:
            return "none"
        return f"{e['shape']} harv={e['harvestable']:.4f} z={strict_z(e):.1f}"

    print("\n" + "=" * 72)
    print("  STRIP PURITY PROBE -- ORIENTATION VERDICT")
    print("=" * 72)
    print(f"  primary map/q     : {v['primary_map']} / q={v['primary_q']}")
    if a:
        print(f"  anisotropy        : xi_row={a.get('xi_row', 0):.2f} xi_col={a.get('xi_col', 0):.2f} "
              f"A={a.get('A', 0):.3f}")
    print(f"  best 1xN (wide)   : {fmt(v['best_1xN'])}")
    print(f"  best Nx1 (tall)   : {fmt(v['best_Nx1'])}")
    print(f"  square baseline   : {fmt(v['square_baseline'])}")
    print(f"  WINNER            : {v['winner']}   (largest harvestable among significant shapes)")
    print("  NOTE              : purity is the MASK-ONLY ceiling; the downstream ladder decides.")
    print("=" * 72 + "\n")


# ----------------------------------------------------------------------------
# Self-test: rectangular counting + orientation selectivity (CPU only)
# ----------------------------------------------------------------------------

def selftest():
    print("=" * 72)
    print("  SELF-TEST: rectangular tiling / purity / null math (CPU)")
    print("=" * 72)
    rng = np.random.default_rng(0)
    purity = 0.9
    fails = 0

    def check(name, cond, detail=""):
        nonlocal fails
        if not cond:
            fails += 1
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    # ---- Case 0: exact rectangular counting ------------------------------
    R = np.zeros((4, 8), dtype=bool)
    R[0, 0:4] = True                                        # a planted 1x4 strip
    c14 = tile_counts_rect(R, 1, 4)
    check("1x4 counting: tile[0,0]=4, total=4", c14[0, 0] == 4 and c14.sum() == 4, f"counts=\n{c14}")
    np_, nt_, rp_ = clean_harvest_rect(c14, 1, 4, purity)
    check("1x4 harvest: exactly 1 pure tile", np_ == 1 and rp_ == 4)

    # ---- Case 1: orientation selectivity, wide strip ---------------------
    n = 64
    imp = rng.random((n, n)) + 1.0
    imp[0, 0:8] = rng.random(8) * 1e-3                      # one low-importance 1x8 row-strip
    q = 8.0 / (n * n)
    Rw = redundant_mask(imp, q)
    check("planted 1x8: redundant set == the strip", Rw[0, 0:8].all() and Rw.sum() == 8, f"sum={Rw.sum()}")
    n18, _, _ = clean_harvest_rect(tile_counts_rect(Rw, 1, 8), 1, 8, purity)
    n81, _, _ = clean_harvest_rect(tile_counts_rect(Rw, 8, 1), 8, 1, purity)
    check("1x8 strip harvested by 1x8, NOT by 8x1", n18 == 1 and n81 == 0, f"1x8={n18} 8x1={n81}")

    # ---- Case 2: orientation selectivity, tall strip ---------------------
    impt = rng.random((n, n)) + 1.0
    impt[0:8, 0] = rng.random(8) * 1e-3                     # one low-importance 8x1 col-strip
    Rt = redundant_mask(impt, q)
    check("planted 8x1: redundant set == the strip", Rt[0:8, 0].all() and Rt.sum() == 8, f"sum={Rt.sum()}")
    m18, _, _ = clean_harvest_rect(tile_counts_rect(Rt, 1, 8), 1, 8, purity)
    m81, _, _ = clean_harvest_rect(tile_counts_rect(Rt, 8, 1), 8, 1, purity)
    check("8x1 strip harvested by 8x1, NOT by 1x8", m81 == 1 and m18 == 0, f"1x8={m18} 8x1={m81}")

    # ---- Case 3: binom null area uses Th*Tw ------------------------------
    _, _, p18 = binom_null_rect(1, 8, 0.05, 100, purity)
    _, _, p_sq = binom_null_rect(8, 8, 0.05, 100, purity)   # 64-area square, far rarer
    check("binom null: pure 1x8 (area 8) >> pure 8x8 (area 64) by chance", p18 > p_sq > 0,
          f"p(1x8)={p18:.2e} p(8x8)={p_sq:.2e}")

    # ---- Case 4: diffuse map ~ null for both orientations ----------------
    impd = rng.random((128, 128))
    Rd = redundant_mask(impd, 0.05)
    for sh in [(1, 8), (8, 1)]:
        cnt = tile_counts_rect(Rd, *sh)
        npr, ntl, _ = clean_harvest_rect(cnt, sh[0], sh[1], purity)
        bm, bv, _ = binom_null_rect(sh[0], sh[1], float(Rd.mean()), ntl, purity)
        z = zscore(npr, bm, bv)
        check(f"diffuse map ~ binom null for {sh[0]}x{sh[1]} (|z|<4)", abs(z) < 4, f"z={z:.2f}")

    # ---- Case 5: permutation nulls kill a planted strip's excess ---------
    s_np = shuffle_null_rect_np(Rw, (1, 8), purity, 100, rng)
    check("shuffle null ~0 pure 1x8 tiles (planted strip is real structure)", s_np.mean() < 0.5,
          f"mean={s_np.mean():.3f}")
    wc_np = within_col_null_rect_np(Rt, (8, 1), purity, 100, rng)
    check("within-col null ~0 pure 8x1 tiles (destroys the vertical strip)", wc_np.mean() < 0.5,
          f"mean={wc_np.mean():.3f}")

    print("=" * 72)
    print(f"  SELF-TEST RESULT: {'ALL PASSED' if fails == 0 else str(fails) + ' FAILED'}")
    print("=" * 72)
    return fails == 0


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def default_layers():
    return [0, 9, 18, 27, 34, 35]


def default_shapes():
    return ["2", "4", "8", "16", "32",
            "1x2", "1x4", "1x8", "1x16", "1x32",
            "2x1", "4x1", "8x1", "16x1", "32x1"]


def make_plots(report, maps, primary_q, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qk = str(round(primary_q, 4))
    plt.figure(figsize=(9, 5), dpi=150)
    styles = {"square": ("o", "-"), "1xN": ("s", "--"), "Nx1": ("^", ":")}
    for m in maps:
        c = report["per_map"].get(m, {}).get(qk, [])
        if not c:
            continue
        for o, (mk, ls) in styles.items():
            pts = sorted([e for e in c if e["orient"] == o], key=lambda e: e["area"])
            if not pts:
                continue
            plt.plot([e["area"] for e in pts], [e["harvestable"] for e in pts],
                     marker=mk, linestyle=ls, label=f"{m}:{o}")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("tile area (weights)")
    plt.ylabel("harvestable weight-fraction (purity>=p)")
    plt.title(f"Harvestable vs area by orientation  q={primary_q}  [MASK-ONLY ceiling]")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    p1 = os.path.join(out_dir, "harvestable_by_orientation.png")
    plt.savefig(p1, bbox_inches="tight")
    plt.close()
    return [p1]


def main():
    ap = argparse.ArgumentParser(description="Rectangular-strip purity probe (1xN vs Nx1 vs square).")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--all-layers", action="store_true")
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--matrices", nargs="+", default=None)
    ap.add_argument("--shapes", nargs="+", default=None,
                    help="Tile shapes: '4' (square) or '1x4'/'4x1' (strip). Default: squares + 1xN + Nx1.")
    ap.add_argument("--q", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    ap.add_argument("--maps", nargs="+", default=["wanda", "obs", "magnitude"],
                    choices=["wanda", "obs", "magnitude"])
    ap.add_argument("--null", nargs="+", default=["shuffle", "within_col"],
                    choices=["shuffle", "within_col"])
    ap.add_argument("--nperm", type=int, default=200)
    ap.add_argument("--purity", type=float, default=0.9)
    ap.add_argument("--damp", type=float, default=1e-2)
    ap.add_argument("--calib-samples", type=int, default=128)
    ap.add_argument("--calib-seqlen", type=int, default=512)
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--primary-map", default="wanda", choices=["wanda", "obs", "magnitude"])
    ap.add_argument("--primary-q", type=float, default=0.05)
    ap.add_argument("--z-threshold", type=float, default=4.0)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/tilesize/probe_strips")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.rng_seed)
    shapes = [parse_shape(s) for s in (args.shapes or default_shapes())]

    import torch
    from run_pruning import (MATRICES, build_target_name, get_target_weight,
                             get_target_module, get_num_layers)
    from redundancy.models import load_model_and_tokenizer
    from redundancy.data import load_calibration_dataset
    from redundancy.hooks import collect_gram_stats

    matrices = args.matrices or list(MATRICES.keys())
    model, tokenizer = load_model_and_tokenizer(args.model)
    if args.all_layers:
        layers = list(range(get_num_layers(model)))
    elif args.layers is not None:
        layers = args.layers
    else:
        layers = default_layers()
    print(f"Probing layers {layers}  matrices {matrices}  shapes {[f'{a}x{b}' for a, b in shapes]}")

    calib = load_calibration_dataset(tokenizer, n_samples=args.calib_samples,
                                     seqlen=args.calib_seqlen, seed=args.calib_seed)

    agg = defaultdict(lambda: defaultdict(float))
    aniso_accum = defaultdict(lambda: defaultdict(float))

    for li in layers:
        target_names = [build_target_name(li, m) for m in matrices]
        modules_by_name = {n: get_target_module(model, n) for n in target_names}
        print(f"[layer {li}] collecting Grams ({len(modules_by_name)} matrices)...")
        collectors = collect_gram_stats(model, modules_by_name, calib)
        for mtype, tname in zip(matrices, target_names):
            H = collectors[tname].H
            W = get_target_weight(model, tname)
            maps = build_maps_for_matrix(W, H, args.maps, args.damp)
            for map_name, imp in maps.items():
                analyze_map_rect(imp.astype(np.float32), map_name, mtype, shapes, args.q,
                                 args.null, args.nperm, args.purity, rng, agg, aniso_accum,
                                 args.primary_q)
            del maps
        del collectors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = summarize(dict(agg), dict(aniso_accum), shapes, args.q, args.maps,
                       args.null, args.primary_map, args.primary_q, args.purity,
                       args.z_threshold)
    report["config"] = {
        "model": args.model, "layers": layers, "matrices": matrices,
        "shapes": [f"{a}x{b}" for a, b in shapes], "q": args.q, "maps": args.maps,
        "null": args.null, "nperm": args.nperm, "purity": args.purity, "damp": args.damp,
        "calib_samples": args.calib_samples, "calib_seqlen": args.calib_seqlen,
        "calib_seed": args.calib_seed, "primary_map": args.primary_map,
        "primary_q": args.primary_q, "z_threshold": args.z_threshold,
    }

    out_json = os.path.join(args.out, "strip_purity_probe.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved probe report -> {out_json}")
    plots = make_plots(report, args.maps, args.primary_q, args.out)
    print(f"Saved plots -> {plots}")
    print_verdict(report)


if __name__ == "__main__":
    main()
