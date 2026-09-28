import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import binom


def count_pure_tiles(R, T):
    rows, cols = R.shape
    nr, nc = rows // T, cols // T
    if nr == 0 or nc == 0:
        return np.zeros((0, 0), dtype=np.int32)
    cut_r, cut_c = nr * T, nc * T
    block = R[:cut_r, :cut_c].astype(np.int32).reshape(nr, T, nc, T)
    return block.sum(axis=(1, 3))


def harvest_tiles(counts, T, purity):
    n_tiles = counts.size
    if n_tiles == 0:
        return 0, 0, 0
    thresh_count = math.ceil(purity * T * T)
    pure = counts >= thresh_count
    n_pure = int(pure.sum())
    red_pure = int(counts[pure].sum())
    return n_pure, n_tiles, red_pure


def binom_null(T, q, n_tiles, purity):
    thresh_count = math.ceil(purity * T * T)
    p_pure = float(binom.sf(thresh_count - 1, T * T, q))
    mean = n_tiles * p_pure
    var = n_tiles * p_pure * (1.0 - p_pure)
    return mean, var, p_pure


def null_shuffle(R, T, purity, nperm, rng):
    flat = R.reshape(-1).astype(np.int8)
    rows, cols = R.shape
    thresh_count = math.ceil(purity * T * T)
    out = np.empty(nperm, dtype=np.int64)
    for p in range(nperm):
        perm = rng.permutation(flat)
        counts = count_pure_tiles(perm.reshape(rows, cols), T)
        out[p] = int((counts >= thresh_count).sum())
    return out


def null_wcol(R, T, purity, nperm, rng):
    rows, cols = R.shape
    thresh_count = math.ceil(purity * T * T)
    Ri = R.astype(np.int8)
    out = np.empty(nperm, dtype=np.int64)
    for p in range(nperm):
        order = np.argsort(rng.random((rows, cols)), axis=0)
        Rp = np.take_along_axis(Ri, order, axis=0)
        counts = count_pure_tiles(Rp, T)
        out[p] = int((counts >= thresh_count).sum())
    return out


def tiles_for(R, tiles):
    rows, cols = R.shape
    return [T for T in tiles if T > 1 and rows % T == 0 and cols % T == 0]


def null_dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def counts_torch(Rp, T):
    rows, cols = Rp.shape
    nr, nc = rows // T, cols // T
    block = Rp[:nr * T, :nc * T].reshape(nr, T, nc, T)
    return block.sum(dim=(1, 3))


def null_shuffle_multi(R, tiles, purity, nperm, rng):
    dev = null_dev()
    rows, cols = R.shape
    Ts = tiles_for(R, tiles)
    out = {T: np.empty(nperm, dtype=np.int64) for T in Ts}
    if not Ts:
        return out
    flat = torch.as_tensor(np.ascontiguousarray(R), dtype=torch.int32, device=dev).reshape(-1)
    n = flat.numel()
    g = torch.Generator(device=dev).manual_seed(int(rng.integers(0, 2**31 - 1)))
    thr = {T: math.ceil(purity * T * T) for T in Ts}
    for p in range(nperm):
        Rp = flat[torch.randperm(n, generator=g, device=dev)].reshape(rows, cols)
        for T in Ts:
            out[T][p] = int((counts_torch(Rp, T) >= thr[T]).sum().item())
    return out


def null_wcol_multi(R, tiles, purity, nperm, rng):
    dev = null_dev()
    rows, cols = R.shape
    Ts = tiles_for(R, tiles)
    out = {T: np.empty(nperm, dtype=np.int64) for T in Ts}
    if not Ts:
        return out
    Rt = torch.as_tensor(np.ascontiguousarray(R), dtype=torch.int32, device=dev)
    g = torch.Generator(device=dev).manual_seed(int(rng.integers(0, 2**31 - 1)))
    thr = {T: math.ceil(purity * T * T) for T in Ts}
    for p in range(nperm):
        order = torch.argsort(torch.rand(rows, cols, generator=g, device=dev), dim=0)
        Rp = torch.gather(Rt, 0, order)
        for T in Ts:
            out[T][p] = int((counts_torch(Rp, T) >= thr[T]).sum().item())
    return out


def zscore(obs, mean, var):
    sd = math.sqrt(var) if var > 0 else 0.0
    if sd == 0.0:
        return 0.0 if obs <= mean else float("inf")
    return (obs - mean) / sd


def fano(samples):
    m = float(np.mean(samples))
    if m <= 0:
        return 0.0
    return float(np.var(samples) / m)


def corr_len(M, axis, max_lag=64):
    n = M.shape[axis]
    max_lag = int(min(max_lag, n - 1))
    if max_lag < 1:
        return 0.0
    x = (M - M.mean(axis=axis, keepdims=True)).astype(np.float64)
    nf = 1
    while nf < 2 * n:
        nf *= 2
    F = np.fft.rfft(x, n=nf, axis=axis)
    ac = np.fft.irfft(F * np.conj(F), n=nf, axis=axis)
    sl = [slice(None), slice(None)]
    sl[axis] = slice(0, max_lag + 1)
    ac = ac[tuple(sl)]
    ac = ac.mean(axis=1 - axis)
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    thr = 1.0 / math.e
    for lag in range(1, len(ac)):
        if ac[lag] < thr:
            prev = ac[lag - 1]
            frac = (prev - thr) / (prev - ac[lag] + 1e-12)
            return float((lag - 1) + frac)
    return float(max_lag)


def low_mask(imp, q):
    thr = np.quantile(imp, q)
    return imp <= thr


def importance_maps(weight, H, wanted_maps, damp):
    import torch
    W = weight.detach().float()
    Hf = H.detach().float()
    maps = {}
    col_norms = torch.sqrt(torch.clamp(torch.diag(Hf), min=0.0))
    if "magnitude" in wanted_maps:
        maps["magnitude"] = W.abs().cpu().numpy()
    if "wanda" in wanted_maps:
        maps["wanda"] = (W.abs() * col_norms.unsqueeze(0)).cpu().numpy()
    if "obs" in wanted_maps:
        from redundancy.recovery import damped_inv
        Hinv = damped_inv(Hf, damp)
        diag_hinv = torch.clamp(torch.diag(Hinv), min=1e-12)
        maps["obs"] = ((W * W) / diag_hinv.unsqueeze(0)).cpu().numpy()
        del Hinv
    return maps


def analyze(imp, mname, matrix_type, tiles, qs, nulls, nperm, purity, rng,
            agg, aniso, primary_q):
    rows, cols = imp.shape
    weights = rows * cols
    for q in qs:
        R = low_mask(imp, q)
        q_hat = float(R.mean())
        shuffle_null = null_shuffle_multi(R, tiles, purity, nperm, rng) if "shuffle" in nulls else {}
        wcol_null = null_wcol_multi(R, tiles, purity, nperm, rng) if "within_col" in nulls else {}
        for T in tiles:
            if rows % T or cols % T:
                continue
            counts = count_pure_tiles(R, T)
            n_pure, n_tiles, red_in_pure = harvest_tiles(counts, T, purity)
            total_red = int(R.sum())

            b_mean, b_var, _ = binom_null(T, q_hat, n_tiles, purity)

            rec = agg[(mname, round(q, 4), T)]
            rec["obs_pure"] += n_pure
            rec["n_tiles"] += n_tiles
            rec["weights"] += weights
            rec["red_in_pure"] += red_in_pure
            rec["total_red"] += total_red
            rec["binom_mean"] += b_mean
            rec["binom_var"] += b_var
            rec["cells"] += 1

            if "shuffle" in nulls and T > 1:
                s = shuffle_null[T]
                rec["shuffle_mean"] += float(np.mean(s))
                rec["shuffle_var"] += float(np.var(s))
                rec["shuffle_fano"] += fano(s) if np.mean(s) > 0 else 0.0
                rec["shuffle_cells"] += 1
            if "within_col" in nulls and T > 1:
                w = wcol_null[T]
                rec["wcol_mean"] += float(np.mean(w))
                rec["wcol_var"] += float(np.var(w))
                rec["wcol_cells"] += 1

            trec = agg[("__bytype__", matrix_type, mname, round(q, 4), T)]
            trec["obs_pure"] += n_pure
            trec["n_tiles"] += n_tiles
            trec["binom_mean"] += b_mean
            trec["binom_var"] += b_var
            trec["cells"] += 1

    if round(primary_q, 4) in [round(x, 4) for x in qs]:
        xi_row = corr_len(imp, axis=0)
        xi_col = corr_len(imp, axis=1)
        a = aniso[(mname, matrix_type)]
        a["xi_row"] += xi_row * (rows * cols)
        a["xi_col"] += xi_col * (rows * cols)
        a["w"] += rows * cols


def summarize(agg, aniso, tiles, qs, maps, nulls, primary_map, primary_q,
              purity, z_thresh):
    report = {"per_map": {}, "by_type": {}, "anisotropy": {}}

    for m in maps:
        report["per_map"][m] = {}
        for q in qs:
            qk = round(q, 4)
            curve = []
            for T in tiles:
                rec = agg.get((m, qk, T))
                if rec is None or rec["n_tiles"] == 0:
                    continue
                obs = rec["obs_pure"]
                n_tiles = rec["n_tiles"]
                clean = obs / n_tiles
                harvest = obs / n_tiles
                harvest_eff = (rec["red_in_pure"] / rec["total_red"]) if rec["total_red"] else 0.0
                z_binom = zscore(obs, rec["binom_mean"], rec["binom_var"])
                entry = {
                    "T": T,
                    "clean_fraction": clean,
                    "harvestable": harvest,
                    "harvest_efficiency": harvest_eff,
                    "obs_pure": obs,
                    "n_tiles": n_tiles,
                    "binom_null_mean": rec["binom_mean"],
                    "z_binom": z_binom,
                }
                if rec.get("shuffle_cells"):
                    entry["z_shuffle"] = zscore(obs, rec["shuffle_mean"], rec["shuffle_var"])
                    entry["shuffle_null_mean"] = rec["shuffle_mean"]
                    entry["shuffle_fano"] = rec["shuffle_fano"] / rec["shuffle_cells"]
                if rec.get("wcol_cells"):
                    entry["z_within_col"] = zscore(obs, rec["wcol_mean"], rec["wcol_var"])
                    entry["within_col_null_mean"] = rec["wcol_mean"]
                curve.append(entry)
            report["per_map"][m][str(qk)] = curve

    for (m, mtype), a in aniso.items():
        if a["w"] == 0:
            continue
        xr = a["xi_row"] / a["w"]
        xc = a["xi_col"] / a["w"]
        report["anisotropy"].setdefault(m, {})[mtype] = {
            "xi_row": xr, "xi_col": xc, "A": (xc / xr) if xr > 0 else float("inf"),
        }
    aniso_map = {}
    for m in maps:
        tw = {"xr": 0.0, "xc": 0.0, "w": 0.0}
        for (mm, mtype), a in aniso.items():
            if mm == m:
                tw["xr"] += a["xi_row"]; tw["xc"] += a["xi_col"]; tw["w"] += a["w"]
        if tw["w"] > 0:
            xr, xc = tw["xr"] / tw["w"], tw["xc"] / tw["w"]
            aniso_map[m] = {"xi_row": xr, "xi_col": xc, "A": (xc / xr) if xr > 0 else float("inf")}
    report["anisotropy_map_weighted"] = aniso_map

    def curve_for(m, q):
        return report["per_map"].get(m, {}).get(str(round(q, 4)), [])

    def strict_z(entry):
        if "z_within_col" in entry:
            return entry["z_within_col"]
        if "z_shuffle" in entry:
            return entry["z_shuffle"]
        return entry["z_binom"]

    CAP = 1e6
    s_by_map = {}
    for m in maps:
        c = curve_for(m, primary_q)
        sig_Ts, peak_z, peak_z_T = [], 0.0, 1
        for e in c:
            if e["T"] == 1:
                continue
            z = strict_z(e)
            zc = CAP if not math.isfinite(z) else min(z, CAP)
            if zc > peak_z:
                peak_z, peak_z_T = zc, e["T"]
            if (not math.isfinite(z)) or z >= z_thresh:
                sig_Ts.append(e["T"])
        clustered = len(sig_Ts) > 0
        s_by_map[m] = {
            "S": max(sig_Ts) if clustered else 1,
            "peak_z": peak_z,
            "peak_z_T": peak_z_T,
            "clustered": clustered,
        }
    report["S_by_map"] = s_by_map

    w_clust = s_by_map.get(primary_map, {}).get("clustered", False)
    obs_clust = s_by_map.get("obs", {}).get("clustered", w_clust)
    agree = (w_clust == obs_clust)
    S = s_by_map.get(primary_map, {}).get("S", 1)

    columnar_only = False
    if "within_col" in nulls and "shuffle" in nulls:
        c = curve_for(primary_map, primary_q)
        for e in c:
            if e["T"] == S and S > 1:
                zc = e.get("z_within_col", 0.0)
                zs = e.get("z_shuffle", 0.0)
                zc = zc if math.isfinite(zc) else 1e9
                zs = zs if math.isfinite(zs) else 1e9
                if zs >= z_thresh and zc < z_thresh:
                    columnar_only = True

    if not w_clust:
        verdict = "DIFFUSE"
        S = 1
    elif columnar_only:
        verdict = "COLUMNAR_ONLY"
    else:
        verdict = "CLUSTERED"

    aniso_primary = aniso_map.get(primary_map, {})
    A = aniso_primary.get("A", 1.0)

    report["verdict"] = {
        "verdict": verdict,
        "S": S,
        "primary_map": primary_map,
        "primary_q": primary_q,
        "wanda_obs_agree": agree,
        "columnar_only": columnar_only,
        "anisotropy_A": A,
        "z_threshold": z_thresh,
    }

    c = curve_for(primary_map, primary_q)
    harvest_T = {e["T"]: e["harvestable"] for e in c}
    h32 = harvest_T.get(32, 0.0)
    new_sizes = [t for t in (16, 8, 4, 1) if t in harvest_T]
    gains = [(t, harvest_T[t] - h32) for t in new_sizes]
    gains.sort(key=lambda x: x[1], reverse=True)
    priority = [t for t, _ in gains]
    report["sweep_priority"] = {
        "order": priority,
        "predicted_gain_over_T32": {str(t): g for t, g in gains},
        "harvestable_by_T": {str(t): harvest_T[t] for t in sorted(harvest_T)},
        "run_T2": bool(S < 4),
    }
    return report


def print_result(report):
    v = report["verdict"]
    print("\n" + "=" * 72)
    print("  TILE PURITY PROBE -- VERDICT")
    print("=" * 72)
    print(f"  Verdict           : {v['verdict']}")
    print(f"  Correlation length: S = {v['S']}  (primary map={v['primary_map']}, q={v['primary_q']})")
    print(f"  Anisotropy A      : {v['anisotropy_A']:.3f}   (A=xi_col/xi_row; >>1 => 1xN strips would pay)")
    print(f"  Wanda/OBS agree   : {v['wanda_obs_agree']}")
    if v["columnar_only"]:
        print("  NOTE: pure-tile excess survives the within-matrix null but NOT the")
        print("        within-column null -> columnar modulation, not 2D block structure.")
    print(f"  S per map         : " + ", ".join(
          f"{m}:S={d['S']}(z={d['peak_z']:.1f})" for m, d in report["S_by_map"].items()))
    sp = report["sweep_priority"]
    print(f"  SWEEP PRIORITY    : {sp['order']}   (descending predicted gain over T=32)")
    print(f"    predicted gain  : " + ", ".join(
          f"T{t}:{g:+.4f}" for t, g in sp["predicted_gain_over_T32"].items()))
    if sp["run_T2"]:
        print("    -> S < 4: ADD T=2 to the sweep (1-vs-2-vs-4 resolution matters).")
    print("  CIRCULARITY GUARD : purity predicts the MASK-ONLY ceiling; the repaired")
    print("        ceiling is higher and less shape-sensitive. Probe predicts+prioritizes;")
    print("        the downstream ladder decides. A probe/ladder mismatch is itself a finding.")
    print("=" * 72 + "\n")


def make_plots(report, tiles, maps, primary_q, out_dir):
    qk = str(round(primary_q, 4))
    plt.figure(figsize=(8, 5), dpi=150)
    for m in maps:
        c = report["per_map"].get(m, {}).get(qk, [])
        if not c:
            continue
        Ts = [e["T"] for e in c]
        hv = [e["harvestable"] for e in c]
        plt.plot(Ts, hv, marker="o", label=f"{m} (obs)")
    c = report["per_map"].get(maps[0], {}).get(qk, [])
    if c:
        Ts = [e["T"] for e in c]
        nn = [e["binom_null_mean"] / e["n_tiles"] if e["n_tiles"] else 0 for e in c]
        plt.plot(Ts, nn, linestyle="--", color="grey", label="binomial null")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("tile size T"); plt.ylabel("harvestable weight-fraction (purity>=p)")
    plt.title(f"Harvestable(T) at q={primary_q}  [MASK-ONLY ceiling proxy]")
    plt.grid(True, which="both", alpha=0.3); plt.legend()
    plt.tight_layout()
    p1 = os.path.join(out_dir, "harvestable_vs_T.png")
    plt.savefig(p1, bbox_inches="tight"); plt.close()

    plt.figure(figsize=(8, 5), dpi=150)
    for m in maps:
        c = report["per_map"].get(m, {}).get(qk, [])
        if not c:
            continue
        Ts = [e["T"] for e in c if e["T"] > 1]
        for key, style in (("z_within_col", "-"), ("z_shuffle", "--"), ("z_binom", ":")):
            zs = [e.get(key) for e in c if e["T"] > 1]
            if any(z is not None for z in zs):
                zz = [min(z, 1e3) if (z is not None and math.isfinite(z)) else (1e3 if z is not None else np.nan)
                      for z in zs]
                plt.plot(Ts, zz, style, marker=".", label=f"{m}:{key}")
    plt.axhline(report["verdict"]["z_threshold"], color="red", linestyle="-", alpha=0.4,
                label="z threshold")
    plt.xscale("log", base=2)
    plt.xlabel("tile size T"); plt.ylabel("excess-pure z-score")
    plt.title(f"Clustering z-score(T) at q={primary_q}  (S = argmax_T z)")
    plt.grid(True, which="both", alpha=0.3); plt.legend(fontsize=7)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "zscore_vs_T.png")
    plt.savefig(p2, bbox_inches="tight"); plt.close()
    return [p1, p2]


def self_test():
    print("=" * 72)
    print("  SELF-TEST: tiling / purity / null math on synthetic KNOWN structure")
    print("=" * 72)
    rng = np.random.default_rng(0)
    purity = 0.9
    fails = 0

    def check(name, cond, detail=""):
        nonlocal fails
        status = "PASS" if cond else "FAIL"
        if not cond:
            fails += 1
        print(f"  [{status}] {name}{('  -- ' + detail) if detail else ''}")

    R = np.zeros((4, 4), dtype=bool)
    R[0:2, 0:2] = True
    counts = count_pure_tiles(R, 2)
    check("reshape counts: 2x2 block -> tile[0,0]=4, rest=0",
          counts[0, 0] == 4 and counts.sum() == 4, f"counts=\n{counts}")
    n_pure, n_tiles, red_in_pure = harvest_tiles(counts, 2, purity)
    check("clean_and_harvest: exactly 1 pure tile of 4",
          n_pure == 1 and n_tiles == 4 and red_in_pure == 4)
    check("T=1 clean_fraction equals q (self-consistency)",
          harvest_tiles(count_pure_tiles(R, 1), 1, purity)[0] == 4)

    n = 64
    imp = rng.random((n, n)) + 1.0
    imp[0:8, 0:8] = rng.random((8, 8)) * 1e-3
    q = 64.0 / (n * n)
    Rb = low_mask(imp, q)
    check("planted block: redundant set == the 8x8 block",
          Rb[0:8, 0:8].all() and Rb.sum() == 64, f"sum={Rb.sum()}")
    counts8 = count_pure_tiles(Rb, 8)
    n_pure8, n_tiles8, _ = harvest_tiles(counts8, 8, purity)
    check("planted block: T=8 has exactly one pure tile", n_pure8 == 1 and n_tiles8 == 64)
    b_mean, b_var, p_pure = binom_null(8, q, n_tiles8, purity)
    check("binomial null: a pure 8x8 tile is astronomically unlikely (p<1e-30)",
          p_pure < 1e-30, f"p_pure={p_pure:.3e}")
    z_b = zscore(n_pure8, b_mean, b_var)
    check("binomial z-score huge (=> CLUSTERED) for planted block",
          not math.isfinite(z_b) or z_b > 100, f"z={z_b}")
    s_sh = null_shuffle(Rb, 8, purity, 200, rng)
    check("within-matrix shuffle null ~ 0 pure tiles for T=8", s_sh.mean() < 0.05,
          f"mean={s_sh.mean():.4f}")
    s_wc = null_wcol(Rb, 8, purity, 200, rng)
    check("within-column null also ~0 (block is genuine 2D structure, not columnar)",
          s_wc.mean() < 0.5, f"mean={s_wc.mean():.4f}")

    impd = rng.random((128, 128))
    qd = 0.05
    Rd = low_mask(impd, qd)
    countsd = count_pure_tiles(Rd, 8)
    n_pured, n_tilesd, _ = harvest_tiles(countsd, 8, purity)
    bmd, bvd, _ = binom_null(8, float(Rd.mean()), n_tilesd, purity)
    zd = zscore(n_pured, bmd, bvd)
    check("diffuse map: T=8 pure tiles ~ binomial null (|z|<4)",
          abs(zd) < 4, f"obs={n_pured} null={bmd:.3f} z={zd:.2f}")

    impc = rng.random((64, 64)) + 1.0
    impc[:, 0:4] = rng.random((64, 4)) * 1e-3
    qc = (64 * 4) / (64.0 * 64.0)
    Rc = low_mask(impc, qc)
    check("columnar: redundant set == 4 full columns",
          Rc[:, 0:4].all() and Rc.sum() == 64 * 4, f"sum={Rc.sum()}")
    countsc = count_pure_tiles(Rc, 4)
    n_purec, n_tilesc, _ = harvest_tiles(countsc, 4, purity)
    bmc, bvc, _ = binom_null(4, float(Rc.mean()), n_tilesc, purity)
    z_shuf = zscore(n_purec, bmc, bvc)
    wc = null_wcol(Rc, 4, purity, 200, rng)
    z_wc = zscore(n_purec, wc.mean(), wc.var())
    check("columnar: strong signal vs within-matrix/binomial null (z>10)",
          not math.isfinite(z_shuf) or z_shuf > 10, f"z_shuffle={z_shuf}")
    check("columnar: NO excess vs within-column null (structure is columnar, |z|<4)",
          abs(z_wc) < 4 or not math.isfinite(z_wc) is False and abs(z_wc) < 4,
          f"obs={n_purec} wcol_null={wc.mean():.2f} z_wcol={z_wc:.2f}")

    base = rng.random((128, 128))
    kernel = np.ones(16)
    smoothed = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 1, base)
    xr = corr_len(smoothed, axis=0)
    xc = corr_len(smoothed, axis=1)
    check("anisotropy: column-smoothed map has xi_col > xi_row (A>1)",
          xc > xr, f"xi_row={xr:.2f} xi_col={xc:.2f} A={xc/max(xr,1e-9):.2f}")

    print("=" * 72)
    print(f"  SELF-TEST RESULT: {'ALL PASSED' if fails == 0 else str(fails) + ' FAILED'}")
    print("=" * 72)
    return fails == 0


def default_layers():
    return [0, 9, 18, 27, 34, 35]


def main():
    ap = argparse.ArgumentParser(description="Tile purity probe: correlation-length measurement.")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--all-layers", action="store_true", help="Probe every transformer layer.")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="Specific layers (default: 0 9 18 27 34 35).")
    ap.add_argument("--matrices", nargs="+", default=None,
                    help="Restrict to these projection types (default: all seven).")
    ap.add_argument("--tiles", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--q", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    ap.add_argument("--scope", choices=["per_matrix", "global"], default="per_matrix",
                    help="Redundant-set threshold per matrix (default) or global sensitivity view.")
    ap.add_argument("--maps", nargs="+", default=["wanda", "obs", "magnitude"],
                    choices=["wanda", "obs", "magnitude"])
    ap.add_argument("--null", nargs="+", default=["shuffle", "within_col"],
                    choices=["shuffle", "within_col"])
    ap.add_argument("--nperm", type=int, default=200)
    ap.add_argument("--purity", type=float, default=0.9, help="Purity threshold for a 'clean' tile.")
    ap.add_argument("--damp", type=float, default=1e-2, help="OBS damped-inverse damping.")
    ap.add_argument("--calib-samples", type=int, default=128)
    ap.add_argument("--calib-seqlen", type=int, default=512)
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--primary-map", default="wanda", choices=["wanda", "obs", "magnitude"])
    ap.add_argument("--primary-q", type=float, default=0.05)
    ap.add_argument("--z-threshold", type=float, default=4.0,
                    help="Excess-pure z above this => clustered at that T.")
    ap.add_argument("--rng-seed", type=int, default=0, help="Permutation-null RNG seed.")
    ap.add_argument("--out", default="experiments/experiment_2_qwen3_4b/results/10_tile_size_and_shape/probe")
    ap.add_argument("--selftest", action="store_true",
                    help="CPU-only: validate the tiling/purity/null math on synthetic matrices.")
    args = ap.parse_args()

    if args.selftest:
        ok = self_test()
        sys.exit(0 if ok else 1)

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.rng_seed)

    import torch
    from run_tile_pruning import (PROJS, param_name, get_weight,
                                  get_module, count_layers)
    from redundancy.models import load_model
    from redundancy.data import get_calib_data
    from redundancy.hooks import collect_grams

    matrices = args.matrices or list(PROJS.keys())
    model, tokenizer = load_model(args.model)
    if args.all_layers:
        layers = list(range(count_layers(model)))
    elif args.layers is not None:
        layers = args.layers
    else:
        layers = default_layers()
    print(f"Probing layers {layers}  matrices {matrices}")

    calib = get_calib_data(tokenizer, n_calib=args.calib_samples,
                           seqlen=args.calib_seqlen, seed=args.calib_seed)

    agg = defaultdict(lambda: defaultdict(float))
    aniso = defaultdict(lambda: defaultdict(float))

    for li in layers:
        pnames = [param_name(li, m) for m in matrices]
        mods = {n: get_module(model, n) for n in pnames}
        print(f"[layer {li}] collecting Grams ({len(mods)} matrices)...")
        collectors = collect_grams(model, mods, calib)
        for mtype, tname in zip(matrices, pnames):
            H = collectors[tname].H
            W = get_weight(model, tname)
            maps = importance_maps(W, H, args.maps, args.damp)
            for mname, imp in maps.items():
                analyze(imp.astype(np.float32), mname, mtype, args.tiles, args.q,
                        args.null, args.nperm, args.purity, rng, agg, aniso,
                        args.primary_q)
            del maps
        del collectors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = summarize(dict(agg), dict(aniso), args.tiles, args.q, args.maps,
                       args.null, args.primary_map, args.primary_q, args.purity,
                       args.z_threshold)
    report["config"] = {
        "model": args.model, "layers": layers, "matrices": matrices,
        "tiles": args.tiles, "q": args.q, "scope": args.scope, "maps": args.maps,
        "null": args.null, "nperm": args.nperm, "purity": args.purity, "damp": args.damp,
        "calib_samples": args.calib_samples, "calib_seqlen": args.calib_seqlen,
        "calib_seed": args.calib_seed, "primary_map": args.primary_map,
        "primary_q": args.primary_q, "z_threshold": args.z_threshold,
    }

    out_json = os.path.join(args.out, "purity_probe.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved probe report -> {out_json}")
    plots = make_plots(report, args.tiles, args.maps, args.primary_q, args.out)
    print(f"Saved plots -> {plots}")
    print_result(report)


if __name__ == "__main__":
    main()
