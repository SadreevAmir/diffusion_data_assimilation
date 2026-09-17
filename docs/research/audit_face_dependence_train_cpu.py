#!/usr/bin/env python3
"""Train-only CPU falsifier for a *local* face-aware refinement kernel.

Test A (decisive).  A face-aware refinement kernel assigns every child cell a
face z in {0: open, 1: interior, 2: cap}.  If the kernel is conditioned only on
the parent field and the initial state (as a convolutional backbone would be)
and children are sampled conditionally independently, the field of faces is a
*product law*.  We fit that product law empirically on train years
2016--2019 with a generous feature set (parent bin, neighbour-parent bin, d0
child bin, child position) and sample it on 2020 anchors under *truth*
parents.  If the sampled texture in consolidated ice is far rougher than truth
(salt-and-pepper faces), a local kernel cannot fix the observed grain no matter
how it is trained.

Test B (reference).  An "ideal" ensemble made of truth fields at the same lead
shifted by +-1..8 days gives absolute reliability/texture statistics that a
genuinely good forecast ensemble would attain on this archive.  It is reported
for comparison with the baseline numbers, not as a forecast.

Only train dates 2016--2020 (hour 23) are read.  Future truth enters only as
the object being measured, never as conditioning of the sampled kernel.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label

ROOT = Path("/mnt/sciml/data_assimilation")
PREDS = ROOT / "da_arctic_2015-2024_v0.1/preds"
WATER = ~np.load(ROOT / "land_mask.npy")
HOUR = 23
LEADS = (3, 6, 9)
CAP = np.float32(0.9970703)
STABLE = 0.95
EDGE_THRESHOLD = 0.15
TUBE_PX = 8
RESERVOIR = 512


def path(value: date) -> Path:
    if not date(2016, 1, 1) <= value <= date(2020, 12, 31):
        raise ValueError(f"outside train window: {value}")
    return PREDS / f"ocean+atmosphere_24_{value.isoformat()}.npy"


def sic(value: date) -> np.ndarray:
    field = np.asarray(np.load(path(value), mmap_mode="r")[0, HOUR], dtype=np.float32)
    if np.isinf(field[WATER]).any():
        raise ValueError(f"Inf in {path(value)}")
    field = np.where(np.isnan(field), 0.0, field)
    return np.where(WATER, field, 0.0).astype(np.float32)


def faces(field: np.ndarray) -> np.ndarray:
    z = np.ones(field.shape, dtype=np.int8)
    z[field <= 0.0] = 0
    z[field >= CAP] = 2
    return z


def anchors_for(years: tuple[int, ...], days: tuple[int, ...]) -> list[date]:
    out = []
    for year in years:
        for month in range(1, 13):
            for day in days:
                try:
                    anchor = date(year, month, day)
                except ValueError:
                    continue
                if anchor + timedelta(days=max(LEADS)) > date(2020, 12, 31):
                    continue
                if all(path(anchor + timedelta(days=lead)).is_file() for lead in (0, *LEADS)):
                    out.append(anchor)
    return out


# ----------------------------------------------------------------------------
# Feature construction for the local kernel
# ----------------------------------------------------------------------------

PARENT_EDGES = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.97], dtype=np.float32)
NEIGHBOUR_EDGES = np.array([0.2, 0.5, 0.8, 0.97], dtype=np.float32)
D0_EDGES = np.array([0.15, 0.5, 0.9, 0.95], dtype=np.float32)


def pad_to(field: np.ndarray, factor: int, fill=0) -> np.ndarray:
    h, w = field.shape
    ph, pw = (-h) % factor, (-w) % factor
    return np.pad(field, ((0, ph), (0, pw)), constant_values=fill)


def block_reduce(field: np.ndarray, water: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    """Water-weighted block mean (parent) and water count per block."""
    f = pad_to(field * water, factor)
    w = pad_to(water.astype(np.float32), factor)
    hb, wb = f.shape[0] // factor, f.shape[1] // factor
    fsum = f.reshape(hb, factor, wb, factor).sum(axis=(1, 3))
    wsum = w.reshape(hb, factor, wb, factor).sum(axis=(1, 3))
    parent = np.where(wsum > 0, fsum / np.maximum(wsum, 1.0), 0.0).astype(np.float32)
    return parent, wsum


def parent_bin(parent: np.ndarray, wsum: np.ndarray, full: np.ndarray) -> np.ndarray:
    frac = parent / CAP
    b = 1 + np.searchsorted(PARENT_EDGES, frac, side="right")  # 1..7
    b = np.where(frac <= 0.0, 0, b)
    b = np.where(full, 8, b)  # every water child exactly at cap
    b = np.where(wsum <= 0, 9, b)  # no water
    return b.astype(np.int16)


def neighbour_bin(parent: np.ndarray, wsum: np.ndarray) -> np.ndarray:
    pw = np.pad(parent * wsum, 1)
    ww = np.pad(wsum, 1)
    s = np.zeros_like(parent)
    c = np.zeros_like(wsum)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            s += pw[1 + dy : 1 + dy + parent.shape[0], 1 + dx : 1 + dx + parent.shape[1]]
            c += ww[1 + dy : 1 + dy + parent.shape[0], 1 + dx : 1 + dx + parent.shape[1]]
    frac = np.where(c > 0, s / np.maximum(c, 1.0), 0.0) / CAP
    b = 1 + np.searchsorted(NEIGHBOUR_EDGES, frac, side="right")  # 1..5
    b = np.where(frac <= 0.0, 0, b)
    b = np.where(c <= 0, 6, b)
    return b.astype(np.int16)


def d0_bin(initial: np.ndarray) -> np.ndarray:
    b = 1 + np.searchsorted(D0_EDGES, initial, side="right")  # 1..5 interior bins
    b = np.where(initial <= 0.0, 0, b)
    b = np.where(initial >= CAP, 6, b)
    return b.astype(np.int16)


def child_features(future: np.ndarray, initial: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-child integer feature id (native grid) and the padded water mask."""
    water = pad_to(WATER, factor, fill=False)
    fut = pad_to(future, factor)
    ini = pad_to(initial, factor)
    parent, wsum = block_reduce(fut, water, factor)
    zf = faces(fut)
    capcount = pad_to(((zf == 2) & water).astype(np.float32), factor)
    hb, wb = parent.shape
    capsum = capcount.reshape(hb, factor, wb, factor).sum(axis=(1, 3))
    full = (wsum > 0) & (capsum >= wsum)
    pb = parent_bin(parent, wsum, full)
    nb = neighbour_bin(parent, wsum)
    pb_native = np.repeat(np.repeat(pb, factor, 0), factor, 1)
    nb_native = np.repeat(np.repeat(nb, factor, 0), factor, 1)
    yy, xx = np.mgrid[0 : fut.shape[0], 0 : fut.shape[1]]
    pos = (yy % factor) * factor + (xx % factor)
    db = d0_bin(ini)
    fid = (((pb_native.astype(np.int64) * 7 + nb_native) * 7 + db) * (factor * factor) + pos)
    return fid, water


# ----------------------------------------------------------------------------
# Texture metrics
# ----------------------------------------------------------------------------

def neighbour_l1(field: np.ndarray, region: np.ndarray) -> float:
    h = region[:, 1:] & region[:, :-1]
    v = region[1:, :] & region[:-1, :]
    num = np.abs(field[:, 1:] - field[:, :-1])[h].sum() + np.abs(field[1:, :] - field[:-1, :])[v].sum()
    den = h.sum() + v.sum()
    return float(num / den) if den > 0 else float("nan")


def isolated_fraction(z: np.ndarray, region: np.ndarray) -> float:
    zp = np.pad(z, 1, constant_values=-1)
    rp = np.pad(region, 1, constant_values=False)
    diff_all = np.ones(z.shape, dtype=bool)
    has_nb = np.zeros(z.shape, dtype=bool)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nz = zp[1 + dy : 1 + dy + z.shape[0], 1 + dx : 1 + dx + z.shape[1]]
        nr = rp[1 + dy : 1 + dy + z.shape[0], 1 + dx : 1 + dx + z.shape[1]]
        diff_all &= (~nr) | (nz != z)
        has_nb |= nr
    sel = region & has_nb
    return float((diff_all & sel).sum() / max(int(sel.sum()), 1))


def adjacent_agreement(z: np.ndarray, region: np.ndarray) -> float:
    h = region[:, 1:] & region[:, :-1]
    v = region[1:, :] & region[:-1, :]
    agree = (z[:, 1:] == z[:, :-1])[h].sum() + (z[1:, :] == z[:-1, :])[v].sum()
    den = h.sum() + v.sum()
    return float(agree / den) if den > 0 else float("nan")


def cap_components(z: np.ndarray, water: np.ndarray) -> int:
    _, n = label((z == 2) & water)
    return int(n)


def edge_tube(initial: np.ndarray) -> np.ndarray:
    ice = (initial >= EDGE_THRESHOLD) & WATER
    open_water = (~ice) & WATER
    edge = (ice & binary_dilation(open_water)) | (open_water & binary_dilation(ice))
    if not edge.any():
        return np.zeros(initial.shape, dtype=bool)
    return (distance_transform_edt(~edge) <= TUBE_PX) & WATER


# ----------------------------------------------------------------------------
# Test A
# ----------------------------------------------------------------------------

def fit_local_kernel(anchors: list[date], factor: int, rng: np.random.Generator) -> dict:
    counts: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    reservoir: dict[int, list] = defaultdict(list)
    seen: dict[int, int] = defaultdict(int)
    for anchor in anchors:
        initial = sic(anchor)
        for lead in LEADS:
            future = sic(anchor + timedelta(days=lead))
            fid, water = child_features(future, initial, factor)
            z = faces(pad_to(future, factor))
            fw = fid[water]
            zw = z[water]
            vals = pad_to(future, factor)[water]
            for key in np.unique(fw):
                m = fw == key
                counts[int(key)] += np.bincount(zw[m], minlength=3)
                interior = vals[m & (zw == 1)]
                if interior.size:
                    r = reservoir[int(key)]
                    room = RESERVOIR - len(r)
                    if room > 0:
                        r.extend(float(v) for v in interior[:room])
                    else:
                        k = min(64, int(interior.size))
                        pick = rng.choice(interior, size=k, replace=False)
                        slots = rng.integers(0, RESERVOIR, size=k)
                        for j, v in zip(slots, pick):
                            r[int(j)] = float(v)
    return {"counts": dict(counts), "reservoir": {k: np.asarray(v, dtype=np.float32) for k, v in reservoir.items()}}


def sample_product(kernel: dict, fid: np.ndarray, water: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(fid.shape, dtype=np.float32)
    fw = fid[water]
    zs = np.ones(fw.shape, dtype=np.int8)
    vs = np.zeros(fw.shape, dtype=np.float32)
    for key in np.unique(fw):
        m = fw == key
        c = kernel["counts"].get(int(key))
        n = int(m.sum())
        if c is None or c.sum() == 0:
            p = np.array([1 / 3, 1 / 3, 1 / 3])
        else:
            p = c / c.sum()
        z = rng.choice(3, size=n, p=p).astype(np.int8)
        zs[m] = z
        res = kernel["reservoir"].get(int(key))
        v = np.full(n, 0.5, dtype=np.float32)
        if res is not None and res.size:
            v = res[rng.integers(0, res.size, size=n)]
        vs[m] = np.where(z == 0, 0.0, np.where(z == 2, CAP, v))
    out[water] = vs
    return out


def test_a(fit_anchors: list[date], eval_anchors: list[date], factor: int, draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    kernel = fit_local_kernel(fit_anchors, factor, rng)
    acc = {lead: defaultdict(list) for lead in LEADS}
    panel = None
    for anchor in eval_anchors:
        initial = sic(anchor)
        tube = pad_to(edge_tube(initial), factor, fill=False)
        for lead in LEADS:
            future = sic(anchor + timedelta(days=lead))
            fid, water = child_features(future, initial, factor)
            fut = pad_to(future, factor)
            zt = faces(fut)
            stable = (fut >= STABLE) & water
            capmask = (zt == 2) & water
            a = acc[lead]
            a["truth_nl1_stable"].append(neighbour_l1(fut, stable))
            a["truth_nl1_tube"].append(neighbour_l1(fut, tube))
            a["truth_isolated_stable"].append(isolated_fraction(zt, stable))
            a["truth_agree_stable"].append(adjacent_agreement(zt, stable))
            a["truth_cap_components"].append(cap_components(zt, water))
            ini = pad_to(initial, factor)
            a["persistence_nl1_stable"].append(neighbour_l1(ini, stable))
            for d in range(draws):
                s = sample_product(kernel, fid, water, rng)
                zs = faces(s)
                a["product_nl1_stable"].append(neighbour_l1(s, stable))
                a["product_nl1_tube"].append(neighbour_l1(s, tube))
                a["product_isolated_stable"].append(isolated_fraction(zs, stable))
                a["product_agree_stable"].append(adjacent_agreement(zs, stable))
                a["product_cap_components"].append(cap_components(zs, water))
                a["product_cap_retention"].append(float(((zs == 2) & capmask).sum() / max(int(capmask.sum()), 1)))
                a["product_rmse_water"].append(float(np.sqrt(np.mean((s[water] - fut[water]) ** 2))))
                if panel is None and anchor.month in (2, 3) and lead == 3 and d == 0:
                    panel = {"anchor": anchor.isoformat(), "lead": lead, "truth": fut, "initial": ini, "sample": s, "sample2": sample_product(kernel, fid, water, rng)}
    summary = {}
    for lead in LEADS:
        a = acc[lead]
        m = {k: float(np.nanmean(v)) for k, v in a.items()}
        m["ratio_nl1_stable_product_over_truth"] = m["product_nl1_stable"] / m["truth_nl1_stable"]
        m["ratio_nl1_tube_product_over_truth"] = m["product_nl1_tube"] / m["truth_nl1_tube"]
        m["ratio_isolated_stable_product_over_truth"] = m["product_isolated_stable"] / max(m["truth_isolated_stable"], 1e-9)
        m["ratio_cap_components_product_over_truth"] = m["product_cap_components"] / max(m["truth_cap_components"], 1e-9)
        summary[f"lead_{lead}"] = m
    gates = {
        "local_kernel_sufficient_nl1_ratio_le_1p5_all_leads": all(summary[f"lead_{l}"]["ratio_nl1_stable_product_over_truth"] <= 1.5 for l in LEADS),
        "local_kernel_isolated_le_2x_truth_all_leads": all(summary[f"lead_{l}"]["ratio_isolated_stable_product_over_truth"] <= 2.0 for l in LEADS),
        "local_kernel_rejected_nl1_ratio_gt_3_any_lead": any(summary[f"lead_{l}"]["ratio_nl1_stable_product_over_truth"] > 3.0 for l in LEADS),
    }
    return {"factor": factor, "num_feature_cells": len(kernel["counts"]), "summary": summary, "gates": gates, "_panel": panel}


# ----------------------------------------------------------------------------
# Test B: ideal date-shifted truth ensemble
# ----------------------------------------------------------------------------

def fair_crps(members: np.ndarray, y: np.ndarray) -> float:
    m = members.shape[0]
    term1 = np.abs(members - y[None]).mean(axis=0)
    diff = 0.0
    for i in range(m):
        diff = diff + np.abs(members[i][None] - members).sum(axis=0)
    term2 = diff / (2 * m * (m - 1))
    return float((term1 - term2).mean())


def randomized_rank_tv(members: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    m = members.shape[0]
    below = (members < y[None]).sum(axis=0)
    ties = (members == y[None]).sum(axis=0)
    extra = np.floor(rng.random(y.shape) * (ties + 1)).astype(np.int64)
    rank = below + extra
    hist = np.bincount(rank.ravel(), minlength=m + 1).astype(np.float64)
    hist /= hist.sum()
    return float(0.5 * np.abs(hist - 1.0 / (m + 1)).sum())


def test_b(eval_anchors: list[date], seed: int) -> dict:
    rng = np.random.default_rng(seed + 7)
    out = {}
    shifts = [k for k in range(-8, 9) if k != 0]
    for lead in LEADS:
        rows = defaultdict(list)
        for anchor in eval_anchors:
            target_day = anchor + timedelta(days=lead)
            dates = [target_day + timedelta(days=k) for k in shifts]
            if not all(date(2016, 1, 1) <= d <= date(2020, 12, 31) and path(d).is_file() for d in dates):
                continue
            y = sic(target_day)
            members = np.stack([sic(d) for d in dates])
            w = WATER
            mw = members[:, w]
            yw = y[w]
            mean = mw.mean(axis=0)
            rmse = float(np.sqrt(np.mean((mean - yw) ** 2)))
            spread = float(np.sqrt(mw.var(axis=0, ddof=1).mean()))
            rows["fair_crps"].append(fair_crps(mw, yw))
            rows["rmse"].append(rmse)
            rows["spread"].append(spread)
            rows["rank_tv_pooled"].append(randomized_rank_tv(mw, yw, rng))
            zt = faces(y)[w]
            for name, sel in (("open", zt == 0), ("interior", zt == 1), ("cap", zt == 2)):
                if sel.sum() > 100:
                    rows[f"rank_tv_{name}"].append(randomized_rank_tv(mw[:, sel], yw[sel], rng))
            stable = (y >= STABLE) & w
            rows["truth_nl1_stable"].append(neighbour_l1(y, stable))
            rows["member_nl1_stable"].append(float(np.mean([neighbour_l1(members[i], stable) for i in range(len(shifts))])))
            capmask = (y >= CAP) & w
            rows["cap_retention"].append(float(((members >= CAP) & capmask[None]).sum() / max(int(capmask.sum()) * len(shifts), 1)))
        summary = {k: float(np.mean(v)) for k, v in rows.items()}
        summary["spread_over_rmse"] = summary["spread"] / summary["rmse"]
        summary["ratio_nl1_stable_member_over_truth"] = summary["member_nl1_stable"] / summary["truth_nl1_stable"]
        summary["num_cases"] = len(rows["rmse"])
        out[f"lead_{lead}"] = summary
    return {"members": len(shifts), "shift_days": shifts, "summary": out}


def save_panel(panel: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fields = [("initial d0", panel["initial"]), ("truth +%d" % panel["lead"], panel["truth"]), ("product sample 1", panel["sample"]), ("product sample 2", panel["sample2"])]
    water = pad_to(WATER, 1, False)
    water = np.pad(water, ((0, panel["truth"].shape[0] - water.shape[0]), (0, panel["truth"].shape[1] - water.shape[1])), constant_values=False)
    for ax, (name, f) in zip(axes[0], fields):
        ax.imshow(np.where(water, f, np.nan), vmin=0, vmax=1, cmap="Blues_r")
        ax.set_title(f"{name} {panel['anchor']}")
        ax.axis("off")
    h, w = panel["truth"].shape
    y0, x0 = h // 3, w // 3
    for ax, (name, f) in zip(axes[1], fields):
        ax.imshow(f[y0 : y0 + 60, x0 : x0 + 60], vmin=0.9, vmax=1.0, cmap="Blues_r")
        ax.set_title(f"zoom {name} (0.9..1)")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--factors", type=int, nargs="+", default=(4, 2))
    parser.add_argument("--fit-days", type=int, nargs="+", default=(2, 6, 10, 14, 18, 22, 26, 30))
    parser.add_argument("--eval-days", type=int, nargs="+", default=(4, 8, 12, 16, 20, 24, 28))
    parser.add_argument("--skip-test-b", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fit_anchors = anchors_for((2016, 2017, 2018, 2019), tuple(args.fit_days))
    eval_anchors = anchors_for((2020,), tuple(args.eval_days))
    result = {
        "schema_version": "face_dependence_train_cpu_audit_v1",
        "claim_boundary": "Train-only 2016--2020 CPU audit. Test A samples an empirically fitted product law of child faces under truth parents; it bounds what any locally conditioned refinement kernel can reproduce. Test B is a date-shifted truth ensemble, a reference for attainable statistics, not a forecast.",
        "config": {"seed": args.seed, "draws": args.draws, "factors": list(args.factors), "fit_anchors": len(fit_anchors), "eval_anchors": len(eval_anchors), "cap": float(CAP), "stable_threshold": STABLE, "tube_px": TUBE_PX},
        "test_a": {},
    }
    for factor in args.factors:
        r = test_a(fit_anchors, eval_anchors, factor, args.draws, args.seed + factor)
        panel = r.pop("_panel")
        if panel is not None:
            save_panel(panel, args.output_dir / f"product_law_panel_factor{factor}.png")
        result["test_a"][f"factor_{factor}"] = r
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, default=float))
        print(json.dumps({f"factor_{factor}": {k: v for k, v in r["summary"].items()}, "gates": r["gates"]}, indent=1, default=float), flush=True)
    if not args.skip_test_b:
        result["test_b"] = test_b(eval_anchors, args.seed)
        print(json.dumps(result["test_b"]["summary"], indent=1, default=float), flush=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, default=float))
    print("done", args.output_dir / "result.json")


if __name__ == "__main__":
    main()
