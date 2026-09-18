#!/usr/bin/env python3
"""Real SRAL track observations vs AMSR2 (independent passive-microwave SIC on the same grid): what errors do
track values actually have? Channels of sral_new_format/*.npy (8, 311, 225): 0 = altimeter SIC in %, 1 = ice class
{0 water, 1/2 ice, 3 unknown}, 7 = AMSR2 SIC (full grid). Reports, over all available days:
  * class agreement with AMSR2 (>=0.15 = ice), by AMSR2 bin
  * distribution of altimeter SIC - AMSR2 by AMSR2 bin (bias, sd, quantiles), share of class-3 (unknown)
  * along-track autocorrelation of the error (is the obs error white or coherent?)
  * track pixel counts per day and geometry stats (number of connected segments, mean segment length)
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
from scipy.ndimage import label

BINS = [0.0, 0.15, 0.5, 0.85, 0.95, 1.01]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sral-dir", default="/mnt/sciml/data_assimilation/sral_si/sral_new_format"); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--years", default="2019,2020,2021,2022")
    a = ap.parse_args(); years = a.years.split(",")
    files = [f for f in sorted(glob.glob(a.sral_dir + "/*.npy")) if Path(f).stem[:4] in years]
    acc = {i: {"n": 0, "sum": 0.0, "sq": 0.0, "vals": [], "ice_agree": 0, "water_agree": 0, "unknown": 0} for i in range(len(BINS) - 1)}
    npx = []; nseg = []; seglen = []; lag_num = np.zeros(4); lag_den = np.zeros(4)
    rng = np.random.default_rng(0)
    for f in files:
        arr = np.load(f); sic = arr[0] / 100.0; cls = np.round(arr[1]); amsr = arr[7]
        obs = np.isfinite(cls) & np.isfinite(amsr)
        if obs.sum() == 0:
            continue
        npx.append(int(obs.sum()))
        lab, n = label(obs, structure=np.ones((3, 3))); nseg.append(n); seglen.append(float(obs.sum() / max(n, 1)))
        b = np.clip(np.digitize(amsr[obs], BINS) - 1, 0, len(BINS) - 2)
        s = sic[obs]; c = cls[obs]; am = amsr[obs]
        for i in range(len(BINS) - 1):
            m = b == i
            if not m.any():
                continue
            e = s[m] - am[m]; fin = np.isfinite(e)
            acc[i]["n"] += int(m.sum()); acc[i]["sum"] += float(e[fin].sum()); acc[i]["sq"] += float((e[fin] ** 2).sum())
            if fin.sum() > 0:
                take = e[fin] if fin.sum() <= 200 else rng.choice(e[fin], 200, replace=False); acc[i]["vals"].extend(take.tolist())
            acc[i]["unknown"] += int((c[m] == 3).sum())
            acc[i]["ice_agree"] += int(((c[m] == 1) | (c[m] == 2)).sum()); acc[i]["water_agree"] += int((c[m] == 0).sum())
        # along-track error autocorrelation: horizontal/vertical neighbours both observed
        err = np.where(obs, sic - amsr, np.nan)
        for k, lag in enumerate((1, 2, 4, 8)):
            for e1, e2 in ((err[:, lag:], err[:, :-lag]), (err[lag:, :], err[:-lag, :])):
                ok = np.isfinite(e1) & np.isfinite(e2)
                if ok.sum() > 10:
                    x, y = e1[ok], e2[ok]; lag_num[k] += float(np.sum((x - x.mean()) * (y - y.mean()))); lag_den[k] += float(np.sqrt(np.sum((x - x.mean()) ** 2) * np.sum((y - y.mean()) ** 2)))
    out = {"n_days": len(npx), "track_px_per_day": {"mean": float(np.mean(npx)), "q10": float(np.percentile(npx, 10)), "q90": float(np.percentile(npx, 90))},
           "segments_per_day": float(np.mean(nseg)), "mean_segment_len_px": float(np.mean(seglen)),
           "error_autocorr_by_lag": {str(l): float(lag_num[k] / max(lag_den[k], 1e-9)) for k, l in enumerate((1, 2, 4, 8))}, "by_amsr2_bin": {}}
    print(f"days {len(npx)}, track px/day {np.mean(npx):.0f} [{np.percentile(npx,10):.0f}, {np.percentile(npx,90):.0f}], segments/day {np.mean(nseg):.0f}, mean segment {np.mean(seglen):.1f} px")
    print("error autocorrelation along track by lag:", out["error_autocorr_by_lag"])
    print(f"{'AMSR2 bin':12s} {'n':>8s} {'bias':>7s} {'sd':>6s} {'q05':>6s} {'q50':>6s} {'q95':>6s} {'P(class=ice)':>13s} {'P(class=water)':>15s} {'P(unknown)':>11s}")
    for i in range(len(BINS) - 1):
        d = acc[i]; n = max(d["n"], 1); v = np.array(d["vals"]) if d["vals"] else np.zeros(1)
        bias = d["sum"] / n; sd = float(np.sqrt(max(d["sq"] / n - bias ** 2, 0)))
        row = {"n": d["n"], "bias": bias, "sd": sd, "q05": float(np.percentile(v, 5)), "q50": float(np.percentile(v, 50)), "q95": float(np.percentile(v, 95)), "p_ice": d["ice_agree"] / n, "p_water": d["water_agree"] / n, "p_unknown": d["unknown"] / n}
        out["by_amsr2_bin"][f"[{BINS[i]:.2f},{BINS[i+1]:.2f})"] = row
        print(f"[{BINS[i]:.2f},{BINS[i+1]:.2f}) {d['n']:8d} {bias:+7.3f} {sd:6.3f} {row['q05']:+6.2f} {row['q50']:+6.2f} {row['q95']:+6.2f} {row['p_ice']:13.2f} {row['p_water']:15.2f} {row['p_unknown']:11.2f}")
    a.output.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
