#!/usr/bin/env python3
"""Upper bound for a per-date dispersion scale: oracle scaling of layered members about their mean.

For each saved case, members' = clip(mean + k_d (members - mean), 0, CAP) with k_d = rmse_d / spread_d (oracle, uses truth),
and with k = harmonic-of-doy fit on the oracle k_d (leave-one-out, ±1 purge). Reports fair CRPS (all water) for
layered / oracle-scaled / seasonal-scaled. If oracle gain is small, per-date scale is not worth modelling.
"""
from __future__ import annotations
import argparse, glob, json
from datetime import date
from pathlib import Path
import numpy as np

CAP = 0.9970703125


def fair_crps(m, y):
    M = m.shape[0]
    t1 = np.mean(np.abs(m - y[None]), axis=0)
    ms = np.sort(m, axis=0)
    # E|X-X'| via sorted members: sum_{i<j}(x_j-x_i) * 2 / (M(M-1))
    idx = np.arange(M)
    t2 = np.sum(ms * (2 * idx - M + 1)[:, None], axis=0) * 2 / (M * (M - 1))
    return float(np.mean(t1 - 0.5 * t2))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--members-glob", required=True); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cases = []
    for f in sorted(glob.glob(a.members_glob)):
        z = np.load(f); w = z["water"]; y = z["truth"].astype(np.float64)[w]; m = z["members"].astype(np.float64)[:, w]
        d = date.fromisoformat(Path(f).stem.replace("layered_", ""))
        mm = m.mean(0); rmse = np.sqrt(np.mean((mm - y) ** 2)); spread = np.sqrt(np.mean(m.var(0, ddof=1)))
        cases.append(dict(d=d, doy=d.timetuple().tm_yday, m=m, y=y, k=rmse / spread, crps=fair_crps(m, y)))
    doy = np.array([c["doy"] for c in cases]); logk = np.log([c["k"] for c in cases])
    X = np.stack([np.ones_like(doy, float), np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)], 1)
    out = []
    for i, c in enumerate(cases):
        keep = [j for j in range(len(cases)) if abs(j - i) > 1]
        beta = np.linalg.lstsq(X[keep], logk[keep], rcond=None)[0]
        k_seas = float(np.exp(X[i] @ beta))
        sc = lambda k: np.clip(c["m"].mean(0)[None] + k * (c["m"] - c["m"].mean(0)[None]), 0, CAP)
        out.append(dict(date=c["d"].isoformat(), k_oracle=c["k"], k_seasonal=k_seas, crps_layered=c["crps"], crps_oracle=fair_crps(sc(c["k"]), c["y"]), crps_seasonal=fair_crps(sc(k_seas), c["y"])))
    for key in ("crps_layered", "crps_oracle", "crps_seasonal"):
        print(key, round(float(np.mean([o[key] for o in out])), 5))
    print("dates improved seasonal:", sum(o["crps_seasonal"] < o["crps_layered"] for o in out), "/", len(out))
    print("k_oracle:", " ".join(f"{o['date'][5:]}:{o['k_oracle']:.2f}/{o['k_seasonal']:.2f}" for o in out))
    a.output.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
