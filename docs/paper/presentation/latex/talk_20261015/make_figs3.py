"""Figures of the final part 2 of the 15 Oct 2026 talk: OSG against the two baselines only - the standard monolith FMPE-amr-wide
(conditional flow matching, noise level as an input, trained down to sigma x .03) and guidance MMPS (FA-APF settings).
Usage: python make_figs3.py <scen_main> <scen_rank> <out> <hours_trainval.npy>
  precision3.png  CRPS against the observation-error multiplier (free cells, held-out stations) with the 'K with exact station values' level
  stations3.png   ensemble spread and error at the assimilated stations in units of the observation error
  ranks3.png      rank histograms (sigma x 1 and x .1; free cells, held-out stations, stations)
  corr3.png       correlated observation errors: FMPE-amr-wide relative to OSG with the full R
  maps3.png       one analysis at sigma x .1: msl increments over the North Atlantic and Europe
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SC, RK, OUT = (Path(p) for p in sys.argv[1:4]); OUT.mkdir(parents=True, exist_ok=True); HOURS = np.load(sys.argv[4])
plt.rcParams.update({'font.family': 'Arial', 'font.size': 16, 'axes.titlesize': 17, 'axes.labelsize': 16, 'legend.fontsize': 15,
                     'xtick.labelsize': 14.5, 'ytick.labelsize': 14.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#555', 'axes.linewidth': .8, 'axes.grid': True, 'grid.color': '#e8e8e8', 'grid.linewidth': .8,
                     'axes.axisbelow': True, 'savefig.dpi': 250, 'savefig.bbox': 'tight', 'figure.facecolor': 'white', 'legend.frameon': False})
C = {'osg': '#E4572E', 'amr': '#2E86AB', 'mmps': '#7768AE', 'level': '#444444'}
NAME = {'osg': 'OSG (наш)', 'amr': 'монолит FMPE (шум на входе)', 'mmps': 'наведение MMPS'}
SIG0 = np.array([100., 1., 1.5, 1.5, 1.5, 100.])                                     # nominal observation SD per channel (physical)
SIG = [('nom', 1.), ('sig05', .5), ('sig03', .3), ('sig02', .2), ('stress', .1), ('sig005', .05), ('sig002', .02), ('sig001', .01)]
PAT = {'osg': 'nhead_{}_big', 'amr': 'fmpe_amr_{}_wide', 'mmps': 'mmps_{}'}


def load(name, d=SC): return np.load(d / f'scen_{name}_pertime.npz')


def agg(name, reg, d=SC):
    z = load(name, d); r = z[reg]; c = r[:, :, 0].sum(0) / r[:, :, 3].sum(0); return float(np.mean(c / z['gsb']))


def relci(na, nb, reg='free', boot=2000):
    A, B = load(na), load(nb); g = A['gsb']
    f = lambda rr_, i: float(np.mean((rr_[i, :, 0].sum(0) / rr_[i, :, 3].sum(0)) / g))
    m = (np.datetime64('1959-01-01T00', 'h') + HOURS[A['times']].astype('timedelta64[h]')).astype('datetime64[M]').astype(int)
    bl = [np.flatnonzero(m == u) for u in np.unique(m)]; rng_ = np.random.default_rng(0); full = np.arange(len(m))
    v = [f(A[reg], i) / f(B[reg], i) - 1 for i in (np.concatenate([bl[j] for j in rng_.integers(0, len(bl), len(bl))]) for _ in range(boot))]
    return 100 * (f(A[reg], full) / f(B[reg], full) - 1), 100 * np.quantile(v, .025), 100 * np.quantile(v, .975)


def present(key):
    xs = []
    for sc, v in SIG:
        if (SC / f'scen_{PAT[key].format(sc)}_pertime.npz').exists(): xs.append((sc, v))
    return xs


# 1. precision sweep, as the gap to the 'K with exact station values' level (0 = the level)
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.1))
for ax, reg, title in ((axs[0], 'free', 'клетки без станций'), (axs[1], 'held', 'отложенные станции')):
    lim = agg('oracle_nom', reg); lo_ = 0.
    ax.axvspan(.0085, .03, color='#f1f2f4', zorder=0); ax.text(.0155, .985, 'вне\nобучения\nмонолита', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=12.5, color='#777')
    for key in ('amr', 'osg'):
        pts = present(key); xs = [v for _, v in pts]; ys = [100 * (agg(PAT[key].format(sc), reg) / lim - 1) for sc, _ in pts]; lo_ = min(lo_, min(ys))
        ax.plot(xs, ys, '-o', color=C[key], lw=2.6, ms=6, label=NAME[key], zorder=3)
        if key in ('amr', 'osg'):
            ax.annotate(f'{ys[-1]:+.1f}%', (xs[-1], ys[-1]), xytext=(-8, 9), textcoords='offset points', ha='right', color=C[key], fontsize=14, fontweight='bold')
    ax.axhline(0, color=C['level'], lw=1.8, ls=(0, (2, 2)), label='0 = предел: $K$ с точными значениями в станциях', zorder=2)
    ax.set_ylim(min(-1, lo_ - .5), None)
    ax.set_xscale('log'); ax.invert_xaxis(); ax.set_xticks([v for _, v in SIG]); ax.set_xticklabels(['1', '.5', '.3', '.2', '.1', '.05', '.02', '.01'])
    ax.minorticks_off(); ax.set_xlabel('ошибка наблюдений, σ × …'); ax.set_title(title, loc='left')
axs[0].set_ylabel('CRPS хуже предела, %')
h, l = axs[0].get_legend_handles_labels(); fig.legend(h[::-1], l[::-1], loc='lower center', ncol=2, bbox_to_anchor=(.5, -.1))
fig.tight_layout(rect=(0, .12, 1, 1)); fig.savefig(OUT / 'precision3.png'); plt.close(fig)

# 2. stations: analysis spread and error at the assimilated stations in units of the NOMINAL observation error sigma_0 (linear, from 0);
#    the dashed line is the error of the instrument itself, sigma / sigma_0 = lambda
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True)
lam = np.array([v for _, v in SIG])
for ax, k_, title in ((axs[0], 2, 'разброс ансамбля в станции'), (axs[1], 1, 'ошибка анализа в станции')):
    lg = np.geomspace(1, .01, 200); ax.plot(lg, lg, color=C['level'], lw=1.8, ls=(0, (2, 2)), label='ошибка самого прибора', zorder=2)
    for key in ('amr', 'osg'):
        xs, ys = [], []
        for sc, v in present(key):
            r = load(PAT[key].format(sc))['active']; m2 = r[:, :, k_].sum(0) / r[:, :, 3].sum(0)
            xs.append(v); ys.append(float(np.mean(np.sqrt(m2) / SIG0)))
        ax.plot(xs, ys, '-o', color=C[key], lw=2.6, ms=6, label=NAME[key], zorder=3)
        if key == 'amr': ax.annotate(f'монолит: «пол» ≈ {ys[-1]:.2f}, при σ × .01\nв {ys[-1] / xs[-1]:.0f} раз хуже прибора', xy=(xs[-1], ys[-1]), xytext=(.03, .5),
                                     fontsize=13, color=C['amr'], ha='center', va='bottom', arrowprops=dict(arrowstyle='->', color=C['amr'], lw=1.2))
    ax.set_xscale('log'); ax.invert_xaxis(); ax.set_xticks([v for _, v in SIG]); ax.set_xticklabels(['1', '.5', '.3', '.2', '.1', '.05', '.02', '.01'])
    ax.set_ylim(0, 1.05); ax.minorticks_off(); ax.set_xlabel('ошибка наблюдений, σ × …'); ax.set_title(title, loc='left')
axs[0].set_ylabel('в единицах обычной\nошибки прибора $\\sigma_0$')
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(.5, -.06))
fig.tight_layout(rect=(0, .1, 1, 1)); fig.savefig(OUT / 'stations3.png'); plt.close(fig)

# 3. rank histograms (channel mean); M + 1 = 17 bins, step lines; OSG vs the standard monolith
RF = {'osg': 'scen_nhead_{}_big.json', 'amr': 'scen_fmpe_amr_{}_wide.json'}
if all((RK / RF[k].format(sc)).exists() for k in RF for sc in ('nom', 'stress')):
    REG = [('free', 'клетки без станций'), ('held', 'отложенные станции'), ('active', 'сами станции')]
    SCR = [('nom', 'обычная ошибка\n(σ × 1)'), ('stress', 'точные наблюдения\n(σ × .1)')]
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 6.4), sharex=True, sharey=True)
    for i, (sc, sct) in enumerate(SCR):
        for j, (reg, rt) in enumerate(REG):
            ax = axs[i, j]
            for k, key in enumerate(('amr', 'osg')):
                R = json.loads((RK / RF[key].format(sc)).read_text())['regions'][reg]
                H = np.array([v for v in R['rank_hist'].values() if v is not None]).mean(0); x = np.arange(len(H) + 1) - .5
                ax.stairs(H, x, color=C[key], lw=3.2 if key == 'osg' else 2.6, label=NAME[key], zorder=3 + (key == 'osg'))
                ssr = np.mean([v for v in R['ssr'].values() if v is not None])
                ax.text(.97, .95 - .12 * k, f'SSR {ssr:.2f}', transform=ax.transAxes, ha='right', va='top', fontsize=14, color=C[key], fontweight='bold')
            ax.axhline(1, color=C['level'], lw=1.3, ls=(0, (2, 2)), zorder=2); ax.grid(axis='x', visible=False)
            if i == 0: ax.set_title(rt, loc='left')
            if j == 0: ax.set_ylabel(sct)
            if i == 1: ax.set_xlabel('ранг правды среди 16 членов')
    axs[0, 0].set_ylim(0, 2.0); axs[0, 0].set_xticks([0, 4, 8, 12, 16])
    h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h[::-1], l[::-1], loc='lower center', ncol=2, bbox_to_anchor=(.5, -.05))
    fig.tight_layout(rect=(0, .05, 1, 1)); fig.savefig(OUT / 'ranks3.png'); plt.close(fig)

# 4. correlated errors: FMPE-amr-wide relative to OSG with the full R, and the station rank histograms at gamma .5, 670 km
SCN = [('nom', 'независимые\nошибки'), ('cg50l2', 'γ = .5\n670 км'), ('cg50l4', 'γ = .5\n1340 км')]
rk_ok = (RK / 'scen_nhead_cg50l2_big.json').exists() and (RK / 'scen_fmpe_amr_cg50l2_wide.json').exists()
if all((SC / f'scen_fmpe_amr_{sc}_wide_pertime.npz').exists() for sc, _ in SCN):
    fig, axs = plt.subplots(1, 3 if rk_ok else 2, figsize=(15.5 if rk_ok else 10.5, 4.1), gridspec_kw={'width_ratios': [1, 1, .85] if rk_ok else [1, 1]})
    for ax, reg, title in ((axs[0], 'free', 'клетки без станций'), (axs[1], 'active', 'станции')):
        for i, (sc, _) in enumerate(SCN):
            d, lo, hi = relci(f'fmpe_amr_{sc}_wide', f'nhead_{sc}_big', reg)
            ax.bar(i, d, width=.55, color=C['amr'], zorder=2)
            ax.errorbar([i], [d], yerr=[[d - lo], [hi - d]], color='#333', capsize=4, lw=1.2, zorder=3)
            off = (abs(hi - lo) / 2 + (.12 if reg == 'free' else 1.2)) * (1 if d >= 0 else -1)
            ax.text(i, d + off, f'{d:+.1f}%', ha='center', va='bottom' if d >= 0 else 'top', fontsize=14, color=C['amr'], fontweight='bold')
        ax.axhline(0, color=C['level'], lw=1.2); ax.set_xticks(range(len(SCN))); ax.set_xticklabels([s_[1] for s_ in SCN], fontsize=13); ax.grid(axis='x', visible=False)
        ax.set_title(title, loc='left'); ax.margins(y=.18)
    axs[0].set_ylabel('монолит хуже OSG, %')
    if rk_ok:
        ax = axs[2]
        for key, f in (('amr', 'scen_fmpe_amr_cg50l2_wide.json'), ('osg', 'scen_nhead_cg50l2_big.json')):
            R = json.loads((RK / f).read_text())['regions']['active']; H = np.array([v for v in R['rank_hist'].values() if v is not None]).mean(0)
            ax.stairs(H, np.arange(len(H) + 1) - .5, color=C[key], lw=3 if key == 'osg' else 2.4, label=NAME[key])
        ax.axhline(1, color=C['level'], lw=1.3, ls=(0, (2, 2))); ax.set_ylim(0, None); ax.set_xticks([0, 4, 8, 12, 16]); ax.grid(axis='x', visible=False)
        ax.set_title('ранги в станциях, γ = .5, 670 км', loc='left', fontsize=15); ax.set_xlabel('ранг правды среди 16 членов'); ax.legend(loc='lower center', fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / 'corr3.png'); plt.close(fig)

# 5. maps: one analysis at sigma x .1 (2019-01-01 00 UTC), msl increments, OSG vs the standard monolith
mo, ma = RK / 'members_nhead_stress_big.npz', RK / 'members_fmpe_amr_stress_wide.npz'
if mo.exists() and ma.exists() and (RK / 'era5_static.npz').exists():
    A, B, ST = np.load(mo), np.load(ma), np.load(RK / 'era5_static.npz')
    lat, lon = ST['lat'], ST['lon']; lon_c = np.where(lon >= 180, lon - 360, lon); order = np.argsort(lon_c); lon_s = lon_c[order]
    rows = np.flatnonzero((lat >= 30) & (lat <= 72)); cols = np.flatnonzero((lon_s >= -45) & (lon_s <= 40))
    def crop(f): g = f[..., order]; return g[..., rows, :][..., cols][..., ::-1, :]
    la, lo = lat[rows][::-1], lon_s[cols]; ext = [lo[0] - .75, lo[-1] + .75, la[-1] - .75, la[0] + .75]
    lsm = crop(ST['lsm']); c, scl = 5, 100.
    inc_t = crop(A['x'][c] - A['xb'][c]) / scl; act = crop(A['active'][c]); v = float(np.nanpercentile(np.abs(inc_t), 99))
    sds = {nm: crop(Z['X'][:, c].std(0, ddof=1)) / scl for Z, nm in ((A, 'OSG'), (B, 'монолит'))}; vs = float(max(np.nanpercentile(x_, 99) for x_ in sds.values()))
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.2), constrained_layout=True)
    def show(ax, F, title, cm, lim, dots=False):
        im = ax.imshow(F, extent=ext, cmap=cm, vmin=lim[0], vmax=lim[1], aspect=1 / np.cos(np.deg2rad(51)), interpolation='nearest')
        ax.contour(lo, la, lsm, levels=[.5], colors='k', linewidths=.7)
        if dots: yy, xx = np.nonzero(act); ax.scatter(lo[xx], la[yy], s=2.5, color='k', lw=0, alpha=.6)
        ax.set_title(title, fontsize=15, loc='left'); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); return im
    for i_, (Z, nm) in enumerate(((A, 'OSG'), (B, 'монолит'))):
        X = Z['X'][:, c]
        if i_ == 0: im0 = show(axs[0, 0], inc_t, 'правда − фон', 'RdBu_r', (-v, v))
        else: show(axs[1, 0], np.where(act, inc_t, np.nan), 'наблюдения (станции)', 'RdBu_r', (-v, v))
        show(axs[i_, 1], crop(X.mean(0) - Z['xb'][c]) / scl, f'{nm}: среднее − фон', 'RdBu_r', (-v, v))
        show(axs[i_, 2], crop(X[0] - Z['xb'][c]) / scl, f'{nm}: один член − фон', 'RdBu_r', (-v, v))
        im3 = show(axs[i_, 3], sds[nm], f'{nm}: разброс', 'magma_r', (0, vs), dots=True)
    fig.colorbar(im0, ax=axs[:, :3], shrink=.7, label='приращение давления, гПа', pad=.01)
    fig.colorbar(im3, ax=axs[:, 3], shrink=.7, label='SD, гПа', pad=.01)
    fig.savefig(OUT / 'maps3.png'); plt.close(fig)
print('saved', sorted(p.name for p in OUT.iterdir() if p.name.endswith('3.png')))
