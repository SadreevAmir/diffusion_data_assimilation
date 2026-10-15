"""Clean figures for the reworked part 2 of the 15 Oct 2026 talk: only OSG against OA-FM (the strongest monolith), one style.
Usage: python make_figs2.py <scen_main> <chi_audit> <toy2> <out> <hours_trainval.npy> [<scen_rank>]
  osg_idea.png     how OSG works on a 1-D Gaussian-process example (illustration, no model): head prior at the stations, exact Bayes, fibre
  precision2.png   free / held-out CRPS against the noise multiplier, OSG vs OA-FM and the 'K with exact station values' level
  chi2.png         response test chi for the stations and their hidden neighbours, sigma x 1 and x .1
  corr2.png        correlated observation errors: OA-FM and OSG with a diagonal R relative to OSG with the full R
  toy2.png         the exact 2-D coupling experiment, reduced to the OA-FM-like and OSG-like coordinates
  ranks2.png       rank histograms (needs <scen_rank>: eval_scen_era5.py --rank-hist)
  maps2.png        one analysis at sigma x .1 (needs the member dumps in <scen_rank>)
"""
import json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

SC, CH, TOY, OUT = (Path(p) for p in sys.argv[1:5]); OUT.mkdir(parents=True, exist_ok=True); HOURS = np.load(sys.argv[5])
RK = Path(sys.argv[6]) if len(sys.argv) > 6 else None
plt.rcParams.update({'font.family': 'Arial', 'font.size': 16, 'axes.titlesize': 17, 'axes.labelsize': 16, 'legend.fontsize': 15,
                     'xtick.labelsize': 14.5, 'ytick.labelsize': 14.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#555', 'axes.linewidth': .8, 'axes.grid': True, 'grid.color': '#e8e8e8', 'grid.linewidth': .8,
                     'axes.axisbelow': True, 'savefig.dpi': 250, 'savefig.bbox': 'tight', 'figure.facecolor': 'white', 'legend.frameon': False})
C = {'osg': '#E4572E', 'oafm': '#2E86AB', 'level': '#444444', 'diag': '#F2A541', 'truth': '#222222', 'prior': '#B8BEC6', 'obs': '#111111', 'psamp': '#6B7A8F'}
INK = '#4A5568'


def agg(name, reg):
    z = np.load(SC / f'scen_{name}_pertime.npz'); r = z[reg]; c = r[:, :, 0].sum(0) / r[:, :, 3].sum(0); return float(np.mean(c / z['gsb']))


def relci(na, nb, reg='free', boot=2000):
    """100 (CRPS_a / CRPS_b - 1) on the mean over channels of CRPS / gsb, 95% month-block bootstrap."""
    A = np.load(SC / f'scen_{na}_pertime.npz'); B = np.load(SC / f'scen_{nb}_pertime.npz'); g = A['gsb']
    f = lambda rr_, i: float(np.mean((rr_[i, :, 0].sum(0) / rr_[i, :, 3].sum(0)) / g))
    m = (np.datetime64('1959-01-01T00', 'h') + HOURS[A['times']].astype('timedelta64[h]')).astype('datetime64[M]').astype(int)
    bl = [np.flatnonzero(m == u) for u in np.unique(m)]; rng_ = np.random.default_rng(0); full = np.arange(len(m))
    v = [f(A[reg], i) / f(B[reg], i) - 1 for i in (np.concatenate([bl[j] for j in rng_.integers(0, len(bl), len(bl))]) for _ in range(boot))]
    return 100 * (f(A[reg], full) / f(B[reg], full) - 1), 100 * np.quantile(v, .025), 100 * np.quantile(v, .975)


# 1. how OSG works: 1-D Gaussian-process illustration (prior = GP around the background, fibre = the exact GP conditional)
rng = np.random.default_rng(4); s = np.linspace(0, 10, 500); S = np.array([1.4, 3.9, 6.1, 8.7]); ell, sd = 1.1, 1.
kern = lambda a, b: sd ** 2 * np.exp(-.5 * (a[:, None] - b[None, :]) ** 2 / ell ** 2)
xb = lambda q: .9 * np.sin(.62 * q) + .25 * np.cos(1.7 * q)
Kss, KsS, KSS = kern(s, s), kern(s, S), kern(S, S); Ls = np.linalg.cholesky(Kss + 1e-8 * np.eye(len(s)))
truth = xb(s) + Ls @ rng.standard_normal(len(s)); U_true = np.interp(S, s, truth); sig = .12
y = U_true + sig * rng.standard_normal(len(S)); M = 9
LS = np.linalg.cholesky(KSS); U0 = xb(S)[None] + (LS @ rng.standard_normal((len(S), M))).T                     # head: prior draws at the stations
G = KSS @ np.linalg.inv(KSS + sig ** 2 * np.eye(len(S))); Up = U0 + (G @ (y[None] + sig * rng.standard_normal((M, len(S))) - U0).T).T   # Matheron
X0 = xb(s)[None] + (Ls @ rng.standard_normal((len(s), M))).T; W = KsS @ np.linalg.inv(KSS)
X0S = np.array([np.interp(S, s, x0) for x0 in X0]); Xf = X0 + (W @ (Up - X0S).T).T                                 # fibre: exact GP conditional through U (Matheron)
fig, axs = plt.subplots(1, 3, figsize=(12, 4.0), sharey=True)
titles = ['1. Голова: prior в станциях', '2. Точный Байес', '3. Волокно $K$: поле через $U$']
for k, ax in enumerate(axs):
    ax.plot(s, xb(s), color='#9AA3AD', lw=2, label='фон $x_b$')
    if k < 2:
        for j, Sj in enumerate(S):
            ax.add_patch(plt.Rectangle((Sj - .12, xb(Sj) - 2 * np.sqrt(KSS[j, j])), .24, 4 * np.sqrt(KSS[j, j]), color=C['prior'], alpha=.35, lw=0))
        jit = (np.arange(M) - M / 2) * .03
        Uk = U0 if k == 0 else Up
        for j, Sj in enumerate(S): ax.scatter(Sj + jit, Uk[:, j], s=22, color=C['psamp'] if k == 0 else C['osg'], zorder=4, lw=0)
    if k >= 1: ax.errorbar(S, y, yerr=2 * sig, fmt='s', color=C['obs'], ms=6, capsize=4, lw=1.6, zorder=5, label='наблюдение $y \\pm 2\\sigma$')
    if k == 2:
        for i in range(M): ax.plot(s, Xf[i], color=C['osg'], lw=1.1, alpha=.55)
        ax.plot(s, truth, color=C['truth'], lw=1.6, ls=(0, (4, 3)), label='правда')
        ax.scatter(np.repeat(S, M), Up.T.reshape(-1), s=14, color=C['osg'], zorder=4, lw=0)
    for Sj in S: ax.axvline(Sj, color='#d9dde2', lw=1, zorder=0)
    ax.set_title(titles[k], fontsize=16, loc='left'); ax.set_xlabel('координата вдоль линии'); ax.set_xlim(0, 10); ax.set_ylim(-3.6, 3.6); ax.grid(False)
axs[0].set_ylabel('значение поля')
h = [Line2D([], [], color='#9AA3AD', lw=2, label='фон $x_b$'), Patch(color=C['prior'], alpha=.5, label='prior в станции (±2 SD)'),
     Line2D([], [], marker='o', ls='', color=C['psamp'], label='выборки prior $U_0$'), Line2D([], [], marker='s', ls='', color=C['obs'], label='наблюдение $y\\pm2\\sigma$'),
     Line2D([], [], marker='o', ls='', color=C['osg'], label='выборки апостериора $U$'), Line2D([], [], color=C['osg'], lw=1.4, label='члены ансамбля'),
     Line2D([], [], color=C['truth'], lw=1.6, ls=(0, (4, 3)), label='правда')]
fig.legend(handles=h, loc='lower center', ncol=4, fontsize=14, bbox_to_anchor=(.5, -.13), handletextpad=.4, columnspacing=1.4)
fig.tight_layout(rect=(0, .1, 1, 1)); fig.savefig(OUT / 'osg_idea.png'); plt.close(fig)

# 2. precision: OSG vs OA-FM and the exact-U level
SIG = [('nom', 1.), ('sig05', .5), ('sig03', .3), ('sig02', .2), ('stress', .1), ('sig005', .05), ('sig002', .02), ('sig001', .01)]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.1))
for ax, reg, title in ((axs[0], 'free', 'клетки без станций'), (axs[1], 'held', 'отложенные станции')):
    lim = agg('oracle_nom', reg)
    ax.axvspan(.0085, .03, color='#f1f2f4', zorder=0); ax.text(.0155, .985, 'вне\nобучения\nOA-FM', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=12.5, color='#777')
    for key, pat, lab in (('oafm', 'fmpe_map_{}_residual_s0', 'OA-FM (сильнейший монолит)'), ('osg', 'nhead_{}_big', 'OSG (наш)')):
        xs = [v for sc, v in SIG]; ys = [agg(pat.format(sc), reg) for sc, v in SIG]
        ax.plot(xs, ys, '-o', color=C[key], lw=2.6, ms=6, label=lab, zorder=3)
        ax.annotate(f'{100 * (ys[-1] / lim - 1):+.1f}%' if key == 'oafm' else f'{100 * (ys[-1] / lim - 1):+.2f}%', (xs[-1], ys[-1]),
                    xytext=(-6, 8), textcoords='offset points', ha='right', color=C[key], fontsize=14, fontweight='bold')
    ax.axhline(lim, color=C['level'], lw=1.6, ls=(0, (2, 2)), label='тот же $K$ с точными значениями в станциях', zorder=2)
    ax.set_xscale('log'); ax.invert_xaxis(); ax.set_xticks([v for _, v in SIG]); ax.set_xticklabels(['1', '.5', '.3', '.2', '.1', '.05', '.02', '.01'])
    ax.minorticks_off(); ax.set_xlabel('ошибка наблюдений, σ × …'); ax.set_title(title, loc='left')
axs[0].set_ylabel('CRPS / ошибка фона\n(меньше — лучше)')
h, l = axs[0].get_legend_handles_labels(); fig.legend(h[::-1], l[::-1], loc='lower center', ncol=3, bbox_to_anchor=(.5, -.04))
fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(OUT / 'precision2.png'); plt.close(fig)


# 3. response test
def summ(path, name='smooth'):
    d = json.loads(Path(path).read_text()); dl = d['delta']; rng_ = np.random.default_rng(0)
    rr = [r for r in d['per_case'] if r['dir'] == name]; D = np.array([r[f'D_{dl:g}'] for r in rr]); Cv = np.array([r['cov'] for r in rr])
    b = np.array([D[i].sum(0) / Cv[i].sum(0) for i in (rng_.integers(0, len(rr), len(rr)) for _ in range(2000))])
    return D.sum(0) / Cv.sum(0), np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)


fig, ax = plt.subplots(figsize=(8, 4.6)); grp = [('станции\nσ × 1', 'n1', 0), ('станции\nσ × .1', 'n01', 0), ('скрытые соседи\nσ × 1', 'n1', 2), ('скрытые соседи\nσ × .1', 'n01', 2)]
for k, (key, pat, lab) in enumerate((('osg', 'chi_nhead_{}.json', 'OSG'), ('oafm', 'chi_fmpe_map_res_{}.json', 'OA-FM'))):
    for g, (_, nm, p) in enumerate(grp):
        c, lo, hi = summ(CH / pat.format(nm)); x = g + (k - .5) * .34
        ax.bar(x, c[p], width=.32, color=C[key], label=lab if g == 0 else None, zorder=2)
        ax.errorbar([x], [c[p]], yerr=[[c[p] - lo[p]], [hi[p] - c[p]]], color='#333', capsize=3, lw=1.2, zorder=3)
        ax.text(x, max(hi[p], c[p]) + .04, f'{c[p]:.2f}'.lstrip('0') if c[p] < 1 else f'{c[p]:.2f}', ha='center', fontsize=14, color=C[key], fontweight='bold')
ax.axhline(1, color=C['level'], lw=1.6, ls=(0, (2, 2)), zorder=1)
ax.set_xticks(range(4)); ax.set_xticklabels([g[0] for g in grp]); ax.set_ylim(0, 1.75); ax.set_ylabel('χ')
ax.grid(axis='x', visible=False); ax.legend(loc='upper left', ncol=2)
fig.tight_layout(); fig.savefig(OUT / 'chi2.png'); plt.close(fig)

# 4. correlated errors
SCN = [('nom', 'независимые\nошибки'), ('cg25l2', 'γ = .25\n670 км'), ('cg50l2', 'γ = .5\n670 км'), ('cg50l4', 'γ = .5\n1340 км')]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={'width_ratios': [1, 1]})
for ax, reg, title in ((axs[0], 'free', 'клетки без станций'), (axs[1], 'active', 'станции')):
    for k, (key, pat, lab) in enumerate((('oafm', 'fmpe_map_{}_residual_s0', 'OA-FM (видит только диагональ R)'), ('diag', 'OSG с одной диагональю R', 'OSG с одной диагональю R'))):
        pat = 'nhead_{}_big_diagr' if key == 'diag' else pat
        for i, (sc, _) in enumerate(SCN):
            if key == 'diag' and sc == 'nom': continue
            d, lo, hi = relci(pat.format(sc), f'nhead_{sc}_big', reg); x = i + (k - .5) * .34
            ax.bar(x, d, width=.32, color=C[key], label=lab if i == 1 else None, zorder=2)
            ax.errorbar([x], [d], yerr=[[d - lo], [hi - d]], color='#333', capsize=3, lw=1.1, zorder=3)
            ax.text(x, (hi if d >= 0 else lo) + (.03 if reg == 'free' else .4) * (1 if d >= 0 else -1) * (4 if reg == 'free' else 1), f'{d:+.1f}', ha='center',
                    va='bottom' if d >= 0 else 'top', fontsize=12.5, color=C[key], fontweight='bold')
    ax.axhline(0, color=C['level'], lw=1.2); ax.set_xticks(range(len(SCN))); ax.set_xticklabels([s_[1] for s_ in SCN]); ax.grid(axis='x', visible=False)
    ax.set_title(title, loc='left'); ax.set_ylabel('хуже OSG с полной R, %')
axs[0].set_ylim(-1.5, 5.5); h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=2, bbox_to_anchor=(.5, -.05))
fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(OUT / 'corr2.png'); plt.close(fig)

# 5. exact 2-D experiment, reduced
L2 = lambda nm: json.loads((TOY / f'{nm}.json').read_text()); SG = [1., .3, .1, .03, .01]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
for ax, lab, title in ((axs[0], 'sto', 'обычное обучение FM (метки $x_1-x_0$)'), (axs[1], 'teach', 'учитель: точная скорость')):
    d = L2(f'P1_{lab}_amort'); ax.plot(SG, [d['eval'][str(s_)]['mean']['b_ratio_crn'] for s_ in SG], '-o', color=C['oafm'], lw=2.4, ms=6, label='как OA-FM, одна модель на все σ')
    ys = [L2(f'P1_{lab}_s{s_:g}')['eval'][str(s_)]['mean']['b_ratio_crn'] for s_ in SG]
    ax.plot(SG, ys, '--o', color=C['oafm'], lw=1.8, ms=5, mfc='white', label='как OA-FM, своя модель на каждый σ')
    d = L2(f'P4_{lab}_amort'); ax.plot(SG, [d['eval'][str(s_)]['mean']['b_ratio_crn'] for s_ in SG], '-o', color=C['osg'], lw=2.4, ms=6, label='как OSG: $V-\\rho U$')
    ax.axhline(1, color=C['level'], lw=1.4, ls=(0, (2, 2))); ax.set_xscale('log'); ax.invert_xaxis(); ax.set_xticks(SG); ax.set_xticklabels(['1', '.3', '.1', '.03', '.01'])
    ax.minorticks_off(); ax.set_xlabel('ошибка наблюдения σ (SD prior = 1)'); ax.set_title(title, loc='left')
axs[0].set_ylabel('выученная связь / точная'); axs[0].set_ylim(.45, 1.25)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(.5, -.05))
fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(OUT / 'toy2.png'); plt.close(fig)

# 6. rank histograms (OSG vs OA-FM), averaged over the six channels; whiskers = channel range
if RK is not None and (RK / 'scen_nhead_stress_big.json').exists():
    REG = [('free', 'клетки без станций'), ('held', 'отложенные станции'), ('active', 'сами станции')]; SCR = [('nom', 'обычная ошибка (σ × 1)'), ('stress', 'точные наблюдения (σ × .1)')]
    fig, axs = plt.subplots(len(SCR), len(REG), figsize=(14, 6.4), sharex=True, sharey=True)
    for i, (sc, sct) in enumerate(SCR):
        for j, (reg, rt) in enumerate(REG):
            ax = axs[i, j]
            for k, (key, f, lab) in enumerate((('osg', f'scen_nhead_{sc}_big.json', 'OSG'), ('oafm', f'scen_fmpe_map_{sc}_residual_s0.json', 'OA-FM'))):
                R = json.loads((RK / f).read_text())['regions'][reg]; H = np.array([v for v in R['rank_hist'].values() if v is not None]); m_ = H.mean(0)
                ssr = np.mean([v for v in R['ssr'].values() if v is not None]); xx = np.arange(len(m_)) + (k - .5) * .4
                ax.bar(xx, m_, width=.38, color=C[key], label=f'{lab}' if (i, j) == (0, 0) else None, zorder=2)
                ax.text(.98, .93 - .1 * k, f'{lab}: SSR {ssr:.2f}', transform=ax.transAxes, ha='right', fontsize=11, color=C[key], fontweight='bold')
            ax.axhline(1, color=C['level'], lw=1.3, ls=(0, (2, 2)), zorder=3); ax.grid(axis='x', visible=False)
            if i == 0: ax.set_title(rt, loc='left')
            if j == 0: ax.set_ylabel(sct + '\nчастота / равномерная')
            if i == len(SCR) - 1: ax.set_xlabel('ранг правды среди 16 членов')
    axs[0, 0].set_ylim(0, 2.6); axs[0, 0].set_xticks([0, 4, 8, 12, 16]); fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='lower center', ncol=2, bbox_to_anchor=(.5, -.03))
    fig.tight_layout(rect=(0, .05, 1, 1)); fig.savefig(OUT / 'ranks2.png'); plt.close(fig)

# 7. one analysis at sigma x .1 (first panel time, 2019-01-01 00 UTC): msl increments over the North Atlantic and Europe, OSG vs OA-FM
if RK is not None and (RK / 'members_nhead_stress_big.npz').exists() and (RK / 'era5_static.npz').exists():
    A = np.load(RK / 'members_nhead_stress_big.npz'); B = np.load(RK / 'members_fmpe_map_stress_residual_s0.npz'); ST = np.load(RK / 'era5_static.npz')
    lat, lon = ST['lat'], ST['lon']; lon_c = np.where(lon >= 180, lon - 360, lon); order = np.argsort(lon_c); lon_s = lon_c[order]
    rows = np.flatnonzero((lat >= 30) & (lat <= 72)); cols = np.flatnonzero((lon_s >= -45) & (lon_s <= 40))
    def crop(f): g = f[..., order]; return g[..., rows, :][..., cols][..., ::-1, :]                     # north up
    la, lo = lat[rows][::-1], lon_s[cols]; ext = [lo[0] - .75, lo[-1] + .75, la[-1] - .75, la[0] + .75]
    lsm = crop(ST['lsm']); c, sc_, unit = 5, 100., 'гПа'                                                 # msl in hPa
    inc_t = crop(A['x'][c] - A['xb'][c]) / sc_; act = crop(A['active'][c]); v = float(np.nanpercentile(np.abs(inc_t), 99))
    sds = {nm: crop(Z['X'][:, c].std(0, ddof=1)) / sc_ for Z, nm in ((A, 'OSG'), (B, 'OA-FM'))}; vs = float(max(np.nanpercentile(x_, 99) for x_ in sds.values()))
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.2), constrained_layout=True)
    def show(ax, F, title, cm, lim, dots=False):
        im = ax.imshow(F, extent=ext, cmap=cm, vmin=lim[0], vmax=lim[1], aspect=1 / np.cos(np.deg2rad(51)), interpolation='nearest')
        ax.contour(lo, la, lsm, levels=[.5], colors='k', linewidths=.7)
        if dots: yy, xx = np.nonzero(act); ax.scatter(lo[xx], la[yy], s=2.5, color='k', lw=0, alpha=.6)
        ax.set_title(title, fontsize=14, loc='left'); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); return im
    for i_, (Z, nm) in enumerate(((A, 'OSG'), (B, 'OA-FM'))):
        X = Z['X'][:, c]
        if i_ == 0: im0 = show(axs[0, 0], inc_t, 'правда − фон', 'RdBu_r', (-v, v))
        else: show(axs[1, 0], np.where(act, inc_t, np.nan), 'что видят методы: станции', 'RdBu_r', (-v, v), dots=False)
        show(axs[i_, 1], crop(X.mean(0) - Z['xb'][c]) / sc_, f'{nm}: среднее − фон', 'RdBu_r', (-v, v))
        show(axs[i_, 2], crop(X[0] - Z['xb'][c]) / sc_, f'{nm}: один член − фон', 'RdBu_r', (-v, v))
        im3 = show(axs[i_, 3], sds[nm], f'{nm}: разброс ансамбля', 'magma_r', (0, vs), dots=True)
    fig.colorbar(im0, ax=axs[:, :3], shrink=.7, label=f'приращение давления, {unit}', pad=.01)
    fig.colorbar(im3, ax=axs[:, 3], shrink=.7, label=f'SD, {unit}', pad=.01)
    fig.savefig(OUT / 'maps2.png'); plt.close(fig)
print('saved', sorted(p.name for p in OUT.iterdir() if p.suffix == '.png'))
