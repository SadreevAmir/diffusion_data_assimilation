"""Figures of the 15 Oct 2026 group talk, rebuilt from saved evaluation outputs (no new runs). Russian labels, ONE naming and colour
scheme for every figure (OSG black, OA-FM blue, plain monolith purple, monolith with lambda input green, FMPE at nominal noise grey,
'same K with exact U' orange), fonts sized for a 16:9 beamer slide.
Usage: python make_figs.py <scen_main> <chi_audit> <toy2> <bcf repo root> <out> <hours_trainval.npy>
  precision.png    free / held-out CRPS against the noise multiplier (ERA5 2019 DEV, 67 times, M = 16) with the exact-U level
  response.png     chi for phi1 / phi3 (8 cases, M 256 / 128) and the Cov = SD * SD * Corr split at sigma x .1 (smooth direction)
  coupling.png     illustration for the chi slide: same marginal of V, correct vs 1.8x coupling to the station
  slope.png        effective station slope profiles of the plain monolith and the terminal station SD (plain vs residual)
  toy.png          the exact 2-D coupling experiment (CRN ratio learned / exact coupling)
  corr.png         correlated observation errors: monoliths and OSG with a diagonal R relative to OSG with the full R
  posterior2d.png  illustration: contraction of a 2-D Gaussian posterior
  ice_u_rmse.png   sea-ice patches (validation 2022): RMSE of the ensemble-mean observed summaries against the noise SD
  ice_members.png  full-field SIC, 10 Feb 2022: truth, a member of the smooth kernel, a member of the grainy kernel
Prints the numbers quoted on the slides that are computed here (gaps to the exact-U level).
"""
import json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

SC, CH, TOY, BCF, OUT = (Path(p) for p in sys.argv[1:6]); OUT.mkdir(parents=True, exist_ok=True); HOURS = np.load(sys.argv[6])
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 12.5, 'axes.labelsize': 12, 'legend.fontsize': 10.5, 'xtick.labelsize': 11,
                     'ytick.labelsize': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 220, 'savefig.bbox': 'tight'})
C = {'osg': '#111111', 'oafm': '#2a6fb0', 'plain': '#7a5bb5', 'amr': '#2f9e6a', 'fmpe': '#8a8f98', 'limit': '#c8551d', 'eps': '#a0652a'}
N = {'osg': 'OSG', 'oafm': 'OA-FM (остаточные координаты)', 'plain': 'монолит, обычные координаты', 'amr': 'монолит с λ на входе (FMPE-amr-wide)',
     'fmpe': 'FMPE, обучен при номинальном шуме', 'limit': 'тот же $K$ с точными значениями в станциях'}


def agg(name, reg):
    z = np.load(SC / f'scen_{name}_pertime.npz'); r = z[reg]; c = r[:, :, 0].sum(0) / r[:, :, 3].sum(0); return float(np.mean(c / z['gsb']))


# 1. precision
SIG = [('nom', 1.), ('sig05', .5), ('sig03', .3), ('sig02', .2), ('stress', .1), ('sig005', .05), ('sig002', .02), ('sig001', .01)]
M = [('osg', 'nhead_{}_big', 'o', '-'), ('oafm', 'fmpe_map_{}_residual_s0', 's', '-'), ('plain', 'fmpe_map_{}_plain_s0', '^', '-'),
     ('amr', 'fmpe_amr_{}_wide', 'D', '-'), ('fmpe', 'fmpe_{}', 'v', '--')]
fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.7))
for ax, reg, title in ((axs[0], 'free', 'свободные клетки (никогда не наблюдаются)'), (axs[1], 'held', 'отложенные станции')):
    lim = agg('oracle_nom', reg)
    for key, pat, mk, ls in M:
        xs, ys = [], []
        for sc, s in SIG:
            try: ys.append(agg(pat.format(sc), reg)); xs.append(s)
            except FileNotFoundError: pass
        ax.plot(xs, ys, ls, marker=mk, color=C[key], ms=4, lw=1.7, label=N[key])
        gaps = {s: 100 * (y / lim - 1) for s, y in zip(xs, ys) if s <= .05}
        print(f'precision {reg:5s} {key:5s} gap to the exact-U level at .05/.02/.01:', ' / '.join(f'{gaps.get(s, float("nan")):+.2f}%' for s in (.05, .02, .01)))
    ax.axhline(lim, color=C['limit'], ls=':', lw=1.8, label=N['limit'])
    ax.axvspan(.009, .027, color='#eef1f5', zorder=0)
    ax.text(.0155, ax.get_ylim()[0] + .97 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 'вне\nобучения\nмонолитов', ha='center', va='top', fontsize=9.5, color='#555')
    ax.set_xscale('log'); ax.invert_xaxis(); ax.set_xticks([s for _, s in SIG]); ax.set_xticklabels(['1', '.5', '.3', '.2', '.1', '.05', '.02', '.01'])
    ax.set_xlabel('множитель ошибки наблюдений  σ ×'); ax.set_title(title); ax.grid(alpha=.25, lw=.5)
axs[0].set_ylabel('CRPS / RMS ошибки фона\n(меньше — лучше)')
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, 0), fontsize=10)
fig.tight_layout(rect=(0, .12, 1, 1)); fig.savefig(OUT / 'precision.png'); plt.close(fig)


# 2. response
def summ(path, name='smooth'):
    d = json.loads(Path(path).read_text()); dl = d['delta']; rng = np.random.default_rng(0)
    rr = [r for r in d['per_case'] if r['dir'] == name]; D = np.array([r[f'D_{dl:g}'] for r in rr]); Cv = np.array([r['cov'] for r in rr]); V = np.array([r['var'] for r in rr])
    b = np.array([D[i].sum(0) / Cv[i].sum(0) for i in (rng.integers(0, len(rr), len(rr)) for _ in range(2000))])
    return D.sum(0) / Cv.sum(0), np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0), D.sum(0), Cv.sum(0), V.sum(0)


R = [('osg', 'chi_nhead_{}.json'), ('oafm', 'chi_fmpe_map_res_{}.json'), ('plain', 'chi_fmpe_map_plain_{}.json')]
NR = {'osg': 'OSG', 'oafm': 'OA-FM', 'plain': 'монолит, обычн. коорд.'}
fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.2, 3.7), gridspec_kw={'width_ratios': [1.1, 1]})
for k, (key, pat) in enumerate(R):
    for nm, filled, dx in (('n01', True, 0), ('n1', False, .08)):
        c, lo, hi, *_ = summ(CH / pat.format(nm))
        print(f'chi {key:5s} {nm:4s} phi1 {c[0]:.2f} [{lo[0]:.2f}, {hi[0]:.2f}]  phi3 {c[2]:.2f} [{lo[2]:.2f}, {hi[2]:.2f}]')
        for j, p in enumerate((0, 2)):
            x = j + (k - 1) * .25 + dx
            ax.errorbar([x], [c[p]], yerr=[[c[p] - lo[p]], [hi[p] - c[p]]], fmt='o', ms=6, color=C[key], mfc=C[key] if filled else 'white', capsize=3, lw=1.4,
                        label=f'{NR[key]}, σ × {".1" if filled else "1"}' if j == 0 else None)
ax.axhline(1, color='#999', lw=1); ax.set_xticks([0, 1]); ax.set_xticklabels(['φ₁: проекция\nсамих станций', 'φ₃: свободные клетки\nрядом со станциями'])
ax.set_ylabel('χ = отклик / ковариация\n(точный Байес: 1)'); ax.set_ylim(0, 1.62); ax.grid(axis='y', alpha=.25)
ax.legend(fontsize=9, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.2), columnspacing=.8, handletextpad=.3)
c0 = summ(CH / 'chi_nhead_n01.json'); names = ['отклик', 'SD φ₃', 'SD\nстанций', 'корр.', 'ковар.']
for i, (key, pat) in enumerate(R[1:]):
    c = summ(CH / pat.format('n01')); p = 2
    cov = c[4][p] / c0[4][p]; sdp = np.sqrt(c[5][p] / c0[5][p]); sdv = np.sqrt(c[5][0] / c0[5][0])
    vals = [c[3][p] / c0[3][p], sdp, sdv, cov / (sdp * sdv), cov]; xx = np.arange(5) + (i - .5) * .4
    bx.bar(xx, vals, width=.38, color=C[key], label=NR[key])
    for x_, v in zip(xx, vals): bx.text(x_, v * (1.05 if i == 0 else 1.18), f'{v:.2f}'.lstrip('0'), ha='center', fontsize=7.5, color=C[key])
bx.axhline(1, color='#999', lw=1); bx.set_yscale('log'); bx.set_xticks(range(5)); bx.set_xticklabels(names, fontsize=9); bx.set_ylim(.8, 15)
bx.set_ylabel('во сколько раз больше, чем у OSG\n(φ₃, σ × .1)'); bx.legend(frameon=False, fontsize=9.5); bx.set_title('ковариация = SD · SD · корреляция', fontsize=11)
fig.tight_layout(); fig.savefig(OUT / 'response.png'); plt.close(fig)

# 3. coupling illustration for the chi slide: same V marginal, correct vs 1.8x coupling (rho = .8, sigma = .1)
rho, r = .8, .01; tau = math.sqrt(r / (1 + r)); vV = 1 - rho ** 2 + rho ** 2 * tau ** 2; rng = np.random.default_rng(5); n = 1_000_000
W = rng.standard_normal(n)
fig, ax = plt.subplots(figsize=(4.8, 3.5))
sds = []
for f, col, lab in ((1., C['osg'], 'правильная связь ρτ  (χ = 1)'), (1.8, C['oafm'], 'связь × 1.8  (χ ≈ .56)')):
    b = f * rho * tau; Vd = b * W + math.sqrt(vV - b ** 2) * rng.standard_normal(n)
    bins = np.linspace(-2.5, 2.5, 11); mid = .5 * (bins[1:] + bins[:-1]); means = [Vd[(W >= a) & (W < c)].mean() for a, c in zip(bins[:-1], bins[1:])]
    ax.plot(mid, means, 'o-', color=col, lw=2, ms=5, label=lab); sds.append(Vd.std())
ax.text(.98, .04, f'SD(V): {sds[0]:.3f} и {sds[1]:.3f}', transform=ax.transAxes, ha='right', fontsize=9.5, color='#444')
ax.axhline(0, color='#bbb', lw=.8); ax.set_xlabel('отклонение станции  (U − m)/τ'); ax.set_ylabel('среднее V − E[V]\nв полосе по станции')
ax.set_title('одна и та же маргиналь V,\nразная связь со станцией (σ = .1)', fontsize=11.5); ax.legend(fontsize=9, frameon=False, loc='upper left'); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(OUT / 'coupling.png'); plt.close(fig)

# 4. slope
P = json.loads((BCF / 'docs/results/slope_plain_s0.json').read_text()); Q = json.loads((BCF / 'docs/results/slope_residual_s0.json').read_text())
grid = np.array(P['grid']); tm = .5 * (grid[1:] + grid[:-1]); lams = ['1.0', '0.3', '0.1', '0.05', '0.02']
cols = ['#2a6fb0', '#2f9e6a', '#e0a020', '#e06a2a', '#c8331d']
fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.8, 3.5), gridspec_kw={'width_ratios': [1.35, 1]})
for lam, col in zip(lams, cols):
    e = P['per_lam'][lam]; prof = np.array(e['a_eff_median_profile']); tau_ = e['tau_reference_median']
    ex = (tm * tau_ ** 2 - (1 - tm)) / ((1 - tm) ** 2 + tm ** 2 * tau_ ** 2)
    ax.plot(1 - tm, prof, color=col, lw=2, label=f'λ = {float(lam):g}'); ax.plot(1 - tm, ex, color=col, lw=1.1, ls='--')
ax.set_xscale('log'); ax.set_xlabel('1 − t  (сколько осталось до конца пути)'); ax.set_ylabel('эффективная крутизна\nпо координате станции')
ax.set_title('монолит в обычных координатах:\nсплошная — по ансамблю сети, пунктир — точный гауссов поток', fontsize=10.5); ax.grid(alpha=.25)
ax.legend(fontsize=9.5, frameon=False, loc='lower left')
L = [float(x) for x in lams]
bx.plot(L, [P['per_lam'][x]['sd_flow_station_median'] for x in lams], '-^', color=C['plain'], lw=1.8, label='монолит, обычн. коорд.')
bx.plot(L, [Q['per_lam'][x]['sd_increment_station_median'] for x in lams], '-s', color=C['oafm'], lw=1.8, label='OA-FM (остаточные)')
bx.plot(L, [P['per_lam'][x]['tau_reference_median'] for x in lams], ':', color=C['osg'], lw=1.6, label='τ по одной станции')
bx.set_xscale('log'); bx.set_yscale('log'); bx.invert_xaxis(); bx.set_xlabel('множитель шума λ'); bx.set_ylabel('SD ансамбля в станции\n(единицы приращения)')
bx.set_xticks([1, .3, .1, .05, .02]); bx.set_xticklabels(['1', '.3', '.1', '.05', '.02']); bx.minorticks_off()
bx.set_yticks([.02, .05, .1, .2, .5]); bx.set_yticklabels(['.02', '.05', '.1', '.2', '.5'])
bx.set_title('итоговый разброс в станциях', fontsize=11); bx.grid(alpha=.25, which='both'); bx.legend(fontsize=9.5, frameon=False, loc='upper right')
fig.tight_layout(); fig.savefig(OUT / 'slope.png'); plt.close(fig)

# 5. toy
L2 = lambda nm: json.loads((TOY / f'{nm}.json').read_text())
SG = [1., .3, .1, .03, .01]; TC = {'P1': C['oafm'], 'P2': '#3a9d9a', 'P3': '#b8892a', 'P4': '#555555'}
fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.6), sharey=True)
for ax, lab, title in ((axs[0], 'sto', 'обычные метки FM ($x_1-x_0$)'), (axs[1], 'teach', 'точная скорость как учитель')):
    for Pn, name in (('P1', 'P1: остаточные в станции (как OA-FM)'), ('P2', 'P2: обе координаты стандартизованы'),
                     ('P3', 'P3: как P2, перекрёстный член × τ'), ('P4', 'P4: треугольные, V − ρU (ρ известна)')):
        d = L2(f'{Pn}_{lab}_amort'); ax.plot(SG, [d['eval'][str(s)]['mean']['b_ratio_crn'] for s in SG], '-o', color=TC[Pn], ms=4, lw=1.8, label=name)
        if Pn in ('P1', 'P2'):
            ys = [L2(f'{Pn}_{lab}_s{s:g}')['eval'][str(s)]['mean']['b_ratio_crn'] for s in SG]
            ax.plot(SG, ys, '--s', color=TC[Pn], ms=3.5, lw=1.2, mfc='white', label=f'{Pn}: отдельная модель на каждый σ' if lab == 'sto' else None)
    ax.axhline(1, color='#999', lw=1); ax.set_xscale('log'); ax.invert_xaxis(); ax.set_xticks(SG); ax.set_xticklabels(['1', '.3', '.1', '.03', '.01'])
    ax.set_xlabel('SD ошибки наблюдения σ (SD prior = 1)'); ax.set_title(title, fontsize=11.5); ax.grid(alpha=.25)
axs[0].text(.02, .04, 'сплошные — одна модель на все σ\nпунктир — своя модель на каждый σ', transform=axs[0].transAxes, fontsize=9, color='#444', va='bottom')
axs[0].set_ylabel('выученная / точная связь\nCov(V, (U − m)/τ)')
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(.5, -.03))
fig.tight_layout(rect=(0, .17, 1, 1)); fig.savefig(OUT / 'toy.png'); plt.close(fig)

# 6. correlated errors
SCN = [('nom', 'независимые\nσ × 1'), ('cg25l2', 'γ = .25\n670 км'), ('cg50l2', 'γ = .5\n670 км'), ('cg50l4', 'γ = .5\n1340 км'),
       ('sig03', 'независимые\nσ × .3'), ('cg50l2s03', 'γ = .5, 670 км\nσ × .3')]
MC = [('OSG с диагональной R', 'nhead_{}_big_diagr', C['osg'], 'D'), ('OA-FM', 'fmpe_map_{}_residual_s0', C['oafm'], 'o'),
      ('монолит, обычн. коорд.', 'fmpe_map_{}_plain_s0', C['plain'], 's'), ('монолит с λ на входе (FMPE-amr)', 'fmpe_amr_{}', C['amr'], '^')]


def relci(na, nb, boot=1500):
    A = np.load(SC / f'scen_{na}_pertime.npz'); B = np.load(SC / f'scen_{nb}_pertime.npz'); g = A['gsb']; ok = A['free'][:, :, 3].sum(0) > 0
    f = lambda rr_, i: float(np.mean(((rr_[i, :, 0].sum(0) / rr_[i, :, 3].sum(0)) / g)[ok]))
    m = (np.datetime64('1959-01-01T00', 'h') + HOURS[A['times']].astype('timedelta64[h]')).astype('datetime64[M]').astype(int)
    bl = [np.flatnonzero(m == u) for u in np.unique(m)]; rng_ = np.random.default_rng(0); full = np.arange(len(m))
    v = [f(A['free'], i) / f(B['free'], i) - 1 for i in (np.concatenate([bl[j] for j in rng_.integers(0, len(bl), len(bl))]) for _ in range(boot))]
    return 100 * (f(A['free'], full) / f(B['free'], full) - 1), 100 * np.quantile(v, .025), 100 * np.quantile(v, .975)


fig, ax = plt.subplots(figsize=(8.8, 3.7)); Wd = .17
for k, (lab, pat, col, mk) in enumerate(MC):
    xs, ys, lo, hi = [], [], [], []
    for i, (sc, _) in enumerate(SCN):
        if 'diagr' in pat and not sc.startswith('cg'): continue
        try: d, l_, h_ = relci(pat.format(sc), f'nhead_{sc}_big')
        except FileNotFoundError: continue
        xs.append(i + (k - 1.5) * Wd); ys.append(d); lo.append(d - l_); hi.append(h_ - d)
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk, color=col, ms=6, lw=1.4, capsize=2.5, label=lab)
ax.axhline(0, color='#999', lw=1); ax.axvline(3.5, color='#ccc', lw=1, ls='--')
ax.set_xticks(range(len(SCN))); ax.set_xticklabels([s[1] for s in SCN], fontsize=10)
ax.set_ylabel('CRPS свободных клеток\nк OSG с точной R, %\n(выше — метод хуже OSG)', fontsize=11); ax.grid(axis='y', alpha=.25)
ax.legend(fontsize=10, frameon=False, loc='upper left'); fig.tight_layout(); fig.savefig(OUT / 'corr.png'); plt.close(fig)

# 7. posterior illustration
rng = np.random.default_rng(3); y = .9
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.7), sharey=True)
for ax, sig, title in ((axs[0], 1., 'неточное наблюдение, σ = 1'), (axs[1], .1, 'точное наблюдение, σ = .1')):
    rr = sig ** 2; m = y / (1 + rr); tau = math.sqrt(rr / (1 + rr))
    pu = rng.standard_normal(3000); pv = rho * pu + math.sqrt(1 - rho ** 2) * rng.standard_normal(3000)
    U = m + tau * rng.standard_normal(1500); V = rho * U + math.sqrt(1 - rho ** 2) * rng.standard_normal(1500)
    ax.scatter(pu, pv, s=3, color='#c9cdd3', alpha=.6, label='prior p(U, V)'); ax.scatter(U, V, s=4, color=C['oafm'], alpha=.55, label='апостериор p(U, V | y)')
    ax.axvline(y, color=C['limit'], lw=1.8, ls='--', label='наблюдение y')
    ax.set_title(f'{title}\nSD станции τ = {tau:.2f}', fontsize=11.5)
    ax.set_xlabel('значение в станции U'); ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.grid(alpha=.2)
axs[0].set_ylabel('скрытое значение V\n(корреляция prior ρ = .8)'); axs[0].legend(fontsize=9.5, frameon=False, loc='upper left', markerscale=3)
fig.tight_layout(); fig.savefig(OUT / 'posterior2d.png'); plt.close(fig)

# 8. ice: RMSE of the ensemble-mean observed summaries (validation 2022)
U_ = json.loads((BCF / 'docs/ice_val/u_rmse_diag.json').read_text()); rhos = [1., .1, .01, .0001]
fig, ax = plt.subplots(figsize=(5.4, 3.9))
for key, lab, col, mk in (('osg_final', 'OSG', C['osg'], 'o'), ('fmpe_final', 'FMPE (монолит)', C['plain'], '^'), ('eps_final', 'EPS (диффузия с наведением)', C['eps'], 's')):
    for op, ls in (('grid', '-'), ('v0', '--')):
        ys = [U_[f'{key}/{op}/rho{r_:g}']['rmse_mean'] for r_ in rhos]
        ax.plot([math.sqrt(r_) for r_ in rhos], ys, ls, marker=mk, color=col, ms=4.5, lw=1.7, label=f'{lab}, {"сетка" if op == "grid" else "треки v0"}')
ax.plot([1, .01], [1, .01], ':', color='#666', lw=1.3, label='SD шума наблюдений')
ax.set_xscale('log'); ax.set_yscale('log'); ax.invert_xaxis(); ax.set_xlabel('SD шума наблюдений'); ax.set_ylabel('RMSE среднего ансамбля\nв наблюдаемых сводках')
ax.grid(alpha=.25, which='both'); ax.legend(fontsize=8.5, frameon=False, loc='lower left'); ax.set_title('лёд, патчи толщины, 2022', fontsize=11.5)
fig.tight_layout(); fig.savefig(OUT / 'ice_u_rmse.png'); plt.close(fig)

# 9. ice members: truth | smooth kernel member | grainy kernel member (10 Feb 2022)
a = Image.open(BCF / 'docs/figs/members_q1_k1_dev2022.png'); b = Image.open(BCF / 'docs/figs/members_kv3_dev2022.png')
pw = a.size[0] / 6; box = lambda i: (int(i * pw + 75), 70, int((i + 1) * pw - 10), 415)
crops = [a.crop(box(0)), a.crop(box(2)), b.crop(box(2))]
fig, axs = plt.subplots(1, 3, figsize=(8.6, 3.6))
for ax, im, t in zip(axs, crops, ['правда (NEMO)', 'член гладкого ядра:\nразброс занижен (SSR .51)', 'член зернистого ядра:\n«соль и перец»']):
    ax.imshow(im); ax.set_title(t, fontsize=11); ax.axis('off')
fig.tight_layout(rect=(0, .1, 1, 1))
cax = fig.add_axes([.3, .07, .4, .03]); sm = matplotlib.cm.ScalarMappable(cmap='Blues_r', norm=matplotlib.colors.Normalize(0, 1))
cb = fig.colorbar(sm, cax=cax, orientation='horizontal'); cb.set_ticks([0, .5, 1]); cb.ax.tick_params(labelsize=9)
cb.set_label('концентрация льда: 0 — вода, 1 — сплошной лёд; серое — суша', fontsize=9.5)
fig.savefig(OUT / 'ice_members.png'); plt.close(fig)
ov = Image.open(OUT / 'era5_overview_20121030.png'); W_, H_ = ov.size
ov.crop((0, int(.545 * H_), W_, H_)).save(OUT / 'era5_strip.png')
print('saved', sorted(p.name for p in OUT.iterdir() if p.suffix == '.png'))
