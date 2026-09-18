# Полный отчёт с фигурами: ассимиляция SIC, калибровка, сравнение с 3D-Var

Дата: 2026-09-18 13:10. Дополняет `claude_monthly_report_ru.md` (текст) и
`claude_undertrained_or_overfit_ru.md` (доказательства). Все фигуры — в
`docs/research/figures/`.

## 1. Главный вывод

Ансамбль ассимиляции (`balanced_50m`, learned full condition) **в нормальных
условиях (native-two-field) почти калиброван по разбросу**: spread/RMSE .90,
pooled rank-TV .044 при null идеального ансамбля .034. «Недодисперсия .72»,
которую агент месяц пытался лечить, — артефакт протокола сравнения с 3D-Var
(обнуление sithic): та же модель, те же 40 дат, RMSE .190 → .156, rank-TV
.356 → .044. Реальные дефекты — структурные (нет точных нулей, наблюдения
дня d не соблюдаются, отпечаток треков) — закрываются OOF-слоем: fair CRPS
−29 % на 40/40 held-out дат, нули = truth, ранги на уровне null.

## 2. Rank-гистограммы (dev-2022, 40 дат, рандомизированные ранги, M=10)

![pooled](figures/rank_hist_pooled_all.png)

| ансамбль | дат | pooled TV | ≤2 px | 3–10 px | 11–40 px | >40 px |
|---|---|---|---|---|---|---|
| baseline siconc-only | 40 | 0.356 | 0.466 | 0.365 | 0.262 | 0.397 |
| baseline native-2f | 40 | 0.044 | 0.033 | 0.041 | 0.058 | 0.252 |
| arm A siconc-only | 40 | 0.101 | 0.151 | 0.100 | 0.120 | 0.206 |
| legacy ema_last siconc-only | 40 | 0.433 | 0.529 | 0.459 | 0.388 | 0.486 |
| OOF layer v3 (3 dates) | 3 | 0.016 | 0.043 | 0.017 | 0.060 | 0.383 |

Null для идеального обменного ансамбля (M=10): pooled TV .034 (q95 .083).

По стратумам расстояния до трека:

![baseline siconc-only](figures/rank_hist_baseline_siconc-only.png)
![baseline native](figures/rank_hist_baseline_native-2f.png)
![layer v3](figures/rank_hist_OOF_layer_v3_(3_dates).png)
![arm A siconc](figures/rank_hist_arm_A_siconc-only.png)
![legacy](figures/rank_hist_legacy_ema_last_siconc-only.png)

Чтение: siconc-only — истина систематически на краях у трека; native — почти
плоско; слой v3 — плоско везде (3 даты с сохранёнными членами); legacy
`ema_last` (переобученный) — U-форма, разброс слишком мал.

## 3. Кривые обучения / коллапса

![collapse](figures/fig_collapse_curves.png)

Чистая concat-модель: val_loss растёт после ~20k updates, RMSE .121 → .149,
spread/RMSE .68 → .43 — переобучение (все прошлые оценки брали её худший
checkpoint). `balanced_50m` (50 % шагов без условия): коллапса нет, s/r
.8–1.0, но RMSE на ~25 % хуже. Arm A/C (dropout 30 %, 4 часовых среза,
6.5k updates, ранняя остановка по fair CRPS): RMSE .12–.13, s/r .84–.95 на
training-метрике.

## 4. Слой OOF v3 по стратумам расстояния до трека (held-out)

![layer by distance](figures/fig_layer_v3_by_distance.png)

Cross-fit 5×8 дат: fair CRPS .0585 → .0417 (−29 %, CI95 [−.0198, −.0139],
40/40 дат), RMSE .190 → .158, s/RMSE .93, rank-TV .099, cov80 .82–.87 (null
.83–.98), нули .53 (truth .54), cap .106 (truth .060, +.04 — единственный
хвост), на треке дня d RMSE .025 (raw .093).

## 5. Члены ансамбля: до и после

Baseline, зима (трек в открытой воде) и весна (сплошной лёд): полосы по
геометрии треков, «дыры» в паке, нет точных нулей.

![members jan](figures/members_2022-01-16_zoom.png)
![members apr](figures/members_2022-04-11_zoom.png)

Raw vs слой (внизу карты P(exact 0) и P(cap)): у raw шум с полосами треков,
у слоя — связная вода как в истине, отпечатка нет.

![layer jan](figures/v2_2022-01-16_zoom.png)
![layer apr](figures/v2_2022-04-11_zoom.png)

## 6. Сравнение с 3D-Var (test-2023, 200 дней, протокол коллег)

![timeseries](figures/fig_3dvar_protocol_200d_timeseries.png)
![monthly](figures/fig_3dvar_protocol_200d_monthly.png)

| регион | фон RMSE | анализ RMSE (M=5) | ≈M→∞ | MAE фон→анализ | IIEE фон→анализ | CRPS | spread |
|---|---:|---:|---:|---|---|---:|---:|
| full (48 637 px) | .2748 | **.1569** | .1444 | .1114 → .0678 | .1179 → .0821 | .0481 | .1280 |
| track_imitation (613 px/день) | .2327 | **.0977** | .0863 | .0828 → .0376 | .0977 → .0514 | .0239 | .0928 |

Конфигурация: коммит b539875, `ema_last`, `full`, siconc-only, dopri5/25,
seed 1234, 5 членов; полный список столбцов и протокол — Часть IV
`claude_monthly_report_ru.md`. Строки 3D-Var добавить из данных коллег
(сверить `background_rmse`). В этом протоколе наша модель работает в
OOD-режиме (sithic скрыт); честнее для неё — arm C (обучен под протокол,
оценка в очереди) или native-two-field, если коллеги согласны.

## 7. Прогнозная линия (второстепенно)

![product law](figures/product_law_panel_factor4.png)

CPU-фальсификатор face-aware каскада: условно-независимые грани при истинных
родителях дают зерно 5–7× truth — то же, что у baseline base32. Каскад —
HOLD/NO-GO; подробности `claude_face_aware_cascade_review.md`.

## 8. Что дальше

Одна GPU, очередь: in-sample native → arm A native → legacy last/best native
→ arm C strict. Затем заморозка кандидата (balanced_50m или arm C, плюс слой
v3), один вероятностный прогон на 2023, статья.
