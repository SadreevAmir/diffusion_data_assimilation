# Канонический журнал экспериментов SIC/SIT

Последнее обновление: 2026-09-13. Это единая точка входа в историю
экспериментов, отрицательных результатов и текущих решений проекта. Число
считается доказанным только вместе с указанным JSON/PT-артефактом, ClearML task
или SHA. Validation-2022 является development reuse; test-2023 закрыт до
замораживания окончательного метода.

Статусы: `REFERENCE` — рабочий baseline; `COMPLETED` — расчёт завершён;
`REJECTED` — метод не прошёл объявленный gate; `VALIDATING` — обучение
завершено, окончательный вывод ещё не сделан; `PROPOSED` — идея, а не результат.

## Сводка

| ID | Эксперимент | Статус | Решение |
|---|---|---|---|
| DYN-B0 | Direct joint dynamics EMA6 | REFERENCE | Лучшие подтверждённые динамика и визуальные семплы; ансамбль недодисперсен |
| DYN-B1 | EMA6/EMA8 mixture screen | REJECTED | Соседние checkpoints слишком похожи; EMA8 хуже на проверенных cases |
| CAL-P1 | Global anomaly spread | REJECTED | CRPS/ranks лучше, boundary и spatial metrics хуже |
| CAL-P2 | Hurdle-IDR/ECC-Q | REJECTED | Чинит atoms/ranks, разрушает CRPS и геометрию |
| CAL-P3 | Mean-preserving projected spread | REJECTED | Сохраняет mean, но создаёт cap/boundary failures |
| CAL-P4 | Open-logit | REJECTED | CRPS/ranks лучше, boundary/spatial gate провален |
| CAL-P5 | ZOIB-EMOS/ECC-Q | REJECTED | Explicit marginal atoms не дают полезный joint spatial law |
| CAL-P6 | Прочие frozen postprocessors | REJECTED | Ни один не прошёл общий no-compensation gate |
| CAS-C1 | Coarse-budget train64 + frozen fine | REJECTED | Малый CRPS gain, SIT+3 ranks хуже; не end-to-end cascade |
| GEO-H1 | Endpoint-only geometry suffix continuation | REJECTED | Обе ветви хуже EMA6; это не был geometry-CFM test |
| GEO-P0 | Geometry-CFM zero-update preflight | COMPLETED | Математика, gradients и lifecycle прошли Astra audit |
| GEO-T1 | Full-CFM control512/treatment512 | REJECTED | Geometry term даёт малый matched-control gain, но обе ветви значительно хуже EMA6 |
| IDEA-F1 | One-shot front-plus-source mixed-support flow | CPU PASS / GPU HOLD | Representation корректен; проверяется реальный source+masked-CFM runner |

## DYN-B0 — direct EMA6

Один conditional flow совместно генерирует SIC/SIT на +3, +6 и +9 суток.
Все горизонты принадлежат одному member и создаются one-shot.

- server run:
  `/home/autoresearch_results/direct_dynamics_all_hours_v1/training/seed1701-night_20260909_v1`
- checkpoint: `epoch_snapshots/epoch_0006/ema_last_model.pth`
- SHA-256:
  `9b8bb6954b7a19179aee2e58b0c6cdddfcb8b489b1e29d6f503086c704fa0b1b`
- evidence:
  `/Users/amir/sciml/autoresearch_results/direct_dynamics_baseline_audit_v1/files/result/paired_evaluation.json`
- protocol: 12 validation-2022 cases, 8 members.

| Поле/горизонт | RMSE model / persistence | RMSE gain | Fair-CRPS gain |
|---|---:|---:|---:|
| SIC +3 | 0.090 / 0.104 | 13.3% | 31.8% |
| SIT +3 | 0.142 / 0.177 | 19.8% | 33.2% |
| SIC +6 | 0.108 / 0.132 | 18.2% | 38.9% |
| SIT +6 | 0.173 / 0.229 | 24.2% | 40.2% |
| SIC +9 | 0.123 / 0.150 | 18.2% | 40.3% |
| SIT +9 | 0.208 / 0.262 | 20.5% | 39.3% |

Spread/RMSE приблизительно SIC/SIT +3 `0.764/0.708`, +9 `0.711/0.534`;
fractional rank-TV `0.159…0.220`. Вывод: модель действительно учит динамику,
но uncertainty особенно на SIT и дальних lead недостаточна. Pooled ranks
смешивают exact-zero/cap atoms и сами по себе не являются calibration verdict.

## DYN-B1 — EMA6/EMA8 checkpoint screen

Артефакт:
`/Users/amir/sciml/diffusion_data_assimilation/docs/research/audit_checkpoint_disagreement_cpu.py`.
На двух winter cases корреляция anomaly fields `0.84–0.90`; RMS разницы means
`0.025–0.048` при within-model spread RMS `0.078–0.116`. Для block16 SIT+9
mean error EMA6/EMA8 `0.142/0.163`. Простую checkpoint mixture не развивать;
это не опровергает независимый multi-seed/year-bootstrap ensemble.

## CAL-P1…P6 — предыдущая post-hoc калибровка assimilation SIC

Это отдельный старый protocol: 40 validation dates и 10 members. Его числа
нельзя напрямую смешивать с 12×8 direct-dynamics validation.

| Метод | Ключевой результат | Почему отклонён |
|---|---|---|
| Global spread | fair CRPS `0.058491→0.055690`, SSR `0.724→1.061` | IIEE +5.50%, edge +4.41%, extent error +10.05%, Brier хуже |
| Hurdle-IDR/ECC-Q | zero-mass error `0.396859→0.013451`, rank discrepancy `0.079014→0.005308` | fair CRPS `→0.085581`, IIEE `→0.183385`, edge `→0.305843` |
| Projected spread | fair CRPS `→0.056300`, mean сохранён | Brier `0.056973→0.059650`, upper-cap mass `0→0.167438` |
| Open-logit | fair CRPS `→0.055738`, rank discrepancy `→0.011764` | Brier `→0.059107`, mass above .999 `→0.071936`, variogram fail |
| ZOIB-EMOS/ECC-Q | zero/one mass errors `→0.008275/0.0000687`, ranks `→0.009882` | fair CRPS не лучше, RMSE `→0.200774`, IIEE `→0.087190`, edge `→0.040165` |

Другие frozen negative mechanisms: topology-preserving stratified transport,
analog-residual dressing, guidance mixture, coherent/slack offsets, latent
temperature 1.30, locked MC dropout и IID calendar bias mixture. Ни один не
прошёл одновременно proper-score, reliability, boundary и spatial/physical
families. Точные первоисточники: `paper/PAPER_DRAFT.md`,
`paper/CLAIM_LEDGER.md`, `paper/REPRODUCIBILITY.md`.

## CAS-C1 — coarse-budget train64 + frozen fine

Server result:
`.../coarse_budget_train64_paired_388d8ce_v1/coarse_budget_paired_validation.json`.
Fair CRPS `0.0821684→0.0810646` (`−1.343%`), mean SSR
`0.54567→0.55171`, joint ES `0.216942→0.216815`; mean RMSE skill слегка хуже.
SIC rank-TV улучшился, но SIT+3 rank-TV `0.19934→0.24540`, mean rank
`0.494→0.661`. `fine_resampled_for_candidate=false`, поэтому это не end-to-end
draw нового cascade law. Визуально coarse lift нёс около 91.5% variance, а
fine residual часто выглядел как зерно.

## GEO-H1 — endpoint-only suffix continuation

Два matched продолжения EMA6 по 256 slice23 dates: endpoint-score control и
geometry treatment. Native flow-matching anchor отсутствовал; менялся только
terminal suffix.

- fair CRPS против EMA6 хуже на `15.8%/11.2%` для control/treatment;
- native joint ES хуже на `2.69%/2.47%`;
- geometry ES хуже на `3.60%/3.76%`;
- treatment/control geometry-ES ratio `1.0016`, CI включает нулевой эффект;
- treatment rank-TV `0.304…0.487` против EMA6 `0.159…0.220`.

Семплы оставались гладкими; провал — сдвиг conditional law/ranks. Отклонён
именно endpoint-only/slice23/suffix-only protocol, а не geometry-aware CFM.

## GEO-P0 — geometry-CFM preflight

- run: `astra-go-e4f0e06-20260913-0537`;
- ClearML: `d27b3de33642486a99430a35963dfd55`;
- preflight SHA:
  `c4d290e5a467526eb020f1614224d4f90098d395c13c8a61b1c9323d67d626ce`;
- optimizer steps 0, parameters unchanged;
- control native loss `0.0958311483`;
- treatment native/geometry/total `0.0958311483/0.05304547/0.1011356935`;
- peak allocated GPU `13932.8 MiB`.

Prediction parity, adjoints, gradients, evidence lifecycle и one-GPU admission
получили Astra GO.

## GEO-T1 — matched full-CFM control512/treatment512

Обе ветви стартовали из EMA6. Control продолжал native CFM; treatment сохранял
тот же native CFM и добавлял fixed condition-dependent quadratic metric с
`λ=0.1`: valid-ocean block means 160×128/80×64, lead differences,
d0-linearized SIC×SIT product и d0 ice-edge tube. Future truth не используется
как weight; post-hoc clipping отсутствует.

- run: `geometry-full-cfm-ab-8e432ee-20260913-0638`;
- ClearML: `7c02ac408929452a8c45d4f196631a93`;
- exact training commit:
  `8e432ee9404f90d664ac95281af480e1db4de461`;
- result SHA:
  `f5aadba59042b2700c41060b5239886053276ad5f7da096d042899b2afd2542d`;
- control512 raw SHA:
  `569d68d4f142ea8071b6bf1cf60c8a936a1b1038357b16e6939a7ddb94ba1207`;
- treatment512 raw SHA:
  `e1832b3b3b1dee9a16342dd66451ab1efd51ea29754aa494bb12107a87ca8561`;
- 512 updates/arm, batch 8, AdamW LR `1e-5`, bf16 network/FP32 loss;
- exact common 4096 cases/times/noise/dropout;
- mean native pre-step loss `0.0486220/0.0486232`;
- test-2023 false, exit 0, ClearML закрыт до terminal success.

Frozen paired validation получила Astra GO и завершена:

- run: `geometry-full-cfm-paired-56bc3e7-20260913-1124`;
- ClearML: `fd1aee40fef44d03a8c06702426a56f9`;
- exact evaluation commit:
  `56bc3e7b2f74602a16a208dcd84584356c870f3a`;
- laws: EMA6/control512/treatment512;
- protocol: 12 validation-2022 cases × 8 common-noise members, full public
  RK4-17, zero optimizer;
- numerical result SHA:
  `42d8b7be09142893741d024258af904f14eb60ae93d84fdae882fc6602128039`;
- complete tensor SHA:
  `213900d11312e93dc617c51bc95a0277df04169b570ff45bf4227972225c81dc`;
- test-2023 false; все `3×12×8=288` members сохранены до scoring.

Aggregate standardized fair CRPS: EMA6 `0.076124`, control512 `0.090588`,
treatment512 `0.090248`. Treatment лучше matched control только на `0.376%`
(`ratio=0.996239`, paired CI `0.994705…0.998574`), но хуже EMA6 на `18.55%`
(`ratio=1.185538`, CI `1.07014…1.30310`). Native joint energy score:
`0.205588/0.237327/0.236561`; geometry energy score:
`0.118870/0.137907/0.137545` для EMA6/control/treatment соответственно.
Geometry effect относительно control статистически поддержан, однако
replacement gate провален на всех шести SIC/SIT × lead fair-CRPS и RMSE.

Rank histograms всех трёх laws далеки от uniform: pooled distributions имеют
центральный горб, а truth-support diagnostics показывают различное поведение
zero atoms и interior/positive support. Treatment не исправляет эту структуру;
его pooled rank-TV лишь смешанно и несущественно отличается от control/EMA6.
Adjusted spread/skill treatment равен `0.692/0.662/0.656/0.488/0.658/0.502`
для d3 SIC/SIT, d6 SIC/SIT, d9 SIC/SIT: ансамбль всё ещё слишком узкий по
spread-vs-RMSE diagnostic.

Manual visual gate: ориентация одинакова у truth/initial/sample; fixed scales
SIC `0…1`, SIT `0…4`; инверсии, track leakage и salt-and-pepper зерна не видно.
На малоледных летних случаях почти пустые карты являются корректным следствием
fixed scale. Members различаются главным образом на кромке, но paired
treatment-control delta на фиксированных пределах почти нулевой — визуально
подтверждает слишком слабый geometry effect.

Решение: `REJECT` treatment512 как замену EMA6 и не продолжать blind suffix
training. Результат доказывает только существование слабого полезного
geometry-loss direction относительно matched continuation; он не доказывает
калибровку и не оправдывает дальнейшую настройку λ на тех же development cases.

## IDEA-F1 — front-plus-source mixed-support trajectory law

Статус `PROPOSED`; обучение не запускалось. Предлагаемый закон:

`P(Y|C)=P(B,K|C)P(Q,H|B,K,C)`, где `B` — joint ice/open masks, `K` — cap
events, `Q` — interior SIC, `H` — positive SIT для +3/+6/+9. Shared smooth
trajectory latent двигает кромку; lead innovations и nonlocal source branch
моделируют рост/таяние вдали от неё.

Train-only audit без coastline: среди изменившихся established-ice cells внутри
8 px от d0 edge находятся `93.2/86.0/79.0%` на +3/+6/+9. Это поддерживает
front bias; оставшаяся доля мотивирует unrestricted nonlocal source branch, но
не доказывает физический перенос кромки или превосходство этой архитектуры.

Astra verdict после GEO-T1: `HOLD` на GPU, `GO` только на bounded CPU
representation admission. До обучения нужно явно определить `K<=B`, archive
`a_cap`, inactive-coordinate filler law и decoder
`A=B[K*a_cap+(1-K)*a_cap*sigmoid(Q)]`, `H=B*exp(L)`. Дискретный training path
обязан иметь проверенные ненулевые gradients без скрытого STE. Первый CPU gate:
`B80=1{D_omega A>0}`, потеря occurrence changes при restriction,
mixed-support round trip, land invariance и birth/death при пустой исходной
кромке. Прохождение этого gate подтвердит только корректность representation,
не калибровку.

CPU admission выполнен на exact commit
`be2a345ef185794cfeb62003a4cbedca14ef7de7`, без CUDA, только на train
(`test-2023=false`). JSON результата:
`direct_dynamics_mixed_support_cpu_admission_v1/be2a345/admission.json`,
SHA-256 `f340a6623f31df6854266cd24f4e474184b1439173f7ef627d93a029deb37f9f`.
Новые тесты: `5/5 PASS`. Из `86,859` native occurrence changes на 12
strided anchors в coarse 80×64 видимы `58,772` (`67.66%`); среди coarse
изменений `3,084` births и `2,762` deaths. Mixed-support round trip прошёл:
максимальная ошибка SIC `8.20e-8`, SIT `9.99e-7`; land invariance прошёл;
все четыре проверенных gradient norms конечны и положительны. Synthetic
empty-edge birth/death декодируются точно, без STE.

Astra verdict: `GO` для representation и подготовки compact A/B, но `HOLD`
на GPU. Текущий synthetic source check проверяет codec, а не actual source
branch. До zero-update GPU preflight требуется CPU integration будущего runner:
time+d0 conditioning, одинаковое зануление land до forward и sampling,
положительный per-case denominator без `clamp_min(1)`, land-perturbation
regression, а также birth/death вне исходной кромки и ненулевой gradient
параметров настоящей source branch. Поэтому текущий результат является
математическим admission representation, но не доказательством калибровки или
работоспособности конкретного sampler.

Первый falsifier: compact 80×64 joint mask-only front+source против
equal-capacity ordinary conv mask generator, leave-one-train-year-out, joint
mask/edge proper scores, occurrence reliability и 16-member visual review.

## Не перезапускать без новой гипотезы

- blind spread/temperature sweeps;
- endpoint-only H1;
- EMA6/EMA8 checkpoint mixture;
- naive continuous hard cascade без explicit face/atom law;
- 20×16 как единственный источник macro uncertainty;
- AR с teacher-forced future drivers, отсутствующими на inference;
- post-hoc clipping/rounding, выдаваемый за mixed-support calibration;
- selection только по pooled rank histogram или одной паре метрик.

## Шаблон обязательной записи нового run

Каждая новая запись должна содержать: гипотезу и predeclared falsifier;
dataset/split/years/cases и test-access; input/target/support law; source
checkpoint, git/config SHA и ClearML; GPU/optimizer/updates/batch/precision/
solver/members/seeds; proper scores с paired uncertainty; tie/atom-aware ranks;
boundary/spatial/temporal metrics; fixed-scale all-member images; ручной
визуальный verdict; итог `ACCEPT/REJECT/HOLD`; что именно результат
опровергает и чего не опровергает.

Будущие идеи подробно ведутся отдельно в
`/Users/amir/sciml/diffusion_data_assimilation/docs/research/data_driven_calibration_design.md`.
Этот журнал обновляется только после появления проверяемого evidence или
явного изменения статуса run.
