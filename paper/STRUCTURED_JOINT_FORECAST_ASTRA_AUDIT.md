# Structured joint SIC/SIT trajectory: пакет повторного аудита Astra

Дата фиксации: 2026-09-07. Текущий admission status: **server CPU preflight
`ready`; GPU smoke ожидает финального artifact-review Astra**. Реальные
`structured_archive_semantics_audit_v3` и `structured_joint_state_stats_v3`
получены в изолированном server CPU worktree и синхронизированы сюда. Это ещё
не разрешение на full training и не свидетельство качества будущей модели.

После veto-аудита также добавлены: точный `drop_last` budget (31 эпоха;
ожидаемо 4123 шага при старом числе 2143 train cases), runtime budget gate,
raw SIC/SIT validation до fill для каждого lead/lag, обязательная SRAL coverage,
finite loss/gradient guards, атомарный полный epoch-boundary recovery checkpoint
(Accelerate model/optimizer/scheduler/scaler/RNG + EMA/history + config/data/code
contract) до fallible metrics, truth-free inference builder и строгий physical
support evaluator. На сервере пройдены focused 33/33 и полный `tests/` 80/80;
legacy `test/` имеет прежний несвязанный failure occurrence-E2 (90/91).

## 1. Точная целевая задача и граница утверждения

Следующий run оценивает одну совместную условную меру

\[
p_\theta\!\left(X_{d:d+3}\mid B_{d:d+3}, O_d,O_{d-1},O_{d-2},M,c_d\right),
\]

где каждый `X` и `B` — пара SIC/SIT. `B_{d+h}` является только точной записью предыдущего календарного года для даты `d+h`; nearest-date substitution запрещён. Модель генерирует всю траекторию одним 16-канальным Gaussian draw и одним UNet, поэтому зависимость между lead моделируется совместно, а не четырьмя независимыми моделями.

Доказанная временная семантика уже, чем operational forecast: `trajectory_semantics=consecutive_daily_archive_snapshots`, `target_slice_index=23`, `utc_time_coordinate_verified=false`. Допустимые подписи — «analysis snapshot d» и «forecast archive day d+1/d+2/d+3». Подписи `operational_lead`, `23:00_UTC`, `24h_issue_time` запрещены конфигом и preflight. Это последовательные датированные записи архива на одном и том же внутреннем срезе, а не доказанные lead-times от одного issue time.

Текущие наблюдения — точные SIC+SIT значения M2M truth на реальной геометрии SRAL-track. Это OSSE/model-truth experiment, не утверждение об ассимиляции реальных измеренных значений SRAL. Future observations не используются.

## 2. Почему моделируется полное состояние, а не increment

Обучаемая переменная — структурированный codec полного `X_{d:d+3}`; background trajectory остаётся condition. Это естественно для закона состояния с физическими атомами open water и SIC cap: codec прямо представляет эти ветви, тогда как обычный аддитивный residual `X-B` создаёт background-зависимые атомы и усложняет support. Innovations `O-B` передаются как дополнительные условные признаки, но не подменяют target.

Для реальных, шумных наблюдений математически более полный следующий этап должен задавать observation operator `H`, quality/uncertainty и likelihood `p(y|HX)` либо явный noise model. В этом run псевдонаблюдения exact, поэтому искусственный observation noise не вводится.

## 3. Сингулярный posterior и exact subspace

Exact `O_d=(SIC,SIT)` делает физический posterior сингулярным на наблюдённых пикселях; обычный обратимый Gaussian ODE во всём пространстве был бы некорректен. Здесь flow существует только в ненаблюдённом подпространстве.

Для каждого lead пара кодируется четырьмя координатами: occurrence, cap, interior SIC, positive SIT. Получается `Y in R^16`. Пусть `Q(O_d,M)` — маска всех четырёх координат lead 0 на пикселях с точной парой SIC+SIT. Тогда

\[
Y_t=(1-t)Y+t\varepsilon,\qquad v^*=\varepsilon-Y,\qquad t\sim U[0,1],
\]

но на `Q` clean state, noise, velocity и loss равны нулю на всём ODE. После интегрирования от `t=1` ровно до `t=0` decoder вставляет точную физическую пару `O_d`. На lead 1–3 latent state/noise/loss остаются активными. Маска loss зависит только от static water/padding и observation geometry; она никогда не зависит от target occurrence или cap.

`stratified_uniform` даёт один случайный `t` в каждом из `batch_size` равных интервалов и случайно переставляет их. Это сохраняет равномерную целевую меру и уменьшает дисперсию mini-batch без Beta-перевеса.

## 4. Structured codec

На каждом ocean pixel:

1. `z_occ`: dequantized binary coordinate для совместного атома `(SIC,SIT)=(0,0)` против ice.
2. `z_cap`: dequantized binary coordinate для `SIC=sic_cap` против interior, активный по физическому смыслу только при ice.
3. `z_sic`: standardized `logit(SIC/sic_cap)` на interior branch.
4. `z_sit`: standardized `log(SIT)` на ice branch.

Decoder создаёт exact open-water atom и exact `sic_cap=0.9970703125`; отрицательные SIT и несовпадающий occurrence SIC/SIT невозможны. Неактивные intensity coordinates заполняются независимым `N(0,1)` и decoder их игнорирует. Никакого ложного утверждения, что fillers скрывают spatial/joint class information, нет. Interior float16 значения моделируются как `continuous_resolution_approximation`, а не как полный categorical закон всех quantized уровней.

Все стандартизации codec получены только из train: 2143 anchors x 4 archive days = 8572 samples, 416916364 valid-ocean values. `p(ice)=0.25964502798935474`, `p(cap|ice)=0.0509974751931545`. Train-stats привязаны к data config и exact pair manifest.

## 5. Conditioning: ровно 50 входных каналов

| Global channels | Condition-local | Содержание |
|---:|---:|---|
| 0–15 | — | Текущее 16-канальное latent trajectory state |
| 16–17 | — | Нормированные x/y grid coordinates |
| 18–20 | 0–2 | `B_d`: normalized SIC, normalized SIT, valid background mask |
| 21–23 | 3–5 | `B_d+1`: SIC, SIT, mask |
| 24–26 | 6–8 | `B_d+2`: SIC, SIT, mask |
| 27–29 | 9–11 | `B_d+3`: SIC, SIT, mask |
| 30–34 | 12–16 | lag0: SIC/SIT values, SIC/SIT innovations, track mask |
| 35–39 | 17–21 | lag1: values, innovations, mask |
| 40–44 | 22–26 | lag2: values, innovations, mask |
| 45 | 27 | static valid-water mask |
| 46–47 | 28–29 | day-of-year sin/cos |
| 48–49 | 30–31 | archive slice-index sin/cos (не UTC hour) |

Отсутствующее condition кодируется `value=0, mask=0`. Наблюдавшийся open water кодируется физическим нулём до нормализации, то есть отрицательным normalized SIC/SIT при `mask=1`; эти случаи различимы. Все четыре background dates явные, exact previous-calendar-year и со своими mask. Conditioning dropout/CFG отключены; train и inference используют режим `both` с вероятностью 1.

Causality test меняет future truth и future SRAL track и подтверждает неизменность condition; изменение future background меняет только соответствующие background channels. В condition входят только lag0/1/2 tracks, ни одного future observation channel нет.

Отдельный `M2MForecastDataset.build_structured_forecast_item` строит тот же
condition без `truth`/`structured_physical_truth`: нужны только d,d-1,d-2
archive values на track geometry и exact previous-calendar-year backgrounds
для d..d+3. Unit contract удаляет будущие target-файлы d+1..d+3 и всё равно
строит forecast item; sampler принимает его без будущей истины.

## 6. Полный архивный аудит и missingness

Текущий `config/data/m2m_2f_structured_joint_archive_audit.json` — принятый
server-generated v3 artifact. Он хеширует релевантное содержимое forecast
slices, байты finite/binary static mask и все runtime-visible SRAL-файлы;
SRAL проверяется ровно runtime-порядком ordered first-finite combine → transform
→ ocean mask. Отсутствующая дата является явным observation-unavailable
состоянием, а не ошибкой; полностью пустой источник или split/lag без единого
usable observation отвергается. Любой content/path/shape/inventory drift
отвергается.

- SHA256 артефакта: `e20b851c5f55296077ce835b458afb0afc3a759e399e375c1556befba2b25f56`.
- Forecast manifest: `f8a9014de7a93f842774d2bca643f51258a5fed02143b83197517524ada2a8be`.
- Forecast content SHA: `d5ae6899a2e942af0c44a62f29f0e6aeb019a80b6ddd3120a094ac8fa1b05dc6`.
- Static mask SHA: `d49e5c196496ba0e95bdd03d6e81895f5443dd96728c44835c26359a93560b56`; SRAL content manifest SHA: `ec340c2269f7793d18b44bd658316fe9586fe2379e470f49c0adb38b8640d4ee`.
- 3099 файлов, даты 2015-01-17…2023-07-19; все `float16`, shape `(15,24,311,225)`; duplicate archive dates = 0.
- SIC/SIT ocean NaN mismatch = 0; joint ocean NaN = 109523494; после joint NaN->(0,0): occurrence mismatch = 0, support violations = 0, negative SIT = 0, Inf = 0.
- Finite ocean SIC/SIT = 41202569; exact finite zero = 0. Поэтому правило подтверждено архивом: static mask решает land; paired SIC/SIT NaN на valid ocean означает open water `(0,0)`; Inf или mismatched missingness немедленно отвергаются.
- Train: 2143 retained exact trajectories; validation: 361; test: 197. Все retained targets являются `d,d+1,d+2,d+3`, лежат внутри своего split и имеют exact previous-calendar-year background для каждого lead. Неполные кандидаты исключаются, не интерполируются.
- SRAL availability: train lag0/1/2 usable dates = 1073/1069/1066 из 2143/2137/2133 candidates (около 50%); validation = 361/360/359 из 361/360/359; test = 197/197/197. Missing train dates остаются mask=0, а не заполняются искусственной геометрией.

Currents indices 6/7 имеют 7034730 ocean NaN на канал; winds 13/14 — 0 ocean NaN. Они audited только как archive-unit features. Upstream преобразует их в signed grid displacement через `calculate_flows_l/FlowL`; знак, ориентация, физические единицы и causal issue-time не доказаны. Поэтому clean run намеренно ставит `dynamic_forcing_indices=[]`: winds/currents не включены, NaN не заполняются и future `_48/_72` не используются.

## 7. Split, checkpoint, sampling и метрики

- Train target dates: 2016-01-01…2021-12-31; validation: 2022; test: 2023-01-01…2023-07-19. Horizon обрезается до границы каждого split.
- Frozen plan: 31 epochs, batch 16, `drop_last=true`. На реальных 2143 train
  cases runtime даёт floor(2143/16)=133 batches/epoch и 4123 optimizer steps;
  warmup 200, predeclared minimum 4000. Server preflight и `main` проверяют
  фактический `len(train_loader)` до создания модели.
- Diffusers `EMAModel`, declared schedule `decay=max(min_decay,min(ema_decay,(1+step)/(10+step)))`, `ema_decay=0.999`, no warmup, update-after=0, min-decay=0. Terminal instantaneous decay at planned step: 0.9978218780251694.
- Validation loss вычисляется на EMA weights; best checkpoint выбирается по minimum EMA validation flow loss; sampling тоже использует EMA. Raw/EMA selection mismatch отсутствует.
- До fallible ensemble metrics/dashboard атомарно фиксируется epoch-boundary
  recovery directory + pointer: Accelerate state (model, optimizer, scheduler,
  scaler, RNG), отдельный EMA state, validation history/best и SHA contract
  config/data/audit/code. Payload manifest проверяется при resume; non-finite
  train loss/gradient и validation loss немедленно завершают run без silent skip.
- Dedicated 16-channel trajectory sampler интегрирует к `t=0`, декодирует 8 физических каналов и вставляет exact lag0 pairs.
- Dedicated evaluator считает по каждому lead и полю ensemble-mean RMSE, background RMSE, fair CRPS, tie-correct fractional rank counts, support violations и exact lag0 error. Dashboard рисует крупную матрицу truth/background/member для SIC и SIT на всех четырёх датах; `origin=lower`.
- Run-time monitoring: 4 validation cases x 5 members every 10 epochs, dashboard every 5 epochs. Это smoke/monitoring, не publication evidence. Для статьи нужен отдельный held-out test protocol с существенно большим ensemble и predeclared rank/CRPS/spatial gates.

## 8. Preflight и тесты

Текущий `paper/STRUCTURED_JOINT_FORECAST_PREFLIGHT.json` имеет status
`ready_for_gpu_smoke_pending_astra_artifact_review`. Server CPU preflight
проверил v3 audit/stats, exact hashes, train-only normalization, шесть реальных
train/validation cases, truth-free forecast builder и полный model build:
38911408 parameters, input 50, output 16, smoke output `[1,16,32,32]`,
`cuda_initialized=false`.

На том же синхронизированном server snapshot: focused 33/33 PASS, полный
`tests/` 80/80 PASS. Legacy `test/` — 90/91; единственный failure
`test_occurrence_intensity_e2.E2OperatorTimeMappingTest.test_each_operator_uses_its_matching_generated_state`
существовал до этой работы и не касается structured-joint diff. Все CPU-команды
выполнялись с `CUDA_VISIBLE_DEVICES=""`. Первоначальная медленность исчезла при
`OMP_NUM_THREADS=1`; это был CPU OpenMP oversubscription, не дефект модели.

## 9. Оценка времени и single-GPU safety

Старый live baseline имеет 3241 batch/epoch, а новый fixed-slice trajectory run
только 133 batch/epoch, но существенно более широкий state. Поэтому переносить
старое время эпохи и объявлять 28–35 часов некорректно. Wall-time, peak memory,
step time и sampling NFE будут оценены только коротким GPU smoke; до него ETA
не заявляется. Batch-16 memory на 32 GB ещё не доказана.

Перед launch обязательны: veto Astra снят; ровно одна разрешённая GPU; если память свободна — запуск, если занята — 5 минут наблюдать utilization и запускать только при среднем/устойчивом utilization <5%; никакого вмешательства в чужие процессы; live run не останавливать; Tailscale/SSH watchdog не трогать. Первый разрешённый GPU шаг должен быть коротким memory/throughput admission, затем тот же frozen config либо fail closed.

## 10. Глобальный ERA5 на одной GPU около 32 GB

Число grid tokens: 1° — `180x360=64800`; 0.5° — `360x720=259200`; 0.25° — `720x1440=1036800`. Одна только fp16 attention-logit matrix на один head/layer занимает соответственно около 8.4 GB, 134 GB и 2.15 TB, без softmax, QKV, gradients и optimizer. Глобальный quadratic attention неприемлем уже на 1°.

Естественный baseline — multiscale convolutional UNet: periodic longitude; pole reflection/roll, причём vector channels требуют корректного basis rotation/sign; loss с `cos(latitude)` area weights. Attention допустим только на coarsest grid либо local windows. Coarse FNO + local convolution hybrid сохраняет глобальный контекст без `N^2`; FNO тоже требует ограничения spectral modes. Icosahedral/cubed-sphere convolution геометрически чище для глобуса, но remapping и seams добавляют отдельный методический риск, поэтому это не первый clean baseline.

Для surface 2D SIC/SIT global convolution training реалистичен: 1° и 0.5° комфортно, 0.25° вероятно batch 1 с mixed precision, activation checkpointing и умеренной width. Для 20–50 2D fields 1°/0.5° реалистичны, 0.25° теснее, но возможен с factorized heads/latent bottleneck. Full 3D atmosphere не помещается наивно: `1400` channels (пример 20 variables x 70 levels) на 0.25° — около 2.9 GB bf16 только raw input одного sample до activations/output/gradients; нужен vertical/column encoder, latent factorization и/или domain decomposition.

Patch training допустим только с глобально согласованным inference: одна global noise realization, global coarse context, halo overlap и blending, одинаковые boundary rules, отдельный seam/calibration gate. Независимые patch draws не дают корректной глобальной joint distribution.

## 11. Точная область diff для Astra

Изменённые scoped tracked files: `assim_lib/config.py`, `assim_lib/data.py`,
`assim_lib/main.py`, `assim_lib/runtime.py`, `assim_lib/sampler.py`,
`assim_lib/trainer.py`, `tests/test_distribution_preserving_training.py`.

Новые scoped files: `assim_lib/structured_archive_audit.py`,
`assim_lib/structured_joint_preflight.py`, `assim_lib/structured_joint_state.py`,
`assim_lib/structured_joint_stats.py`, `assim_lib/structured_trajectory_evaluation.py`,
три structured data JSON, method JSON, experiment JSON, publication-evaluation
protocol, preflight JSON и `tests/test_structured_joint_training.py`.

Следующие dirty files существовали до этой задачи и не редактировались: `assim_lib/occurrence_intensity_e1_data_audit.py`, `paper/OCCURRENCE_INTENSITY_E1_REAL_CASES.json`, `paper/validate_occurrence_intensity_e1_data_audit.py`, `test/test_occurrence_intensity_e1_data_audit.py`, `assim_lib/bounded_sic.py`, `paper/analyze_residual_rank10.py`, `test/test_bounded_sic.py`.

Core SHA256: `structured_joint_state.py=fc488b1a...`, `data.py=dd98d86e...`,
`main.py=3b9b7c01...`, `runtime.py=6ade7d03...`,
`trainer.py=56d9c6dc...`, `sampler.py=54f7f14b...`,
`structured_archive_audit.py=3c8dc044...`,
`structured_joint_preflight.py=62d8bc41...`,
`structured_joint_stats.py=c86ba2d4...`,
`structured_trajectory_evaluation.py=1f783fe8...`, method config
`fa2f7da9...`, experiment config `704eda23...`, focused test
`9df3482f...`. Полный machine-readable manifest связывает локальный и
server-tested snapshots отдельно. Astra должна читать exact scoped files, а не
смешанный общий `git diff`, потому что worktree содержит перечисленные чужие
изменения.

## 12. Нерешённые ограничения

1. UTC coordinate и operational issue-time остаются неизвестными; текущий law намеренно не делает этих claims.
2. SRAL values пока pseudo-observations из model truth; observation-error/quality likelihood для реальных измерений ещё не реализован.
3. Wind/current causality, sign/orientation и missingness handling не готовы; признаки исключены.
4. GPU memory/throughput для batch 16 не измерены; нужен короткий admission после снятия veto.
5. Monitoring ensemble 5 слишком мал для публикационной оценки калибровки; нужен отдельный frozen held-out evaluation.
6. Архивные interior float16 уровни аппроксимируются непрерывным законом; exact categorical claim делается только для structural open-water и SIC-cap atoms.
