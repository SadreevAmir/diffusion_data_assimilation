# Многоуровневый вероятностный каскад SIC/SIT: математическая постановка и проверяемый план

Дата: 2026-09-13

Статус: независимый research memo; не является заявлением о полученном улучшении

Область: динамика морского льда на горизонтах `+3/+6/+9` суток; test-2023 не используется

## Краткий вывод

Радикальный каскад

```text
20x16 -> 40x32 -> 80x64 -> 160x128 -> 320x256
```

можно сформулировать как точную вероятностную факторизацию, а не как набор
эвристических super-resolution сетей. Для этого уровни должны моделировать
аддитивные физические величины — площадь льда и объём льда — и на каждом шаге
стохастически распределять родительские бюджеты между дочерними ячейками с
точным сохранением суммы. Такая конструкция samplewise сохраняет уже
сгенерированные coarse marginals и физически естественнее аддитивного residual
по SIC/SIT.

Однако полезность пяти уровней не следует из корректности факторизации.
Дополнительные уровни увеличивают число условных ядер, teacher-forcing mismatch
и стоимость sampling. Более того, текущая реализация `masked_block_average`
может быть корректно продолжена на много уровней только при переносе полной
площади воды, а не бинарного признака active coarse cell. Повторение нынешнего
`space-to-depth` conditioning до `20x16` также непрактично: 11 пространственных
каналов превратятся в `11 * 16^2 = 2816` каналов.

Поэтому первым следует реализовывать не пятиуровневую систему, а проверяемый
трёхуровневый вариант

```text
80x64 -> 160x128 -> 320x256
```

с точным area/volume conservation. Отдельный очень дешёвый `20x16-only`
эксперимент должен заранее проверить, способен ли экстремально coarse law
воспроизводить крупномасштабную неопределённость. Переход к `40x32` и тем более
полной пятиуровневой цепочке оправдан только после прохождения заранее
зафиксированных gates. Если идея не улучшает calibration на агрегированных
масштабах или ухудшает native fair CRPS/texture при равном compute, её следует
отклонить.

## 0. Пять решений, от которых действительно зависит результат

До обсуждения числа блоков и ширин нужно зафиксировать пять научных решений.
Именно они, а не глубина каскада, определяют шанс получить сильный результат.

1. **Моделировать один joint conditional law всей траектории.** Один member
   обязан содержать согласованные `SIC/SIT` на `+3/+6/+9`; отдельные модели или
   независимый noise для горизонтов не описывают нужную временную copula.
2. **Перейти от `(SIC,SIT)` к аддитивным `(A,V)`, где `V=A*SIT`.** Только так
   coarse quantities имеют однозначный физический смысл и могут сохраняться
   samplewise. При этом open-water и compact-ice atoms нельзя выдавать за
   обычную евклидову плотность.
3. **Сначала проверить, что дефицит калибровки действительно масштабный.**
   Multiscale losses на одной модели — более дешёвая причинная проверка, чем
   немедленное строительство пяти генераторов. Если они не меняют coarse rank и
   spread, каскад лечит не тот механизм.
4. **Если нужен hierarchy, генерировать бюджеты и условные allocations, а не
   произвольные residual.** Это даёт точную поддержку и conservation. Но каждый
   дополнительный stochastic kernel должен оправдать себя относительно более
   простого `coarse global latent + one conditional refinement`.
5. **Решение принимать по joint и scale-resolved proper scores.** Красивый SIC,
   один средний CRPS или ровный pooled rank histogram недостаточны. Нужны также
   SIT, все lead times, area/volume laws, spatial/temporal dependence и
   индивидуальные sample maps.

### Что является центральной концепцией

Центральный кандидат для статьи: **joint mixed-support generative forecast в
координатах area/volume, где coarse stochastic budgets сохраняются каждым
native-resolution member, а калибровка измеряется на нескольких физических
масштабах**. Это содержательнее и проще защитить, чем claim «мы добавили больше
resolution stages».

Число уровней, shared colored noise и end-to-end score fine-tuning — опциональные
механизмы. Пятиуровневая цепь, независимые локальные блоки и post-hoc inflation
не являются центральной концепцией и заранее не рекомендуются.

### Альтернативы, с которыми каскад обязан конкурировать

| Идея | Механизм | Возможное преимущество | Главный риск | Дешёвый falsification test |
|---|---|---|---|---|
| Single multiscale flow | Один native joint flow; loss и scores на `320/160/80/40/20` restrictions | Нет teacher-forcing mismatch; самый короткий путь | Coarse loss может не заставить noise нести верный low-frequency spread | Фиксированный короткий run против того же flow без pyramid loss; сравнить rank/spread по масштабам |
| Coarse global law + one refinement | Генератор `80x64` budgets и один native conditional allocation внутри `4x4` blocks | Один conditional kernel вместо двух-четырёх; точное coarse conservation | 16-child allocation сложнее; возможны block seams | CPU decoder + небольшой overfit/validation pilot на `80->320`; проверить seams и conditional sensitivity |
| Spectral/Laplacian flow | Совместно генерировать low modes и условные band-pass coefficients | Естественное разделение пространственных частот | Coast/mask делают базис неортогональным; bounds и atoms не сохраняются | На truth измерить reconstruction/support после exact masked basis; если нужен частый clipping — отклонить |
| Stochastic residual dynamics с conserved low modes | Детерминированный/persistence центр плюс stochastic residual, чей coarse projection генерируется отдельно | Может быстро исправить недодисперсию крупных масштабов | Residual law зависит от состояния и хуже представляет atoms | Сравнить с full-state flow при том же compute; проверить occurrence/cap Brier и tail ranks |
| Autoregressive scale-space allocation | `80->160->320` или длиннее, каждый шаг делит area/volume budgets | Точная физическая интерпретация и scale diagnostics | Накопление kernel error и rollout shift | Matched-compute `80->160->320` против current `160->320` и one-refinement |

Мой независимый prior таков: **первый эксперимент — single multiscale-flow
control; первый физический новый метод — `80x64` global budget law плюс один или
два constrained allocation refiners**. Полная пятиступенчатая цепь имеет смысл
только как поздняя ablation, а не как исходная ставка.

## 1. Зафиксированные факты текущего проекта

Настоящий memo опирается на следующие уже проверенные свойства данных и кода.

1. Native grid имеет размер `320x256`, static valid-ocean mask — 48 637
   валидных ячеек.
2. Целевая trajectory содержит совместно SIC/SIT на `+3/+6/+9` суток.
3. В train archive имеется совместный open-water atom:
   `SIC=0 <=> SIT=0`. Оценённая частота льда около `0.2596`.
4. SIC имеет верхний archive atom около `0.9970703125`, приблизительно `5.1%`
   среди ледовых ячеек. Его exact frequency является прежде всего encoding
   diagnostic; научно значимее truth-referenced события `SIC>=0.90/0.95/0.99`.
5. Текущий двухуровневый каскад использует masked average `D`, right inverse
   `U` и fine residual в `ker(D)`. Это точная линейная декомпозиция обычных
   полей на одном переходе `160x128 -> 320x256`.
6. Fine model обучается с truth coarse condition, а inference получает
   generated coarse condition.
7. Текущий raw `EMA9711 + colored fine2048` на diagnostic `12 cases x 8
   members` превосходит persistence по среднему fair CRPS, но остаётся сильно
   недодисперсным: mean spread/skill около `0.555`; максимальная member/truth
   roughness ratio около `2.15`.
8. Fine-only calibration была отрицательным механизмом, а sensitivity analysis
   показал, что основной proper-score gradient требует изменения coarse budget.
9. Новый 64-update frozen-allocation coarse-budget run завершён механически,
   но без отдельного paired validation ещё не является научным результатом.
10. Семантика оси из 24 archive slices не подтверждена как UTC time:
    `utc_time_coordinate_verified=false`. До публикационного обучения нельзя
    считать эти slices независимыми временными примерами без отдельного аудита.

Многоуровневая гипотеза мотивирована пунктами 7–8: крупномасштабная
неопределённость действительно потеряна. Но эти факты не доказывают, что большее
число уровней восстановит её.

## 2. Целевой условный закон траектории

Пусть `c` обозначает всю информацию, реально доступную при выпуске прогноза:

- начальное состояние или sampled analysis на день `d0`;
- static water/land geometry и grid metrics;
- календарные признаки;
- только те wind/current forcings, которые доступны в заявленном forecast
  protocol.

Будущую reanalysis forcing нельзя использовать как condition, если она не была
доступна на `d0`; это было бы leakage. Если используется future prescribed
forcing, эксперимент обязан быть так и назван.

Для горизонтов `T={3,6,9}` целевой закон равен

\[
P^*(X\mid c),\qquad
X=\{(A_t,H_t):t\in T\},
\]

где `A=SIC`, `H=SIT`. Это совместный закон по пространству, полям и времени, а
не произведение pixelwise или leadwise marginals.

Для физической иерархии удобнее заменить thickness на ice-volume density

\[
V_t(p)=A_t(p)H_t(p).
\]

На физической поддержке отображение

\[
(A,H)\mapsto(A,V=AH)
\]

биективно при соглашении `H=0` на `A=0`: если `A>0`, то `H=V/A`, а если
`A=0`, совместная поддержка требует `V=0` и восстанавливается `H=0`.

Таким образом, основной trajectory law можно записать как

\[
P^*\!\left(
 A_3,V_3,A_6,V_6,A_9,V_9\mid c
\right).
\]

Важно: даже если `A` и `V` используются как координаты генератора, итоговые
SIC/SIT metrics считаются после точного обратного преобразования.

## 3. Смешанная мера и атомы

На одной native water cell физическая мера имеет схематичный вид

\[
\mu(dA,dV)=
\pi_0\,\delta_{(0,0)}
+\pi_{cap}\,\delta_{A_{max}}(dA)\,\nu_{cap}(dV)
+f(A,V)\,\mathbf 1_{0<A<A_{max},\,V>0}\,dA\,dV.
\]

Здесь:

- `A=0,V=0` — совместный open-water atom;
- `A=A_max,V>0` — censored/high-concentration branch;
- interior branch непрерывен;
- мера всей пространственно-временной trajectory не факторизуется по клеткам.

Обычный smooth Euclidean flow с absolutely continuous base law не может
создать exact atom посредством диффеоморфизма. Корректны два класса решений:

1. смешанный discrete/continuous generator;
2. continuous dequantization атомов с детерминированным threshold decoder и
   auxiliary filler coordinates на неактивных ветвях.

Второй путь уже поддерживается идеями `structured_joint_state.py` и реалистичен
на одной GPU. Для multilevel area/volume cascade hurdle variables должны быть
согласованы с conservation decoder, а не применены независимым clipping после
него.

На coarse cells атомы также существуют:

- нулевой budget, если все water children свободны ото льда;
- максимальный area budget, если все water children находятся на верхней
  границе;
- смешанная interior мера в остальных случаях.

## 4. Area-weighted grids и restriction

### 4.1 Native веса

Пусть `g_p>0` — физическая площадь native grid cell, а `m_p in {0,1}` —
valid-water mask. Определим

\[
\omega_p=g_p m_p.
\]

Если проекция grid действительно equal-area, можно положить `g_p=1`. Это
предположение должно быть доказано метаданными grid; текущий бинарный mask сам
по себе его не доказывает.

### 4.2 Аддитивные бюджеты

Для coarse cell `B`:

\[
\Omega_B=\sum_{p\in B}\omega_p,
\]

\[
S^A_B(t)=\sum_{p\in B}\omega_p A_t(p),
\qquad
S^V_B(t)=\sum_{p\in B}\omega_p V_t(p).
\]

Если `Omega_B>0`, для отображения можно хранить средние

\[
\bar A_B=S^A_B/\Omega_B,
\qquad
\bar V_B=S^V_B/\Omega_B,
\]

но conservation следует формулировать через totals `S`, а не средние.
Land-only cell имеет `Omega_B=0`, оба бюджета фиксируются в нуле и исключаются
из loss.

### 4.3 Частично водные coarse cells

Если в родительском блоке только часть детей содержит воду, максимальная масса
концентрации равна

\[
C_B=A_{max}\sum_j\Omega_{Bj},
\]

а не площади полного прямоугольника. Land child имеет capacity zero и никогда
не получает area/volume budget.

Критически важно переносить `Omega` рекурсивно:

\[
\Omega_{parent}=\sum_j\Omega_{child,j}.
\]

Текущий `masked_block_average` правильно возвращает `ocean_fraction` для одного
factor-2 перехода при equal-area native cells. Но при следующем coarsening
нельзя заменить дробную площадь бинарным `active = ocean_fraction > 0`: тогда
маленькая прибрежная water sliver получит тот же вес, что полностью водная
coarse cell. Многоуровневая реализация обязана coarsen накопленные area weights.

### 4.4 Exact samplewise conservation

Для каждого case, member, lead, parent и поля требуется

\[
S^A_{parent}=\sum_j S^A_{child,j},
\qquad
S^V_{parent}=\sum_j S^V_{child,j}.
\]

Это проверяется до усреднения по ensemble. Aggregate equality недостаточна:
она допускает компенсацию ошибок между members и уничтожает coarse copula.

Обычный right inverse `U` существует, например constant replication на water
children. Текущий `smooth_right_inverse` также обеспечивает `D U=I` для
линейного поля. Но для bounded SIC и nonnegative volume более естественно сразу
генерировать допустимые child allocations; аддитивный lift может выйти за
физическую поддержку и потребовать нелинейную проекцию.

## 5. Точная параметризация child allocations

Рассмотрим один parent с `n<=4` water children. Пусть

\[
c_j=A_{max}\Omega_j
\]

— максимальный area budget ребёнка, а `S` — area budget родителя.
Допустимое множество

\[
\mathcal C(S)=\{x:\;0\le x_j\le c_j,\;\sum_jx_j=S\}
\]

является capped simplex и непусто тогда и только тогда, когда

\[
0\le S\le C=\sum_jc_j.
\]

### 5.1 Interior capped-simplex map

Для фиксированных конечных logits `z_j` и `c_j>0` положим

\[
x_j(\lambda)=c_j\sigma(z_j+\lambda),
\]

где `sigma` — logistic sigmoid. Определим

\[
F(\lambda)=\sum_j c_j\sigma(z_j+\lambda).
\]

**Лемма.** Если хотя бы одно `c_j>0` и `0<S<C`, существует единственное
конечное `lambda`, для которого `F(lambda)=S`.

**Доказательство.** `F` непрерывна. Для каждого положительного `c_j`
производная `c_j sigma(1-sigma)>0`, поэтому сумма строго возрастает.
Пределы при `lambda -> -infinity/+infinity` равны `0/C`. Утверждение следует из
теоремы о промежуточном значении и строгой монотонности. QED.

Практически `lambda` находится bracketed bisection или safeguarded Newton.
Root solve должен иметь residual gate в физических totals.

Исключения:

- при `S=0` конечного корня нет; единственное решение `x_j=0`;
- при `S=C` конечного корня нет; единственное решение `x_j=c_j`;
- child с `c_j=0` детерминированно получает zero;
- при единственном active child allocation детерминирован;
- logits неидентифицируемы относительно общего сдвига, потому что сдвиг
  поглощается `lambda`; следует фиксировать `sum z_j=0` или удалить одну
  степень свободы.

### 5.2 Boundary faces и mixed support

При `0<S<C` sigmoid map создаёт только `0<x_j<c_j`. Он не способен положить
положительную массу на faces, где отдельный child равен нулю или capacity.
Поэтому для точных child atoms нужна дополнительная active-set переменная.

Пусть:

- `K` — множество saturated children с `x_j=c_j`;
- `Z` — множество zero children;
- `I` — остальные interior children.

После фиксации `K,Z` residual budget равен

\[
S'=S-\sum_{j\in K}c_j,
\]

и interior map применим, если

\[
0<S'<\sum_{j\in I}c_j.
\]

Если `I` пусто, pattern допустим только при точном равенстве residual budget
нулю. Для четырёх детей число patterns конечно; generator может моделировать
их dequantized categorical/hurdle coordinates, а decoder обязан отбрасывать
неосуществимые patterns либо параметризовать только осуществимые ветви.

Нельзя независимо threshold-ить четыре Bernoulli heads, а затем silently
project: это меняет объявленный target law и создаёт трудно контролируемые
атомы.

### 5.3 Volume allocation

После area allocation пусть `J={j:x_j>0}`. При родительском volume budget
`T>0`:

\[
v_j=T\frac{\exp q_j}{\sum_{k\in J}\exp q_k},\quad j\in J,
\qquad
v_j=0,\quad j\notin J.
\]

Тогда автоматически:

- `v_j>0` на ice children;
- `v_j=0` на open-water children;
- `sum v_j=T` точно с точностью root/softmax arithmetic.

При `T=0` и `S=0` единственное решение — все `v_j=0`. Случай `T=0,S>0`
противоречит принятой archive support `SIC>0 <=> SIT>0` и должен fail closed,
если это не объяснённый quantization case. Как и area logits, `q` определены с
точностью до общего сдвига.

Никакой верхней границы thickness из текущего data contract не следует.
Искусственный SIT cap вводить нельзя; достаточно positivity, finite checks и
train-support diagnostics.

### 5.4 Восстановление физических полей

Для water child:

\[
A_j=x_j/\Omega_j,
\qquad
V_j=v_j/\Omega_j,
\]

\[
H_j=\begin{cases}
V_j/A_j,&A_j>0,\\
0,&A_j=0.
\end{cases}
\]

Деление при очень малом положительном `A` может усиливать численную ошибку.
Это не даёт права менять физический закон epsilon clipping. Следует считать
decoder в FP64, фиксировать минимальное положительное archive-resolvable `A`
как часть mixed support либо моделировать positive `log H` совместно и
проверять согласованность volume. Выбор должен быть сделан по train inventory.

## 6. Вероятностная факторизация по уровням

Пусть `X_L` — native trajectory в координатах `(A,V)`, а

\[
X_\ell=D_{\ell\leftarrow L}X_L.
\]

Поскольку каждый более грубый уровень является детерминированной функцией
следующего, chain rule даёт

\[
P^*(dX_L\mid c)=
P^*_0(dX_0\mid c)
\prod_{\ell=1}^{L}
K^*_\ell(dX_\ell\mid X_{\ell-1},c),
\]

где kernels поддержаны на fibers

\[
\{x_\ell:D_\ell x_\ell=x_{\ell-1}\}.
\]

Если allocation coordinates `Q_l` вместе с parent однозначно определяют child,
можно эквивалентно моделировать

\[
P^*_0(dX_0\mid c)
\prod_{\ell=1}^{L}
P^*_\ell(dQ_\ell\mid X_{\ell-1},c).
\]

Это точная статистическая факторизация, а не предположение о независимости
масштабов. Предположение появляется только в выбранной neural parameterization
условных kernels.

Для реалистичного первого варианта:

```text
X0: 80x64 joint area/volume trajectory
Q1: 80x64 parent -> 160x128 allocations
Q2: 160x128 parent -> 320x256 allocations
```

Все шесть trajectory channels `(A3,V3,A6,V6,A9,V9)` генерируются совместно на
каждом уровне.

## 7. Copula и noise law

### 7.1 Почему независимые блоки неверны

Независимый Dirichlet/logistic-normal draw в каждом 2x2 parent block приводит
к следующим дефектам:

- видимым block seams;
- неверному спектру и variogram;
- отсутствию coherent ice-edge displacement;
- разрушению совместных SIC/SIT structures;
- независимым изменениям на +3/+6/+9 вместо траектории.

Conservation сам по себе сохраняет только coarse sums, но не spatial copula.

### 7.2 Условные field generators

Каждый `Q_l` должен генерироваться convolutional field model, который видит:

- весь parent `(A,V)` trajectory;
- multi-scale features начального состояния;
- water-area/capacity maps;
- допустимые forcing features;
- предыдущие refinement features;
- все горизонты одновременно.

Output — spatial fields allocation logits/status coordinates, после чего
локальный физический decoder обеспечивает budgets.

### 7.3 Связь noise между масштабами

Удобен nested Gaussian law:

\[
Z=U_0Z_0+\sum_{\ell=1}^{L}U_\ell R_\ell,
\]

где `Z0` — coarse random field, а `R_l` — scale-band innovations. Для каждого
member используются фиксированные case/member/level seeds. Компоненты могут
быть независимы условно на parent, но member identity никогда не меняется.

Shared coarse latent несёт крупномасштабную межгоризонтную зависимость;
scale-specific innovations добавляют условную неопределённость. Colored base
covariance можно оценить только на train detail coefficients. Она является
preconditioning/base-law choice, но не доказательством calibration.

### 7.4 Copula SIC/SIT и времени

Area pattern и volume allocation нельзя генерировать отдельными независимыми
сетями: volume support зависит от area support. Минимум нужен общий backbone и
условная факторизация

\[
P(Q^A,Q^V\mid parent,c)
=P(Q^A\mid parent,c)
 P(Q^V\mid Q^A,parent,c).
\]

Аналогично, отдельные модели по lead разрушат trajectory copula. Общий output
для `+3/+6/+9` — обязательное условие direct-trajectory версии.

## 8. Conditioning pyramid

Нынешний `lossless_coarse_condition` использует factor-2 `pixel_unshuffle`:
11 spatial channels становятся 44 каналами на `160x128`. Повторять это до
`20x16` нельзя: factor 16 дал бы 2816 spatial channels.

Нужен общий deterministic condition encoder с features на всех уровнях:

```text
encoder(initial state, forcing, masks, grid metrics)
  -> E20, E40, E80, E160, E320
```

Он может быть лёгким convolutional pyramid. Lossless transmission каждого
native pixel на самый coarse level не требуется для статистической
корректности; требуется достаточная информация для целевого conditional law.
Однако следует провести ablation против area-weighted summaries и убедиться,
что coastline/initial ice edge не потеряны.

Calendar features остаются отдельными constant channels. Grid coordinates
должны включать не только normalized `x,y`, но и physical cell area/metric, если
grid не equal-area.

## 9. Training objectives и идентифицируемость

### 9.1 Conditional flow matching

Для каждого уровня можно обучать flow на dequantized coordinates target
`Y_l` и base `E_l`:

\[
Y_t=(1-t)Y_l+tE_l,
\qquad
u_t=E_l-Y_l,
\]

\[
L_{FM}=\mathbb E
\|v_\theta(Y_t,t,condition)-u_t\|_W^2.
\]

При population optimum, корректной path construction, достаточной capacity и
точном ODE conditional flow идентифицирует целевой dequantized law. Из этого не
следует finite-model calibration.

Weights `W` должны учитывать physical water area и train-only channel scales.
Primary domain — вся вода; regime splitting допустим как diagnostic, но не как
скрытая смена целевого закона.

### 9.2 Proper ensemble objectives

После FM pretraining допустим end-to-end ensemble fine-tune:

\[
L_{proper}=
\lambda_C L_{fairCRPS}
+\lambda_E L_{energy}
+\lambda_V L_{variogram}
+\lambda_B L_{Brier}.
\]

- Fair CRPS строго идентифицирует только univariate marginals. Unbiased
  finite-ensemble form требует не менее двух members; практически нужно хотя бы
  четыре.
- Energy score является строго proper для полного конечномерного закона при
  конечном первом моменте, но в высокой размерности имеет низкую чувствительность
  к copula errors.
- Variogram score proper, но не strictly proper; он фиксирует выбранные
  increment moments, а не полный закон.
- Brier strictly proper только для указанного события: open water, ice presence,
  high-SIC threshold.
- Rank uniformity необходима, но не достаточна и не должна быть единственным
  training loss.

Практична смесь global area/volume energy score и patch/multiscale scores. Один
pixelwise CRPS не гарантирует хороших individual maps.

### 9.3 Levelwise calibration

Если `D_l X_l = X_{l-1}` samplewise, следующий уровень не меняет никаких
statistics, измеримых только через parent field. Поэтому уже достигнутая
coarse calibration сохраняется точно.

Но marginal rank calibration каждого уровня отдельно не гарантирует правильный
joint law. Для корректной полной trajectory нужны правильные conditional
kernels, spatial/temporal energy/variogram diagnostics и конечная native
проверка.

## 10. Teacher forcing и distribution shift

### 10.1 Формальная проблема

При supervised/CFM training уровня `l` condition равен truth parent
`X*_(l-1)`. На inference он равен sampled `Xhat_(l-1)`. Поэтому kernel обучен
под распределением `P*_(l-1)`, но применяется под `Phat_(l-1)`.

В асимптотически корректной модели `Phat=P*`, и mismatch исчезает. В конечной
модели он может накапливаться с числом уровней.

Особенно важно: если sampled parent budget не совпадает с truth parent budget,
native truth не лежит на fiber sampled parent. Поэтому не существует честного
обычного paired regression target, который одновременно:

- равен observed truth;
- точно сохраняет generated parent.

Нельзя silently rescale truth children к generated budget и затем утверждать,
что это sample из истинного conditional law. Это может быть robustness
augmentation, но не exact likelihood training.

### 10.2 Честный протокол

Рекомендуется:

1. Teacher-forced CFM pretraining каждого conditional kernel на истинных
   parents. Это статистически корректная факторизация target law.
2. Каждый upstream stage обучается до frozen validation gate до добавления
   следующего.
3. Затем все stages запускаются в inference mode от sampled parents.
4. Проводится end-to-end proper-score fine-tune против настоящего native truth.
   Proper score сравнивает распределение generated ensemble с одной реализацией
   truth и не требует, чтобы truth находился на fiber каждого sampled parent.
5. Для экономии сначала fine-tune только два соседних последних stages или
   terminal blocks; затем при необходимости размораживать upstream.

Дополнительная conditioning corruption, оценённая по out-of-fold upstream
errors, допустима как заранее объявленная robustness ablation. Она не должна
выдаваться за exact conditional maximum likelihood.

### 10.3 Propagation bound

Пусть `P_l` — истинный закон уровня, `Phat_l` — модельный, `K_l` и `Khat_l` —
истинный и модельный kernels. Для total variation стандартное telescoping и
contraction Markov kernels дают грубую оценку

\[
\|\hat P_L-P_L\|_{TV}
\le
\|\hat P_0-P_0\|_{TV}
+\sum_{l=1}^{L}
\sup_x\|\hat K_l(x,\cdot)-K_l(x,\cdot)\|_{TV}.
\]

Это upper bound, не прогноз реальной ошибки. Он показывает, почему добавление
levels не бесплатно: ошибки kernels могут суммироваться. Для Wasserstein bound
потребуются Lipschitz constants kernels; при constants больше единицы ошибка
может усиливаться.

## 11. Допустимые scale-wise corrections

Нетривиальная calibration correction без оценки по данным невозможна. Identity
— единственное полностью нефитированное преобразование. Все параметры ниже
должны оцениваться только на train или заранее выделенном calibration split и
замораживаться до validation/test evaluation.

### 11.1 Наиболее безопасные

1. Train-only standardization/Gaussianization continuous coordinates.
2. Train-only whitening/recoloring scale-band base noise.
3. Train-only bias centring coarse area/volume coordinates.
4. Одна заранее объявленная temperature для allocation logits на уровне.
   Decoder после temperature всё ещё точно сохраняет parent budget.
5. Nullspace correction child allocations, реализованная внутри constrained
   parameterization; она не меняет parent totals.
6. Proper-score fine-tuning generator parameters на train/calibration data.

Logit temperature может менять child spread и conditional copula, поэтому даже
при сохранении budget она требует полного proper/spatial gate.

### 11.2 Ограниченно допустимые baselines

- scalar anomaly scale вокруг ensemble mean;
- monotone marginal quantile mapping;
- affine EMOS-like correction area/volume budgets.

Они должны быть cross-fitted/frozen и называться post-processing baselines. Они
не доказывают native generative calibration.

### 11.3 Что ломает закон или conservation

- независимый pixelwise rank/quantile mapping разрушает spatial copula;
- независимая перестановка members между уровнями разрушает trajectory identity;
- clipping child fields после additive correction меняет parent sums;
- изменение parent ensemble при сохранении уже сгенерированного downstream
  residual создаёт condition mismatch;
- отдельная SIC correction без согласованного volume update нарушает
  `(A=0)<=> (V=0)` и меняет SIT;
- independent inflation каждого lead разрушает temporal copula;
- проекция только ensemble-average conservation допускает memberwise нарушение;
- validation-selected набор многочисленных temperatures является подгоном,
  даже если каждая отдельная операция проста.

Если parent correction применена, downstream stages должны быть заново sampled
условно на исправленном parent.

## 12. Falsifiable experiments

### 12.1 Общий протокол

До запуска:

- семантика 24 slices должна быть зафиксирована;
- validation dates и archive slice rule заморожены;
- test-2023 остаётся закрыт;
- один GPU максимум;
- одинаковые case/member seeds для paired comparisons;
- compute сравнивается по measured GPU time, forward-call FLOPs и числу
  параметров, а не только по optimizer updates.

Для screening: не менее 48 сезонно распределённых validation-2022 dates и 20
members. Для publication — предпочтительно все допустимые validation dates;
не считать тысячи пространственно зависимых pixels независимыми replicates.
Confidence intervals строятся по датам и отдельно по temporal blocks.

### 12.2 Experiment A: `20x16-only`

Цель — проверить только гипотезу, что extreme coarse law способен моделировать
global area/volume trajectory.

Сравнения:

- persistence, restricted до `20x16`;
- restriction текущего двухуровневого ensemble;
- новый `20x16` joint generator.

Обязательные metrics по lead и `A/V`:

- fair CRPS;
- ordinary CRPS;
- spread/skill;
- randomized-rank histogram с absolute uniformity interval;
- joint energy score по трём горизонтам;
- total-domain ice area/volume distribution;
- empty/full coarse-cell Brier scores.

Stop/reject:

- generator не превосходит persistence по fair CRPS;
- spread/skill остаётся вне заранее фиксированного acceptable interval,
  например `[0.8,1.2]`, и paired uncertainty исключает 1;
- rank discrepancy не лучше restricted current cascade;
- area/volume bias значимо хуже baseline;
- individual members показывают нефизические скачки между leads.

Прохождение A не доказывает полезность полной пятиуровневой цепочки.

### 12.3 Experiment B: `80->160->320` против `160->320`

Это первый реалистичный end-to-end эксперимент. Сравнение должно иметь:

- одинаковый data envelope;
- одинаковый FM/proper-score protocol;
- matched total GPU time или FLOP budget;
- одинаковое число members и solver accuracy;
- одинаковый structured physical decoder.

Предлагаемый заранее фиксированный gate:

1. Samplewise area/volume conservation на каждом parent имеет FP64 residual
   `<1e-10` и FP32 replay residual `<1e-6`.
2. Land budgets строго zero; все native `0<=SIC<=Amax`, `SIT>=0`, joint-zero
   support точен.
3. Fair CRPS ни на одном обязательном масштабе/lead/field не хуже baseline
   более чем на 1%.
4. Минимум на одном масштабе `80` или `160` fair CRPS лучше не менее чем на 3%
   с paired-date interval ниже нуля.
5. Absolute rank discrepancy на `80` и `160` уменьшается не менее чем на 20%; ни
   один mandatory native rank histogram не ухудшается более чем на 10%.
6. Native joint energy не хуже более чем на 1–2%; variogram/spectrum family не
   хуже более чем на 2%.
7. Mean member/truth roughness ratio для каждого lead/field находится в заранее
   заданном диапазоне, например `[0.8,1.2]`, либо статистически ближе к 1, чем
   baseline.
8. Native fair CRPS не ухудшен; иначе scale improvement не компенсирует
   практический проигрыш.
9. Measured peak memory помещается в текущий one-GPU envelope; wall time не
   превышает заранее фиксированный множитель baseline.

Пороговые числа являются предложением и должны быть заморожены до результатов,
а не изменены после них.

### 12.4 Experiment C: `40->80->160->320`

Запускать только если B проходит. Он проверяет, добавляет ли отдельный large-scale
geometry level пользу сверх `80` base. Сравнивать B и C при равном compute.

Отклонить дополнительный `40` level, если:

- coarse rank/CRPS не улучшаются;
- native score ухудшается;
- teacher-forcing sensitivity растёт;
- runtime растёт без статистически различимого выигрыша.

### 12.5 Full `20->40->80->160->320`

Запускать только если отдельно прошли A и C. Extreme `20` level должен улучшать
именно domain/global budget law; если он лишь повторяет `40` distribution, это
архитектурная сложность без научной функции.

### 12.6 Обязательные ablations

1. `A,V` против raw `SIC,SIT` hierarchy.
2. Exact constrained allocation против additive nullspace residual.
3. Joint area-then-volume generator против независимых heads.
4. Joint three-lead output против independent lead outputs.
5. Teacher-forced-only против teacher-forced + end-to-end proper fine-tune.
6. White против train-only colored multiscale base noise.
7. Два, три и четыре spatial levels при matched compute.
8. Binary-active recursive mask против корректного cumulative area weighting —
   только как bug-demonstration, не как допустимый candidate.

## 13. Compute, память и скорость

При одинаковой ширине сети сумма площадей пяти dyadic grids относительно native:

\[
1+1/4+1/16+1/64+1/256\approx1.332.
\]

Но convolutional cost масштабируется также примерно как квадрат channel width.
Текущий `160x128` coarse model шире `320x256` fine model, поэтому его FLOPs
могут быть сопоставимы с fine stage несмотря на вчетверо меньшую площадь.

Разумные стартовые widths:

```text
20x16:    64
40x32:    64
80x64:    48-64
160x128:  32-48
320x256:  32
```

Stages следует обучать последовательно. При sequential sampling можно держать
на GPU только активную сеть; тогда peak activation memory определяется native
stage, а не суммой уровней. Нынешний coarse-budget preflight использовал около
4.9 GiB, но это не является измерением полной proposed chain. Реальную память
надо получить отдельным zero-update preflight.

Ожидание, а не гарантия:

- `80->160->320` может иметь training cost порядка `1.2-1.7` compact native
  model;
- `40->80->160->320` — порядка `1.3-2.0`;
- по сравнению с текущим wide `160->320` узкий multilevel cascade может быть
  сопоставим по FLOPs;
- inference замедлится из-за отдельных ODE solves и launch overhead, даже если
  FLOPs невелики.

Меньшее число solver steps на fine allocations является отдельной гипотезой и
требует paired solver-sensitivity. Нельзя заранее считать fine dynamics
«простыми» и без проверки сокращать integration.

Главный compute risk — не memory, а multiplication числа model evaluations на
число levels, members и ODE stages. Для 20-member evaluation низкие уровни
дёшевы, но native refinement всё равно повторяется для каждого member.

## 14. Пакет утверждений для статьи

### Proposition 1: area-volume representation

На множестве

\[
\{(A,H):A=0,H=0\}\cup\{(A,H):0<A\le Amax,H>0\}
\]

отображение `(A,H)->(A,V=AH)` является биекцией на соответствующее множество
`(A,V)` с обратным `H=V/A` на positive branch и `H=0` на zero atom.

Это доказуемое утверждение.

### Lemma 2: unique capped-simplex shift

Для положительных capacities и interior total logistic-shift equation имеет
единственный конечный root. Endpoint и zero-capacity исключения описаны в
разделе 5.

Это доказуемое утверждение.

### Proposition 3: samplewise conservation

Если каждый decoder удовлетворяет `D_l T_l(parent,q)=parent`, то любой
сгенерированный member сохраняет area/volume budgets на всех более грубых
уровнях по индукции.

Это доказуемое утверждение.

### Corollary 4: inheritance of coarse observables

Для любой измеримой функции `f` parent field:

\[
f(D_l X_l^{(m)})=f(X_{l-1}^{(m)})
\]

samplewise. Следовательно, downstream refinement не меняет ensemble
distribution любого уже определённого coarse observable.

Это доказуемое утверждение; оно не означает, что исходный parent был хорошо
откалиброван.

### Proposition 5: exact hierarchical factorization

При существовании regular conditional distributions nested deterministic
restrictions дают точную product factorization по условным kernels. Если base
law и все kernels совпадают с истинными, конечный law совпадает с target.

Это стандартное доказуемое утверждение.

### Proposition 6: support preservation

Если area allocation лежит в capped simplex, volume allocation неотрицательна и
поддержана только на positive-area children, то decoded native state выполняет
SIC bounds, SIT nonnegativity, joint zero support и parent area/volume
conservation.

Это доказуемое утверждение при exact arithmetic; численная реализация требует
residual tolerances.

### Proposition 7: TV error accumulation bound

Для exact conditional kernels и их approximations действует telescoping upper
bound раздела 10.3.

Это доказуемое общее утверждение, но sup-kernel errors практически неизвестны и
не оцениваются обычным rank histogram.

### Что остаётся эвристикой

Не являются теоремами и требуют эксперимента:

- что multilevel model лучше откалиброван двухуровневого;
- что `20x16` является полезным base level;
- что area/volume coordinates легче обучить;
- что shared multiscale noise улучшает copula;
- что узкие несколько сетей быстрее одной широкой;
- что end-to-end proper fine-tune устранит teacher-forcing mismatch;
- что exact conservation улучшит fair CRPS;
- что предлагаемый метод даст публикационный результат.

## 15. Итоговый приоритет и условия отказа

### Реализовывать первым

1. Audit grid-cell area и рекурсивной mask semantics.
2. CPU-only reference primitives для area/volume restriction и conservation на
   partial-water cells.
3. FP64 capped-simplex decoder с exhaustive small-block tests, включая все
   endpoint/zero-capacity/one-active-child случаи.
4. `20x16-only` base-law feasibility как дешёвый научный screen.
5. Если coarse screen не отрицателен — `80->160->320` end-to-end candidate с
   matched-compute control `160->320`.
6. Teacher-forced CFM pretraining, затем ограниченный end-to-end proper-score
   fine-tune.
7. Только после прохождения gates — добавление `40` и затем `20` в полную цепь.

### Не делать

- не строить сразу пять полноценных сетей;
- не переносить recursive binary active mask вместо water area;
- не повторять pixel-unshuffle до 2816 conditioning channels;
- не генерировать 2x2 blocks независимо;
- не калибровать каждый pixel/lead отдельным inflation;
- не оставлять downstream residual после изменения parent;
- не выдавать rescaled truth allocations за exact conditional targets;
- не выбирать число levels, temperatures или solver steps по test-2023;
- не считать прохождение conservation tests доказательством calibration.

### Отклонить идею, если

- `20x16-only` не моделирует global area/volume лучше persistence/current
  restricted baseline;
- трёхуровневый вариант не улучшает ни одного coarse proper/calibration metric
  при равном compute;
- native fair CRPS или joint/spatial scores ухудшаются за допустимый порог;
- roughness/спектр остаются хуже truth;
- teacher-forcing sensitivity растёт с уровнем;
- дополнительные уровни дают только повторение уже сохранённых budgets;
- runtime роста не оправдан statistically supported gain.

Наиболее сильный потенциальный claim — не «больше resolution stages», а
`mixed-support, area-volume-conserving probabilistic hierarchy with scale-wise
calibration diagnostics`. Этот claim допустим только после парного
validation-эксперимента; математическая корректность конструкции сама по себе
не обещает улучшения калибровки.
