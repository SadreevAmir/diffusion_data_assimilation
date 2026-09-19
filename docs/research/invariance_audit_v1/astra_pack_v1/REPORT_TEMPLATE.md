# Отчёт контрольной точки

**Версия протокола:** 1.0  
**Дата:** UNKNOWN  
**Этап:** UNKNOWN  
**Статус:** STOP_REVIEW / BLOCKED / COMPLETED_AUTHORIZED_SCOPE  
**Commit / manifest hash:** UNKNOWN

## 1. Что действительно выполнено

Указать разрешённые и фактически выполненные шаги, количество обучений, число независимых датасетов, GPU/CPU и измеренный расход. Явно перечислить невыполненное. Планы не записывать в прошедшем времени.

## 2. Главный результат — не более 200 слов

Одно подтверждённое утверждение с числом, неопределённостью и точным объектом score. Один существенный отрицательный результат. Один главный открытый вопрос. Не использовать «всё подтвердилось» без ссылки на таблицу.

## 3. Аудит корректности

| Проверка | PASS / FAIL / UNKNOWN | Файл и hash | Влияние на вывод |
|---|---|---|---|
| Reference unit tests | UNKNOWN | UNKNOWN | UNKNOWN |
| Repo-score parity | UNKNOWN | UNKNOWN | UNKNOWN |
| Независимость банков и ключей | UNKNOWN | UNKNOWN | UNKNOWN |
| Точное описание forecast estimand | UNKNOWN | UNKNOWN | UNKNOWN |
| Oracle и его сходимость/формула | UNKNOWN | UNKNOWN | UNKNOWN |
| Solver sensitivity | UNKNOWN | UNKNOWN | UNKNOWN |
| Target masks и NaN | UNKNOWN | UNKNOWN | UNKNOWN |
| Совпадение marginals в copula-only опыте | UNKNOWN | UNKNOWN | UNKNOWN |
| Holdout access / история отбора | UNKNOWN | UNKNOWN | UNKNOWN |

`NOT_APPLICABLE` разрешён только с объяснением: например, P1 не содержит геофизических NaN. Отсутствующая проверка — не PASS.

## 4. Основные сравнения

Внести **отдельную строку на seed-блок**, затем агрегат. Единицу оценки указать в заголовке. Не сообщать только лучший seed/checkpoint.

| Run | Метод | Data seed | Шаг | M / L / m | Joint regret | Empirical CRPS | Mean MSE | V без truth | NFE |
|---|---|---|---|---|---|---|---|---|---|
| UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |

Парные контрасты:

| Контраст | Заранее primary? | Оценка разности | MC-неопределённость | Между-run неопределённость | Вывод |
|---|---|---|---|---|---|
| K1 − K0 | Да в P1 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| K1 − Kfresh | Вторичный training-контраст | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| Raw fixed-key − memberwise randomization | Да в P2 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| Измеренный gain − предсказанный по V | Да в P2 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |

Отдельно вывести raw/mean-matched distribution error. Если эффект объясняется только средним, прямо написать это.

## 5. Теория: проверенные предпосылки

| Утверждение | Какие предпосылки доказаны для опыта | Какие только предположены | Что реально проверено численно |
|---|---|---|---|
| T1 — эмпирическая цель | UNKNOWN | UNKNOWN | UNKNOWN |
| T3 — nuisance projection | UNKNOWN | UNKNOWN | UNKNOWN |
| T4 — finite-M budget | UNKNOWN | UNKNOWN | UNKNOWN |
| T5 — label-free V | UNKNOWN | UNKNOWN | UNKNOWN |
| T6 — information cost | UNKNOWN | UNKNOWN | UNKNOWN |
| T7 — cross-score | UNKNOWN | UNKNOWN | UNKNOWN |

Не ставить знак «доказано» напротив поведения нейросети, если доказана только формула риска при инвариантности.

## 6. Реестр научных утверждений

Статусы: `PROVEN_UNDER_ASSUMPTIONS`, `IMPLEMENTATION_TESTED`, `OBSERVED`, `NOT_TESTED`, `NOT_SUPPORTED_IN_THIS_SETTING`, `REFUTED_IN_THIS_SETTING`, `UNKNOWN`.

| Claim ID | Точная формулировка | Статус | Улика | Что изменит вывод |
|---|---|---|---|---|
| C1 | Текущая модель чувствительна к нерелевантному K | UNKNOWN | UNKNOWN | UNKNOWN |
| C2 | Чувствительность касается формы, а не только среднего | UNKNOWN | UNKNOWN | UNKNOWN |
| C3 | V предсказывает risk-gain при неизменном M | UNKNOWN | UNKNOWN | UNKNOWN |
| C4 | Метод полезен сверх канонического baseline | UNKNOWN | UNKNOWN | UNKNOWN |
| C5 | Генеративность нужна на льду сверх MSE + того же слоя | UNKNOWN | UNKNOWN | UNKNOWN |

## 7. Отрицательные исходы, баги и отклонения

Для каждого: что ожидали; что получилось; какая гипотеза опровергнута; что пока не установлено. Любой исправленный баг — с affected-run IDs и повторёнными сравнениями. Изменение дизайна — отдельной строкой с авторизацией. Нельзя удалить неудачный run из реестра.

## 8. Prior art

Кратко заполнить сравнительную таблицу из `REFERENCE_MAP.md`. Если новая формула оказалась частным случаем известной, написать это. Предлагаемый вклад не должен зависеть от замалчивания источника.

## 9. Воспроизведение

Команды, которые **реально существуют и были исполнены**, окружение/lockfile, относительные пути данных, источник каждого figure/table, как повторить smoke-test и primary comparison. Не включать секреты, токены и приватные учётные данные.

## 10. Решения для следующего ревью

Не более трёх точных вопросов, каждый с вариантами и фактической причиной. Например: «эффект только в среднем; подтверждать ли ρ-сетку или менять центральный claim?». Не спрашивать просто «что делать дальше?» при уже заданном протоколе.

Приложить `report.json`, manifests, `per_run.csv`, `per_checkpoint.csv`, тестовый лог и patch. Следующий дорогой этап до решения не запускать.

### Минимальный JSON

```json
{
  "protocol_version": "1.0",
  "checkpoint": null,
  "status": "STOP_REVIEW",
  "repo_commit": null,
  "completed_experiments": [],
  "blocked_experiments": [],
  "new_training_runs": 0,
  "primary_contrasts": [],
  "critical_audit_failures": [],
  "negative_findings": [],
  "holdout_accessed": [],
  "artifacts": [],
  "decisions_requested": []
}
```
