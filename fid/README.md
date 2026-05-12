# Inception FID Guide

В этой папке теперь лежит новый FID-пайплайн под твой текущий подход:

- берём `.npy` sea-ice поля
- рендерим их в `RGB` через фиксированный `matplotlib` colormap
- прогоняем получившиеся картинки через `InceptionV3`
- считаем Fréchet distance в feature space Inception

Старый доменный `VAE-FID` из этой папки убран.

## Что лежит в папке

- `fid/inception_fid.py`
  Ядро расчёта:
  - загрузка `.npy`
  - рендер `field -> RGB`
  - извлечение Inception features
  - расчёт `mu`, `Sigma` и FID

- `fid/compute_folder_fid.py`
  CLI-скрипт для подсчёта FID между двумя папками `.npy`.

- `fid/export_concat_samples.py`
  CLI-скрипт для генерации `.npy`-сэмплов из `concat`-модели, чтобы потом сразу подать их в `compute_folder_fid.py`.

## Что именно считается

Схема такая:

1. Берутся real и fake поля из двух папок `.npy`.
2. Каждое поле преобразуется в `RGB` через фиксированную colormap-конфигурацию.
3. Получившееся изображение ресайзится до `299 x 299`.
4. Из `InceptionV3` берётся penultimate feature vector размерности `2048`.
5. Для real и fake считаются:
   - средний вектор `mu`
   - ковариация `Sigma`
6. Потом считается Fréchet distance между этими двумя гауссианами.

Это уже именно Inception-based FID, а не VAE-based proxy.

## Важная оговорка

Это не "канонический tensorflow/pytorch-fid backend".
Здесь используется `torchvision InceptionV3` и твой собственный `field -> RGB` mapping.

Поэтому сравнения корректны только если у экспериментов одинаковы:

- одна и та же папка `real`
- один и тот же `render-mode`
- те же `cmap`
- те же `vmin/vmax`
- тот же `resize`
- те же Inception weights

## Зависимости

Для запуска этого пайплайна нужны:

- `torch`
- `torchvision`
- `matplotlib`
- `numpy`
- `tqdm`

Если `torchvision` weights не скачаны, `torchvision` может попытаться скачать их при первом запуске.
Если не хочешь этого, передай локальный `--weights-path`.

## Быстрый workflow

### Вариант 1. У тебя уже есть fake `.npy`

```bash
python -m fid.compute_folder_fid \
  --real-dir /mnt/sciml/a.sadreev/sea_ice_data/valid \
  --fake-dir /path/to/generated_samples \
  --render-mode channel0 \
  --channel0-cmap viridis \
  --channel0-vmin 0.0 \
  --channel0-vmax 1.0
```

Это дефолтный и самый безопасный режим:

- берётся только `channel 0`
- он красится в `viridis`
- потом идёт в Inception

### Вариант 2. Сначала сгенерировать fake из concat

```bash
python -m fid.export_concat_samples \
  --output-dir /tmp/concat_fake_valid \
  --num-timesteps 50 \
  --method euler \
  --batch-size 4
```

Потом посчитать FID:

```bash
python -m fid.compute_folder_fid \
  --real-dir /mnt/sciml/a.sadreev/sea_ice_data/valid \
  --fake-dir /tmp/concat_fake_valid \
  --render-mode channel0 \
  --channel0-cmap viridis \
  --channel0-vmin 0.0 \
  --channel0-vmax 1.0
```

## Режимы рендера

Есть три режима.

### `channel0`

Рендерится только первый канал:

```bash
--render-mode channel0
```

Обычно это лучший стартовый вариант, если хочешь FID по концентрации льда.

### `channel1`

Рендерится только второй канал:

```bash
--render-mode channel1 \
--channel1-cmap magma \
--channel1-vmin 0.0 \
--channel1-vmax 5.0
```

### `blend`

Оба канала рендерятся разными colormap, затем смешиваются:

```bash
--render-mode blend \
--channel0-cmap viridis \
--channel1-cmap magma \
--channel0-vmin 0.0 \
--channel0-vmax 1.0 \
--channel1-vmin 0.0 \
--channel1-vmax 5.0 \
--blend-alpha 0.6
```

Тут `blend-alpha` это вес рендера `channel0`.

## Если входы нормализованы

Если `.npy` уже лежат в нормализованном виде, передай:

```bash
--input-normalized \
--stats-json /path/to/stats.json
```

Тогда перед рендером будет сделана денормализация.

Для real/fake из обычного датасета и для файлов из `export_concat_samples.py` это обычно не нужно.

## Реюз статистик real

Один раз сохраняешь real stats:

```bash
python -m fid.compute_folder_fid \
  --real-dir /mnt/sciml/a.sadreev/sea_ice_data/valid \
  --only-real-stats \
  --real-stats-out /tmp/real_inception_stats.npz \
  --render-mode channel0 \
  --channel0-cmap viridis \
  --channel0-vmin 0.0 \
  --channel0-vmax 1.0
```

Потом используешь их для разных fake:

```bash
python -m fid.compute_folder_fid \
  --real-dir /mnt/sciml/a.sadreev/sea_ice_data/valid \
  --fake-dir /tmp/concat_fake_valid \
  --real-stats-in /tmp/real_inception_stats.npz \
  --render-mode channel0 \
  --channel0-cmap viridis \
  --channel0-vmin 0.0 \
  --channel0-vmax 1.0
```

## Практические рекомендации

### 1. Держи mapping фиксированным

Самое важное здесь не Inception сам по себе, а стабильный `field -> RGB`.

Если поменял:

- `cmap`
- `vmin/vmax`
- `render-mode`
- `blend-alpha`

то числа FID уже нельзя честно сравнивать с предыдущими.

### 2. Начни с `channel0`

Для твоих полей разумный дефолт:

- `render-mode channel0`
- `channel0-cmap viridis`
- `channel0-vmin 0`
- `channel0-vmax 1`

Это самый интерпретируемый вариант.

### 3. Для `channel1` обязательно зафиксируй диапазон

Если второй канал не ограничен, не используй "авто-масштаб" от картинки к картинке.
Нужен один и тот же диапазон для real и fake.

### 4. Не опирайся только на FID

Для твоей задачи полезно смотреть вместе:

- FID
- rank histogram / spread-skill
- RMSE ансамблевого среднего
- визуальные карты mean/std

Потому что хороший FID не гарантирует хорошую calibration и наоборот.
