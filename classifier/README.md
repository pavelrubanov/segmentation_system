# Классификатор листьев Sphagnum

Двухступенчатая классификация RGB-кропа листа на 3 класса:
**good_leaf / bad_leaf / non_leaf**.

```
crop  →  MobileNetV3-Small (frozen, ImageNet)  →  576-d вектор  →  XGBoost  →  label
```

MobileNet достаёт общие визуальные признаки; XGBoost учится отображать их в
нужные 3 класса для конкретного типа листа (branch / capillifolium / girgensohnii).

---

## Структура

```
classifier/
├── __init__.py       # re-export LeafClassifier, find_available_leaf_types, get_classifier
├── variants.py       # 6 детерм. аугментаций (hflip × {neutral, brighter, darker})
├── mobilenet.py      # MobileNetV3 feature extractor + сборка features.npz
├── train.py          # обучение XGBoost на features.npz
├── predict.py        # LeafClassifier + get_classifier (синглтон по weights_path)
├── features.npz      # собранные признаки (создаётся один раз)
└── results/          # JSON-отчёты обучения (per-class метрики, CV-фолды)

models/               # рядом с classifier/ (не внутри)
├── mobile_sam.pt     # веса сегментатора (уже лежит)
└── model_<type>.pkl  # обученные XGBoost (создаётся train.py)
```

Пакет намеренно держится **независимым** от `src/core/`: свой venv (`classifier/.venv/`),
не импортирует из основного приложения. Если константа должна совпадать с приложением
(например, `_CROP_INFIX = "leaf_crop"` из `core/file_naming.py`) --- она дублируется
в `mobilenet.py` с пометкой.

## Полный workflow

### 1. Подготовка данных

Положить **препроцессированные** crop-PNG в папки по классам. Это те, что
приложение сохраняет (из `masks_buffer`), они уже прошли `transform_leaf`
(крупнейшая компонента + PCA-поворот, апекс вверху).

```
data/
├── bad_leaf/
│   ├── 0192_leaf_crop_0001.png
│   └── ...
├── good_leaf_branch/
├── good_leaf_capillifolium/
├── good_leaf_girgensohnii/
└── non_leaf/
```
Файлы должны содержать подстроку `_leaf_crop_` в имени.
Чёрный фон = не-лист.

### 2. Сборка признаков — один раз

```bash
cd segmentation_system
python -m classifier.mobilenet --data data --out classifier/features.npz
```

Что происходит:
1. Сканирует `data/*/[*_leaf_crop_*.png]`.
2. Для каждого PNG применяет 6 детерминистских вариантов:
   `{orig, hflip} × {neutral, brighter, darker}`
3. Letterbox в 512×512 (для MobileNet).
4. Batched GPU-инференс → 576-мерный вектор.

Препроцессинг (крупнейшая компонента + PCA-поворот) уже сделан приложением
при сохранении crop-ов --- здесь повторно НЕ применяется.

Результат: `classifier/features.npz`, шаблон строк
`file_i * 6 + variant_j`.

Параметры:
- `--batch-size 32` — размер GPU-батча (уменьшить при OOM)
- `--n-jobs -1` — процессов для CPU-препроцессинга
- `--classes good_leaf_branch bad_leaf` — только указанные папки

Типовое время: **~130 секунд** на 1200 PNG (RTX 5080, 20 CPU-ядер).

### 3. Обучение классификатора

**Hold-out 60/20/20** (по умолчанию):
```bash
python -m classifier.train --leaf-type capillifolium
```

**5-fold CV** для надёжной оценки:
```bash
python -m classifier.train --leaf-type capillifolium --cv 5
```

Что происходит:
1. Читает `classifier/features.npz` через `FeaturesDB`.
2. Находит сэмплы для указанного типа: `good_leaf_<type>` как good_leaf + общие `bad_leaf` и `non_leaf`.
3. Стратифицированный сплит по исходным PNG (не по строкам X — иначе утечка через варианты):
   - Hold-out: 60/20/20 train/val/test
   - CV: `K` стратифицированных фолдов
4. Материализация:
   - **train** — все 6 вариантов на файл (расширение в 6×)
   - **val/test** — только identity (1×), без аугментации
5. XGBoost с class_weight=balanced (компенсация дисбаланса).
6. Сохранение:
   - `models/model_<type>.pkl` — веса
   - `classifier/results/model_<type>.json` — метрики

Параметры:
- `--n-trees 300` — количество деревьев (default 300)
- `--max-depth 6` — глубина (default 6)
- `--lr 0.1` — learning rate (default 0.1)
- `--out <path>` — пользовательский путь для .pkl

Типовое время: **~30 секунд** (фичи уже посчитаны, остаётся только fit XGBoost).

### 4. Использование в приложении

```python
from classifier import find_available_leaf_types, get_classifier
from core.leaf_pipeline import transform_leaf

# 1. Найти доступные модели
types = find_available_leaf_types("models")
# [('branch', 'models/model_branch.pkl'), ...]

# 2. Достать классификатор (синглтон по resolved-пути ---
#    повторные вызовы вернут тот же экземпляр).
clf = get_classifier("models/model_capillifolium.pkl")

# 3. Препроцессинг (тот же, что при обучении)
masked = image.copy()                  # RGB H×W×3
masked[mask == 0] = 0                  # чёрный фон вне маски
crop = transform_leaf(masked)          # PCA-поворот + bbox

# 4. Классификация
label, confidence = clf.classify(crop)
# label ∈ {"good_leaf", "bad_leaf", "non_leaf"}
# confidence ∈ [0, 1]

# Батч (эффективнее для многих кропов):
results = clf.classify_batch([crop1, crop2, crop3])
```

**Важно:** `clf.classify(crop)` ожидает **препроцессированный** кроп из
`transform_leaf`. Если подать сырую картинку --- качество деградирует.

`get_classifier(path)` кэширует загруженные модели в module-level словаре по
resolved-пути --- в одной сессии один и тот же `.pkl` распаковывается ровно один
раз. Прямой `LeafClassifier(path)` всё ещё доступен, если по какой-то причине
кэш нежелателен.

---

## Как добавить новый тип листа

1. Создать папку `data/good_leaf_<new_type>/` с img_crop-PNG.
2. Пересобрать `features.npz`:
   ```bash
   python -m classifier.mobilenet --data data --out classifier/features.npz
   ```
3. Обучить модель:
   ```bash
   python -m classifier.train --leaf-type <new_type> --cv 5
   ```
4. Готово. `find_available_leaf_types("models")` автоматически увидит новую модель.

---

## Формат `features.npz`

```python
import numpy as np
data = np.load("classifier/features.npz", allow_pickle=False)

X             = data["X"]             # float32 [N_files * 6, 576]
file_names    = data["file_names"]    # str     [N_files]
class_dirs    = data["class_dirs"]    # str     [N_files]
n_variants    = int(data["n_variants"])  # 6
```

Выбрать строку для файла `i`, варианта `v`:  `X[i * n_variants + v]`.

## Формат `model_<type>.pkl`

```python
import pickle
blob = pickle.load(open("models/model_branch.pkl", "rb"))
blob = {
    "model":     XGBClassifier,                          # обученная модель
    "classes":   ["good_leaf", "bad_leaf", "non_leaf"],  # в порядке индексов модели
    "leaf_type": "branch",
}
```

## Формат `results/model_<type>.json`

```json
{
  "leaf_type": "branch",
  "n_trees": 300, "max_depth": 6, "lr": 0.1,
  "val_report":  { "accuracy": 0.978, "per_class": [...] },
  "test_report": { "accuracy": 0.978, "per_class": [...] }
}
```

Для CV-отчёта (`cv5_<type>.json`) дополнительно:
- `fold_results` — метрики по каждому fold'у
- `mean_accuracy`, `std_accuracy`
- `errors` — список файлов, на которых модель ошиблась

---

## Требования

- Python 3.10+
- torch, torchvision (CPU или CUDA)
- xgboost, scikit-learn, joblib
- opencv-python, Pillow, numpy

См. `segmentation_system/src/requirements.txt`.

Все вычисления работают и на CPU (тогда сборка features.npz медленнее, ~5-10 мин).
Inference одной картинки в приложении: ~50-100 мс на CPU.
